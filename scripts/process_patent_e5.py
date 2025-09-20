#!/usr/bin/env python3
"""
Script to process patent content with E5-RoPEBase embeddings and upload to database.
"""

import sys
import os
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
import logging
import csv
from io import StringIO
from sentence_transformers import SentenceTransformer
import torch

# Add parent directory to path to import LSH utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.LSH import LSHBinaryEncoder

# Database configuration
DB_CONFIG = {
    'host': '34.92.35.30',
    'port': '5432',
    'database': 'tadreamk-surveillance-dev',
    'user': 'postgres',
    'password': '?rLziM\\8>4,hq#1f'
}

BATCH_SIZE = 100000
EMBEDDING_BATCH_SIZE = 100  # Process embeddings in smaller batches
MAX_CONTENT_LENGTH = 4000  # Truncate content to fit context window

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_e5_model():
    """Load E5-RoPEBase model."""
    logger.info("Loading E5-RoPEBase model...")
    model = SentenceTransformer('dwzhu/e5rope-base')
    
    # Set extended context window for longer patent texts
    model.max_seq_length = 4096
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    return model

def create_e5_columns(conn):
    """Create the E5 embedding columns if they don't exist."""
    try:
        with conn.cursor() as cur:
            # Check if columns exist
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'patent_content' 
                AND column_name IN ('embedding_e5', 'binary_embedding_e5')
            """)
            
            existing_columns = {row[0] for row in cur.fetchall()}
            
            # Create embedding_e5 column if it doesn't exist
            if 'embedding_e5' not in existing_columns:
                logger.info("Creating embedding_e5 column...")
                # Assuming E5 model produces 768-dimensional embeddings
                cur.execute("""
                    ALTER TABLE patent_content 
                    ADD COLUMN embedding_e5 REAL[]
                """)
                logger.info("embedding_e5 column created successfully")
            else:
                logger.info("embedding_e5 column already exists")
            
            # Create binary_embedding_e5 column if it doesn't exist
            if 'binary_embedding_e5' not in existing_columns:
                logger.info("Creating binary_embedding_e5 column...")
                # Using 1536 bits for binary embedding as requested
                cur.execute("""
                    ALTER TABLE patent_content 
                    ADD COLUMN binary_embedding_e5 BIT(1536)
                """)
                logger.info("binary_embedding_e5 column created successfully")
            else:
                logger.info("binary_embedding_e5 column already exists")
            
            conn.commit()
                
    except Exception as e:
        logger.error(f"Error creating E5 columns: {e}")
        raise

def get_total_patent_content_count(conn):
    """Get total number of patents with content."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) 
            FROM patent_content 
            WHERE content IS NOT NULL 
            AND LENGTH(content) > 100
        """)
        return cur.fetchone()[0]

def bulk_update_e5_embeddings(conn, updates_data):
    """Fast bulk update for E5 embeddings using temporary table."""
    if not updates_data:
        return 0
    
    temp_table_name = "temp_patent_e5_updates"
    delim = ";"
    temp_buffer_size = "1000MB"
    
    # Prepare CSV data
    out = StringIO()
    writer = csv.writer(out, delimiter=delim)
    for patent_id, embedding_array, binary_hash in updates_data:
        # Convert numpy array to PostgreSQL array format
        embedding_str = '{' + ','.join(map(str, embedding_array)) + '}'
        writer.writerow([patent_id, embedding_str, binary_hash])
    
    try:
        with conn.cursor() as cur:
            # Set temp buffer size for better performance
            cur.execute(f"SET temp_buffers = '{temp_buffer_size}'")
            
            # Create temp table if not exists
            cur.execute(f"""
                CREATE TEMP TABLE IF NOT EXISTS {temp_table_name}
                (
                    patent_id TEXT,
                    embedding_e5_str TEXT,
                    binary_embedding_e5_str TEXT
                )
            """)
            
            # Clear temp table
            cur.execute(f"TRUNCATE {temp_table_name}")
            
            # Use COPY FROM to load data efficiently
            out.seek(0)
            cur.copy_from(out, temp_table_name, sep=delim, 
                         columns=['patent_id', 'embedding_e5_str', 'binary_embedding_e5_str'])
            
            # Perform bulk update using UPDATE...FROM
            cur.execute(f"""
                UPDATE patent_content 
                SET 
                    embedding_e5 = t.embedding_e5_str::REAL[],
                    binary_embedding_e5 = t.binary_embedding_e5_str::bit(1536)
                FROM {temp_table_name} AS t
                WHERE patent_content.patent_id = t.patent_id
            """)
            
            updated_count = cur.rowcount
            conn.commit()
            return updated_count
            
    except Exception as e:
        logger.error(f"Error in E5 bulk update using temp table: {e}")
        conn.rollback()
        raise

def process_patents_e5_batch(conn, model, lsh_encoder, offset, batch_size):
    """Process a batch of patents with E5 embeddings."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch batch of patents with content, sorted by patent_id
            cur.execute("""
                SELECT patent_id, content 
                FROM patent_content 
                WHERE content IS NOT NULL 
                AND LENGTH(content) > 100
                ORDER BY patent_id
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            
            patents = cur.fetchall()
            
            if not patents:
                return 0
            
            logger.info(f"Processing {len(patents)} patents for E5 embeddings...")
            
            # Process patents in smaller embedding batches
            updates_data = []
            
            for i in range(0, len(patents), EMBEDDING_BATCH_SIZE):
                embedding_batch = patents[i:i + EMBEDDING_BATCH_SIZE]
                
                # Prepare texts for embedding
                texts = []
                batch_patent_ids = []
                
                for patent in embedding_batch:
                    patent_id = patent['patent_id']
                    content = patent['content']
                    
                    if content is None or len(content.strip()) < 100:
                        logger.warning(f"Patent {patent_id} has insufficient content, skipping")
                        continue
                    
                    # Truncate content to fit context window
                    truncated_content = content[:MAX_CONTENT_LENGTH]
                    texts.append(truncated_content)
                    batch_patent_ids.append(patent_id)
                
                if not texts:
                    continue
                
                # Generate E5 embeddings for this sub-batch
                try:
                    embeddings = model.encode(texts, show_progress_bar=False)
                    
                    # Process each embedding
                    for j, embedding in enumerate(embeddings):
                        patent_id = batch_patent_ids[j]
                        
                        # Ensure embedding is numpy array
                        if not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding)
                        
                        # Generate LSH binary hash
                        binary_hash = lsh_encoder.encode(embedding)
                        
                        updates_data.append((patent_id, embedding, binary_hash))
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    continue
            
            # Use fast bulk update method
            if updates_data:
                updated_count = bulk_update_e5_embeddings(conn, updates_data)
                return updated_count
            else:
                return 0
            
    except Exception as e:
        logger.error(f"Error processing E5 batch at offset {offset}: {e}")
        conn.rollback()
        raise

def main():
    """Main processing function."""
    logger.info("Starting patent E5 processing...")
    
    # Load E5 model
    model = load_e5_model()
    
    # Get embedding dimension from a test
    test_embedding = model.encode(["test text"])
    embedding_dim = test_embedding.shape[1] if len(test_embedding.shape) > 1 else test_embedding.shape[0]
    logger.info(f"E5 model embedding dimension: {embedding_dim}")
    
    # Initialize LSH encoder for E5 embeddings
    logger.info(f"Initializing LSH encoder for {embedding_dim}-dimensional E5 embeddings to 1536-bit binary vectors...")
    lsh_encoder = LSHBinaryEncoder(
        input_dim=embedding_dim,
        output_bits=1536,
        seed=42
    )
    
    # Connect to database
    logger.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    try:
        # Create E5 columns if needed
        create_e5_columns(conn)
        
        # Get total count for progress bar
        total_patents = get_total_patent_content_count(conn)
        logger.info(f"Found {total_patents} patents with content to process")
        
        if total_patents == 0:
            logger.info("No patents with content found. Exiting.")
            return
        
        # Process patents in batches
        processed_count = 0
        offset = 0
        
        with tqdm(total=total_patents, desc="Processing patents with E5") as pbar:
            while offset < total_patents:
                batch_processed = process_patents_e5_batch(
                    conn, model, lsh_encoder, offset, BATCH_SIZE
                )
                
                if batch_processed == 0:
                    break
                
                processed_count += batch_processed
                offset += BATCH_SIZE
                pbar.update(batch_processed)
                
                logger.info(f"Processed {processed_count}/{total_patents} patents")
        
        logger.info(f"Successfully processed {processed_count} patents with E5 embeddings")
        
    except Exception as e:
        logger.error(f"Fatal error during E5 processing: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    main()

