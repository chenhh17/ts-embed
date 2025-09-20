#!/usr/bin/env python3
"""
Script to process patent embeddings with LSH hashing and upload binary embeddings.
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
INPUT_EMBEDDING_DIM = 768  # Input embedding dimension from database
OUTPUT_BINARY_BITS = 1536  # Target binary vector dimension

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_binary_embedding_column(conn):
    """Create the binary_embedding column if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            # Check if column exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'patent_abstracts' 
                AND column_name = 'binary_embedding'
            """)
            
            if not cur.fetchone():
                logger.info("Creating binary_embedding column...")
                cur.execute(f"""
                    ALTER TABLE patent_abstracts
                    ADD COLUMN binary_embedding BIT({OUTPUT_BINARY_BITS})
                """)
                conn.commit()
                logger.info("Binary embedding column created successfully")
            else:
                logger.info("Binary embedding column already exists")
                
    except Exception as e:
        logger.error(f"Error creating binary embedding column: {e}")
        raise

def get_total_patent_count(conn):
    """Get total number of patents with embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) 
            FROM patent_abstracts
            WHERE embedding IS NOT NULL
        """)
        return cur.fetchone()[0]

def bulk_update_using_temp_table(conn, updates_data):
    """Fast bulk update using temporary table and COPY FROM approach."""
    if not updates_data:
        return 0
    
    temp_table_name = "temp_patent_binary_updates"
    delim = ";"
    temp_buffer_size = "1000MB"
    
    # Prepare CSV data
    out = StringIO()
    writer = csv.writer(out, delimiter=delim)
    for patent_id, binary_hash in updates_data:
        writer.writerow([patent_id, binary_hash])
    
    try:
        with conn.cursor() as cur:
            # Set temp buffer size for better performance
            cur.execute(f"SET temp_buffers = '{temp_buffer_size}'")
            
            # Create temp table if not exists
            cur.execute(f"""
                CREATE TEMP TABLE IF NOT EXISTS {temp_table_name}
                (
                    patent_id TEXT,
                    binary_embedding_str TEXT
                )
            """)
            
            # Clear temp table
            cur.execute(f"TRUNCATE {temp_table_name}")
            
            # Use COPY FROM to load data efficiently
            out.seek(0)
            cur.copy_from(out, temp_table_name, sep=delim, columns=['patent_id', 'binary_embedding_str'])
            
            # Perform bulk update using UPDATE...FROM
            cur.execute(f"""
                UPDATE patent_abstracts 
                SET binary_embedding = t.binary_embedding_str::bit({OUTPUT_BINARY_BITS})
                FROM {temp_table_name} AS t
                WHERE patent_abstracts.patent_id = t.patent_id
            """)
            
            updated_count = cur.rowcount
            conn.commit()
            return updated_count
            
    except Exception as e:
        logger.error(f"Error in bulk update using temp table: {e}")
        conn.rollback()
        raise

def process_patents_batch(conn, lsh_encoder, offset, batch_size):
    """Process a batch of patents and compute LSH hashes."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch batch of patents with embeddings, sorted by patent_id
            cur.execute("""
                SELECT patent_id, embedding 
                FROM patent_abstracts 
                WHERE embedding IS NOT NULL
                ORDER BY patent_id
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            
            patents = cur.fetchall()
            
            if not patents:
                return 0
            
            # Process each patent in the batch
            updates_data = []
            for patent in patents:
                patent_id = patent['patent_id']
                embedding_data = patent['embedding']
                
                # Handle different embedding formats
                try:
                    if embedding_data is None:
                        logger.warning(f"Patent {patent_id} has null embedding, skipping")
                        continue
                    
                    # Debug: log the type and first few characters for the first patent
                    if patent_id == patents[0]['patent_id']:
                        logger.info(f"Debug - Embedding type: {type(embedding_data)}, Sample: {str(embedding_data)[:100]}...")
                    
                    # Convert to numpy array - handle different possible formats
                    if isinstance(embedding_data, (list, tuple)):
                        embedding = np.array(embedding_data)
                    elif isinstance(embedding_data, str):
                        # If it's a string representation, try to parse it
                        import ast
                        try:
                            embedding = np.array(ast.literal_eval(embedding_data))
                        except (ValueError, SyntaxError):
                            # Try alternative parsing methods if ast.literal_eval fails
                            import json
                            embedding = np.array(json.loads(embedding_data))
                    else:
                        # Assume it's already array-like
                        embedding = np.array(embedding_data)
                    
                    # Ensure it's 1D and has correct dimensions
                    embedding = embedding.flatten()
                    
                    if len(embedding) != INPUT_EMBEDDING_DIM:
                        logger.warning(f"Patent {patent_id} has embedding dimension {len(embedding)}, expected {INPUT_EMBEDDING_DIM}, skipping")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing embedding for patent {patent_id}: {e}")
                    continue
                
                # Compute LSH binary hash
                binary_hash = lsh_encoder.encode(embedding)
                updates_data.append((patent_id, binary_hash))
            
            # Use fast bulk update method
            updated_count = bulk_update_using_temp_table(conn, updates_data)
            return updated_count
            
    except Exception as e:
        logger.error(f"Error processing batch at offset {offset}: {e}")
        conn.rollback()
        raise

def main():
    """Main processing function."""
    logger.info("Starting patent LSH processing...")
    
    # Initialize LSH encoder
    logger.info(f"Initializing LSH encoder for {INPUT_EMBEDDING_DIM}-dimensional embeddings to {OUTPUT_BINARY_BITS}-bit binary vectors...")
    lsh_encoder = LSHBinaryEncoder(
        input_dim=INPUT_EMBEDDING_DIM,
        output_bits=OUTPUT_BINARY_BITS,  # Use 1536 bits as requested
        seed=42
    )
    
    # Connect to database
    logger.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    try:
        # Create binary embedding column if needed
        create_binary_embedding_column(conn)
        
        # Get total count for progress bar
        total_patents = 7118942
        logger.info(f"Found {total_patents} patents with embeddings to process")
        
        if total_patents == 0:
            logger.info("No patents with embeddings found. Exiting.")
            return
        
        # Process patents in batches
        processed_count = 0
        offset = 0
        
        with tqdm(total=total_patents, desc="Processing patents") as pbar:
            while offset < total_patents:
                batch_processed = process_patents_batch(
                    conn, lsh_encoder, offset, BATCH_SIZE
                )
                
                if batch_processed == 0:
                    break
                
                processed_count += batch_processed
                offset += BATCH_SIZE
                pbar.update(batch_processed)
                
                logger.info(f"Processed {processed_count}/{total_patents} patents")
        
        logger.info(f"Successfully processed {processed_count} patents with LSH binary embeddings")
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    main()
