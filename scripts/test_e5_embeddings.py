#!/usr/bin/env python3
"""
Test script to generate E5-RoPEBase embeddings for 10,000 patents.
"""

import sys
import os
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
import torch

# Database configuration
DB_CONFIG = {
    'host': '34.92.35.30',
    'port': '5432',
    'database': 'tadreamk-surveillance-dev',
    'user': 'postgres',
    'password': '?rLziM\\8>4,hq#1f'
}

TEST_LIMIT = 10000

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

def test_embeddings_generation():
    """Test E5 embedding generation on 10,000 patents."""
    logger.info("Starting E5 embedding test...")
    
    # Load model
    model = load_e5_model()
    
    # Connect to database
    logger.info("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch test patents with content
            logger.info(f"Fetching {TEST_LIMIT} patents with content...")
            cur.execute("""
                SELECT patent_id, content 
                FROM patent_content 
                WHERE content IS NOT NULL 
                AND LENGTH(content) > 100
                ORDER BY patent_id
                LIMIT %s
            """, (TEST_LIMIT,))
            
            patents = cur.fetchall()
            logger.info(f"Found {len(patents)} patents with content")
            
            if not patents:
                logger.error("No patents with content found!")
                return
            
            # Test different batch sizes to find optimal performance
            batch_sizes = [1, 10, 50, 100]
            
            for batch_size in batch_sizes:
                logger.info(f"\nTesting batch size: {batch_size}")
                
                # Test with first few patents
                test_patents = patents[:min(100, len(patents))]
                
                import time
                start_time = time.time()
                
                processed = 0
                for i in range(0, len(test_patents), batch_size):
                    batch = test_patents[i:i + batch_size]
                    texts = [patent['content'][:4000] for patent in batch]  # Truncate to fit context
                    
                    # Generate embeddings
                    embeddings = model.encode(texts, show_progress_bar=False)
                    processed += len(batch)
                
                end_time = time.time()
                elapsed = end_time - start_time
                rate = processed / elapsed
                
                logger.info(f"Batch size {batch_size}: {processed} patents in {elapsed:.2f}s ({rate:.2f} patents/s)")
                
                # Show embedding info for first patent
                if batch_size == 1:
                    sample_embedding = model.encode([test_patents[0]['content'][:4000]])
                    logger.info(f"Sample embedding shape: {sample_embedding.shape}")
                    logger.info(f"Sample embedding dtype: {sample_embedding.dtype}")
                    logger.info(f"Sample embedding range: [{sample_embedding.min():.6f}, {sample_embedding.max():.6f}]")
            
            # Test memory usage with larger batch
            logger.info(f"\nTesting memory usage with all {len(patents)} patents...")
            
            # Process in chunks to avoid memory issues
            chunk_size = 100
            all_embeddings = []
            
            with tqdm(total=len(patents), desc="Generating embeddings") as pbar:
                for i in range(0, len(patents), chunk_size):
                    chunk = patents[i:i + chunk_size]
                    texts = [patent['content'][:4000] for patent in chunk]
                    
                    # Generate embeddings for chunk
                    chunk_embeddings = model.encode(texts, show_progress_bar=False)
                    all_embeddings.extend(chunk_embeddings)
                    
                    pbar.update(len(chunk))
                    
                    # Log progress every 1000 patents
                    if (i + chunk_size) % 1000 == 0:
                        logger.info(f"Processed {min(i + chunk_size, len(patents))}/{len(patents)} patents")
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            logger.info(f"Embedding dimension: {all_embeddings[0].shape[0] if all_embeddings else 'N/A'}")
            
            # Save sample results for inspection
            if all_embeddings:
                sample_results = []
                for i in range(min(5, len(patents))):
                    sample_results.append({
                        'patent_id': patents[i]['patent_id'],
                        'content_length': len(patents[i]['content']),
                        'embedding_shape': all_embeddings[i].shape,
                        'embedding_sample': all_embeddings[i][:10].tolist()  # First 10 values
                    })
                
                logger.info("Sample results:")
                for result in sample_results:
                    logger.info(f"Patent {result['patent_id']}: content_len={result['content_length']}, "
                              f"embedding_shape={result['embedding_shape']}, "
                              f"sample_values={result['embedding_sample']}")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    test_embeddings_generation()

