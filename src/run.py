#!/usr/bin/env python3
"""
Startup script for the Research Paper Explorer FastAPI application.
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Set working directory to src
os.chdir(src_dir)

if __name__ == "__main__":
    print("Starting Research Paper Explorer...")
    print("This will:")
    print("1. Load the Qwen embedding model with 8-bit quantization")
    print("2. Load and preprocess embeddings and metadata")
    print("3. Perform clustering and PCA for visualization")
    print("4. Start the FastAPI server on http://localhost:8000")
    print()
    print("Note: Initial startup may take several minutes due to model loading.")
    print("Press Ctrl+C to stop the server.")
    print()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid reloading heavy model
        log_level="info"
    )
