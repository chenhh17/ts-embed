from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import BitsAndBytesConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from typing import List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Paper Explorer", description="Explore and search academic papers using embeddings")

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="src/templates")

# Global variables for preloaded data
embeddings_df = None
meta_df = None
model = None
pca_2d = None
clusters = None
cluster_centers = None
paper_coordinates = None

def load_model():
    """Load the Qwen embedding model with quantization"""
    global model
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False
    )
    
    logger.info("Loading Qwen embedding model...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={
            "attn_implementation": "flash_attention_2", 
            "device_map": "cuda:0",
            "quantization_config": quantization_config
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    logger.info("Model loaded successfully")

def load_data():
    """Load and preprocess the embeddings and metadata"""
    global embeddings_df, meta_df, pca_2d, clusters, cluster_centers, paper_coordinates
    
    # Load embeddings
    logger.info("Loading embeddings...")
    embeddings_df = pd.read_csv('pdf_project/db/embedded.csv', index_col=0)
    
    # Handle NaN values in embeddings
    logger.info("Cleaning embeddings data...")
    embeddings_df = embeddings_df.fillna(0.0)
    
    # Load metadata
    logger.info("Loading metadata...")
    meta_df = pd.read_csv('pdf_project/db/AMJ_2015_2025_meta.csv', skiprows=1)
    
    # Merge data on DOI
    logger.info("Merging data...")
    meta_df = meta_df.set_index('doi')
    
    # Get common DOIs
    common_dois = embeddings_df.index.intersection(meta_df.index)
    embeddings_df = embeddings_df.loc[common_dois]
    meta_df = meta_df.loc[common_dois]
    
    logger.info(f"Loaded {len(embeddings_df)} papers with embeddings and metadata")
    
    # Ensure embeddings are finite
    embeddings_array = embeddings_df.values
    embeddings_array = np.nan_to_num(embeddings_array, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Perform clustering
    logger.info("Performing clustering...")
    n_clusters = min(10, len(embeddings_df))  # Adjust based on data size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_array)
    cluster_centers = kmeans.cluster_centers_
    
    # Perform PCA for 2D visualization
    logger.info("Computing PCA for visualization...")
    pca_2d = PCA(n_components=2, random_state=42)
    paper_coordinates = pca_2d.fit_transform(embeddings_array)
    
    # Ensure coordinates are finite
    paper_coordinates = np.nan_to_num(paper_coordinates, nan=0.0)
    
    logger.info("Data loading and preprocessing completed")

def get_cluster_keywords(cluster_id: int, top_k: int = 10) -> List[str]:
    """Extract representative keywords for a cluster"""
    cluster_papers = meta_df[clusters == cluster_id]
    
    # Extract keywords from papers in this cluster
    all_keywords = []
    for keywords_str in cluster_papers['keywords'].dropna():
        if isinstance(keywords_str, str) and keywords_str.strip():
            # Split by common delimiters
            keywords = [k.strip() for k in keywords_str.replace(';', ',').split(',') if k.strip()]
            all_keywords.extend(keywords)
    
    # Count keyword frequency
    keyword_counts = {}
    for keyword in all_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Return top keywords
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, count in sorted_keywords[:top_k]]

def compute_keyword_distances():
    """Compute distances between papers and cluster keywords"""
    cluster_data = []
    
    for cluster_id in range(len(cluster_centers)):
        keywords = get_cluster_keywords(cluster_id)
        cluster_papers = meta_df[clusters == cluster_id]
        
        # Get paper coordinates for this cluster
        cluster_coords = paper_coordinates[clusters == cluster_id]
        cluster_center_2d = pca_2d.transform(cluster_centers[cluster_id:cluster_id+1])[0]
        
        # Handle NaN values in cluster center
        cluster_center_2d = np.nan_to_num(cluster_center_2d, nan=0.0)
        
        # Compute distances to cluster center
        distances = []
        paper_info = []
        
        for i, (doi, paper) in enumerate(cluster_papers.iterrows()):
            coord = cluster_coords[i]
            
            # Handle NaN values in coordinates
            coord = np.nan_to_num(coord, nan=0.0)
            distance = np.linalg.norm(coord - cluster_center_2d)
            
            # Ensure distance is finite
            if not np.isfinite(distance):
                distance = 0.0
            
            # Handle missing paper data
            title = str(paper['title']) if pd.notna(paper['title']) else "Untitled"
            
            distances.append(distance)
            paper_info.append({
                'doi': str(doi),
                'title': title,
                'x': float(coord[0]),
                'y': float(coord[1]),
                'distance': round(float(distance), 6)
            })
        
        cluster_data.append({
            'cluster_id': int(cluster_id),
            'keywords': keywords,
            'center_x': round(float(cluster_center_2d[0]), 6),
            'center_y': round(float(cluster_center_2d[1]), 6),
            'papers': paper_info
        })
    
    return cluster_data

@app.on_event("startup")
async def startup_event():
    """Load data and model on startup"""
    load_data()
    load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with cluster visualization"""
    cluster_data = compute_keyword_distances()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cluster_data": json.dumps(cluster_data),
        "total_papers": len(embeddings_df)
    })

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page"""
    return templates.TemplateResponse("search.html", {
        "request": request
    })

@app.get("/api/search")
async def search_papers(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return")
):
    """Search papers using cosine similarity"""
    if not query.strip():
        return {"results": []}
    
    try:
        # Encode the query
        logger.info(f"Searching for: {query}")
        query_embedding = model.encode([query], prompt_name="query", dtype=torch.bfloat16)
        
        # Compute cosine similarities using cleaned embeddings
        embeddings_array = embeddings_df.values
        embeddings_array = np.nan_to_num(embeddings_array, nan=0.0, posinf=1.0, neginf=-1.0)
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]
        
        # Handle NaN and infinite values
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doi = embeddings_df.index[idx]
            paper = meta_df.loc[doi]
            similarity = float(similarities[idx])
            
            # Ensure similarity is valid
            if not np.isfinite(similarity):
                similarity = 0.0
            
            # Handle missing or NaN values in paper data
            title = str(paper['title']) if pd.notna(paper['title']) else "Untitled"
            authors = str(paper['authors']) if pd.notna(paper['authors']) else "Unknown"
            abstract = str(paper['abstract']) if pd.notna(paper['abstract']) else "No abstract available"
            keywords = str(paper['keywords']) if pd.notna(paper['keywords']) else ""
            date = str(paper['date']) if pd.notna(paper['date']) else "Unknown"
            
            # Truncate abstract if too long
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            
            results.append({
                'doi': str(doi),
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'keywords': keywords,
                'date': date,
                'similarity': round(similarity, 6),  # Round to avoid precision issues
                'cluster': int(clusters[idx])
            })
        
        return {"results": results, "query": query}
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {"results": [], "query": query, "error": "Search failed. Please try again."}

@app.get("/api/paper/{doi:path}")
async def get_paper_details(doi: str):
    """Get detailed information about a specific paper"""
    try:
        paper = meta_df.loc[doi]
        paper_idx = embeddings_df.index.get_loc(doi)
        
        # Handle NaN values in coordinates
        coords = paper_coordinates[paper_idx]
        coords = np.nan_to_num(coords, nan=0.0)
        
        return {
            'doi': str(doi),
            'title': str(paper['title']) if pd.notna(paper['title']) else "Untitled",
            'authors': str(paper['authors']) if pd.notna(paper['authors']) else "Unknown",
            'abstract': str(paper['abstract']) if pd.notna(paper['abstract']) else "No abstract available",
            'keywords': str(paper['keywords']) if pd.notna(paper['keywords']) else "",
            'date': str(paper['date']) if pd.notna(paper['date']) else "Unknown",
            'source': str(paper['source']) if pd.notna(paper['source']) else "",
            'cluster': int(clusters[paper_idx]),
            'coordinates': {
                'x': round(float(coords[0]), 6),
                'y': round(float(coords[1]), 6)
            }
        }
    except KeyError:
        return {"error": "Paper not found"}
    except Exception as e:
        logger.error(f"Error getting paper details: {str(e)}")
        return {"error": "Failed to retrieve paper details"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
