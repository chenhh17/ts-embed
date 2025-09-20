from sentence_transformers import SentenceTransformer
import torch
from transformers import BitsAndBytesConfig
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=False
)

print("Loading Qwen embedding model...")
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={
        "attn_implementation": "flash_attention_2", 
        "device_map": "cuda:0",
        "quantization_config": quantization_config
    },
    tokenizer_kwargs={"padding_side": "left"},
)

print("Loading CSV data...")
df = pd.read_csv('/home/tadreamk-admin/alex/ts-embed/pdf_project/db/AMJ_2015_2025_meta.csv', skiprows=1)

print(f"Loaded {len(df)} rows from CSV")
print(f"Columns: {list(df.columns)}")

# Create title+abstract combinations
print("Preparing text data...")
texts = []
dois = []
valid_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    title = row.get('title', '')
    abstract = row.get('abstract', '')
    doi = row.get('doi', '')
    
    # Skip rows with missing critical data
    if pd.isna(title) or pd.isna(abstract) or pd.isna(doi):
        continue
    if str(title).strip() == '' or str(abstract).strip() == '' or str(doi).strip() == '':
        continue
    
    # Combine title and abstract
    combined_text = f"Title: {str(title).strip()}\n\nAbstract: {str(abstract).strip()}"
    texts.append(combined_text)
    dois.append(str(doi).strip())
    valid_indices.append(idx)

print(f"Found {len(texts)} valid entries to process")

# Compute embeddings in batches
batch_size = 8  # Adjust based on GPU memory
all_embeddings = []

print("Computing embeddings...")
start_time = time.time()

for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
    batch_texts = texts[i:i+batch_size]
    
    # Encode with bfloat16 for efficiency
    batch_embeddings = model.encode(
        batch_texts, 
        dtype=torch.bfloat16,
        show_progress_bar=False
    )
    
    all_embeddings.append(batch_embeddings)

# Combine all embeddings
if all_embeddings:
    embeddings = np.vstack(all_embeddings)
else:
    print("No embeddings computed!")
    exit(1)
    
end_time = time.time()

print(f"Embedding computation completed in {end_time - start_time:.2f} seconds")
print(f"Embeddings shape: {embeddings.shape}")

# Create DataFrame with DOI as index
embedding_df = pd.DataFrame(
    embeddings,
    index=dois,
    columns=[f'dim_{i}' for i in range(embeddings.shape[1])]
)

# Save to CSV
output_path = '/home/tadreamk-admin/alex/ts-embed/pdf_project/db/embedded.csv'
print(f"Saving embeddings to {output_path}...")
embedding_df.to_csv(output_path)

print("Done!")
print(f"Saved {len(embedding_df)} embeddings with {embeddings.shape[1]} dimensions each")
