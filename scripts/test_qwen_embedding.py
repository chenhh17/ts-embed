from sentence_transformers import SentenceTransformer
import torch
from transformers import BitsAndBytesConfig
import time

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=False
)

# Load model with 8-bit quantization
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={
        "attn_implementation": "flash_attention_2", 
        "device_map": "cuda:0",
        "quantization_config": quantization_config
    },
    tokenizer_kwargs={"padding_side": "left"},
)


# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument

# check embedding speed for two queries and two documents
start_time = time.time()
query_embeddings = model.encode(queries, prompt_name="query", dtype=torch.bfloat16)
document_embeddings = model.encode(documents,  dtype=torch.bfloat16)
end_time = time.time()
print(f"Embedding speed: {end_time - start_time} seconds")

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)