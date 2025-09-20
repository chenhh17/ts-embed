from sentence_transformers import SentenceTransformer
import torch
from transformers import BitsAndBytesConfig
import numpy as np

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

# Test sentence with clear keywords
sentence = "Machine learning algorithms improve organizational performance through data analysis and predictive modeling."

print(f"Test sentence: {sentence}")
print()

# Get different types of embeddings
print("1. Getting sentence-level embedding...")
sentence_embedding = model.encode(sentence, dtype=torch.bfloat16)
print(f"Sentence embedding shape: {sentence_embedding.shape}")
print()

# Get token-level embeddings
print("2. Getting token-level embeddings...")
token_embeddings = model.encode(sentence, output_value="token_embeddings", dtype=torch.bfloat16)
print(f"Token embeddings shape: {token_embeddings.shape}")
print()

# Get tokenizer to understand token mapping
tokenizer = model.tokenizer
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.encode(sentence, add_special_tokens=True)

print("3. Token analysis:")
print(f"Number of tokens: {len(tokens)}")
print(f"Number of token IDs: {len(token_ids)}")
print(f"Token embeddings tensor shape: {token_embeddings.shape}")
print()

print("Token details:")
for i, (token, token_id) in enumerate(zip(tokens, token_ids[1:-1])):  # Skip special tokens
    print(f"  {i+1}: '{token}' (ID: {token_id})")
print()

# Define keywords to investigate
keywords = ["machine", "learning", "algorithms", "organizational", "performance", "data", "analysis", "predictive", "modeling"]

print("4. Investigating keyword embeddings:")
print("=" * 60)

for keyword in keywords:
    print(f"\nKeyword: '{keyword}'")
    
    # Method 1: Direct keyword embedding
    keyword_direct = model.encode(keyword, dtype=torch.bfloat16)
    
    # Method 2: Find keyword in tokens and extract corresponding embedding
    keyword_in_context = None
    token_indices = []
    
    # Look for the keyword in tokens (handle subword tokenization)
    for i, token in enumerate(tokens):
        if keyword.lower() in token.lower() or token.lower() in keyword.lower():
            token_indices.append(i + 1)  # +1 because of special start token
            if keyword_in_context is None:
                keyword_in_context = token_embeddings[i + 1]  # +1 for special start token
    
    if keyword_in_context is not None:
        # Convert to numpy arrays (handle both tensor and numpy cases)
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x
        
        keyword_direct_cpu = to_numpy(keyword_direct)
        keyword_in_context_cpu = to_numpy(keyword_in_context)
        sentence_embedding_cpu = to_numpy(sentence_embedding)
        
        # Calculate similarities
        direct_vs_context_sim = np.dot(keyword_direct_cpu, keyword_in_context_cpu) / (
            np.linalg.norm(keyword_direct_cpu) * np.linalg.norm(keyword_in_context_cpu)
        )
        
        sentence_vs_keyword_sim = np.dot(sentence_embedding_cpu, keyword_direct_cpu) / (
            np.linalg.norm(sentence_embedding_cpu) * np.linalg.norm(keyword_direct_cpu)
        )
        
        sentence_vs_context_sim = np.dot(sentence_embedding_cpu, keyword_in_context_cpu) / (
            np.linalg.norm(sentence_embedding_cpu) * np.linalg.norm(keyword_in_context_cpu)
        )
        
        print(f"  Found in tokens at positions: {token_indices}")
        print(f"  Matching tokens: {[tokens[i-1] for i in token_indices]}")
        print(f"  Direct embedding shape: {keyword_direct.shape}")
        print(f"  Context embedding shape: {keyword_in_context.shape}")
        print(f"  Similarity (direct vs context): {direct_vs_context_sim:.4f}")
        print(f"  Similarity (sentence vs direct keyword): {sentence_vs_keyword_sim:.4f}")
        print(f"  Similarity (sentence vs context keyword): {sentence_vs_context_sim:.4f}")
    else:
        print(f"  Not found in tokens (might be part of subwords)")
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x
        
        keyword_direct_cpu = to_numpy(keyword_direct)
        sentence_embedding_cpu = to_numpy(sentence_embedding)
        sentence_vs_keyword_sim = np.dot(sentence_embedding_cpu, keyword_direct_cpu) / (
            np.linalg.norm(sentence_embedding_cpu) * np.linalg.norm(keyword_direct_cpu)
        )
        print(f"  Direct embedding shape: {keyword_direct.shape}")
        print(f"  Similarity (sentence vs direct keyword): {sentence_vs_keyword_sim:.4f}")

print("\n" + "=" * 60)
print("5. Advanced analysis: Attention-based keyword extraction")

# Let's try a different approach - looking at all token embeddings
print(f"\nAll token embeddings analysis:")
print(f"Sentence embedding: {sentence_embedding[:5]}...")  # Show first 5 dims

print(f"\nToken-by-token similarity to sentence embedding:")
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

sentence_embedding_cpu = to_numpy(sentence_embedding)
for i, token in enumerate(tokens):
    if i + 1 < token_embeddings.shape[0]:  # Make sure we don't go out of bounds
        token_emb = to_numpy(token_embeddings[i + 1])  # +1 for special start token
        similarity = np.dot(sentence_embedding_cpu, token_emb) / (
            np.linalg.norm(sentence_embedding_cpu) * np.linalg.norm(token_emb)
        )
        print(f"  '{token}': {similarity:.4f}")

print("\n" + "=" * 60)
print("6. Experiment: Token Sum vs Sentence Embedding")

# Sum all token embeddings (excluding special tokens)
print("\nTesting if sum of token embeddings approximates sentence embedding...")

# Get token embeddings excluding special tokens (first and last)
token_embeddings_no_special = token_embeddings[1:-1]  # Remove start and end tokens
print(f"Token embeddings shape (no special tokens): {token_embeddings_no_special.shape}")

# Sum all token embeddings
token_sum_embedding = torch.sum(token_embeddings_no_special, dim=0)
print(f"Token sum embedding shape: {token_sum_embedding.shape}")

# Convert to numpy for similarity calculation
token_sum_cpu = to_numpy(token_sum_embedding)
sentence_embedding_cpu = to_numpy(sentence_embedding)

# Calculate cosine similarity
cosine_similarity = np.dot(sentence_embedding_cpu, token_sum_cpu) / (
    np.linalg.norm(sentence_embedding_cpu) * np.linalg.norm(token_sum_cpu)
)

print(f"\nCosine similarity (sentence vs token sum): {cosine_similarity:.4f}")

# Also try mean of token embeddings (more common approach)
token_mean_embedding = torch.mean(token_embeddings_no_special, dim=0)
token_mean_cpu = to_numpy(token_mean_embedding)

cosine_similarity_mean = np.dot(sentence_embedding_cpu, token_mean_cpu) / (
    np.linalg.norm(sentence_embedding_cpu) * np.linalg.norm(token_mean_cpu)
)

print(f"Cosine similarity (sentence vs token mean): {cosine_similarity_mean:.4f}")

# Compare magnitudes
print(f"\nMagnitude comparison:")
print(f"  Sentence embedding magnitude: {np.linalg.norm(sentence_embedding_cpu):.4f}")
print(f"  Token sum magnitude: {np.linalg.norm(token_sum_cpu):.4f}")
print(f"  Token mean magnitude: {np.linalg.norm(token_mean_cpu):.4f}")

# Show first few dimensions for comparison
print(f"\nFirst 5 dimensions comparison:")
print(f"  Sentence embedding: {sentence_embedding_cpu[:5]}")
print(f"  Token sum embedding: {token_sum_cpu[:5]}")
print(f"  Token mean embedding: {token_mean_cpu[:5]}")

print("\n" + "=" * 60)
print("7. Summary and Recommendations:")
print("""
Key findings from this investigation:

1. TOKEN EMBEDDINGS: model.encode(sentence, output_value="token_embeddings") 
   returns embeddings for each token in the sentence, shaped as [num_tokens, embedding_dim]

2. KEYWORD EXTRACTION METHODS:
   a) Direct: model.encode(keyword) - gives keyword embedding in isolation
   b) Contextual: Extract from token_embeddings using token position mapping
   
3. CONTEXTUAL vs DIRECT EMBEDDINGS:
   - Contextual embeddings capture the keyword meaning within sentence context
   - Direct embeddings represent the keyword in isolation
   - Similarity between them varies based on how context affects meaning

4. PRACTICAL APPLICATIONS:
   - Use contextual embeddings for context-aware keyword analysis
   - Use direct embeddings for general keyword matching
   - Token-level similarities can identify important words in the sentence

5. LIMITATIONS:
   - Subword tokenization may split keywords across multiple tokens
   - Need to handle special tokens (start/end tokens)
   - Token alignment requires careful indexing
""")
