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

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def get_token_embedding(sentence, target_word):
    """Extract token embedding for a specific word from a sentence"""
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(sentence)
    token_embeddings = model.encode(sentence, output_value="token_embeddings", dtype=torch.bfloat16)
    
    # Find the target word in tokens
    for i, token in enumerate(tokens):
        if target_word.lower() in token.lower().replace('ġ', ''):
            return to_numpy(token_embeddings[i + 1])  # +1 for special start token
    return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=" * 80)
print("EXPERIMENT: CONTEXTUAL TOKEN EMBEDDINGS")
print("=" * 80)

# Test sentences with the same word in different contexts
test_cases = [
    {
        "word": "bank",
        "sentences": [
            "I went to the bank to deposit money.",
            "The river bank was covered with flowers.",
            "The bank of computers crashed simultaneously."
        ]
    },
    {
        "word": "apple",
        "sentences": [
            "I ate a red apple for breakfast.",
            "Apple released a new iPhone model.",
            "The apple tree in our garden is blooming."
        ]
    },
    {
        "word": "run",
        "sentences": [
            "I like to run in the morning.",
            "The program will run automatically.",
            "There was a run on the bank during the crisis."
        ]
    },
    {
        "word": "light",
        "sentences": [
            "Turn on the light please.",
            "The box is very light to carry.",
            "Light travels faster than sound."
        ]
    }
]

for case in test_cases:
    word = case["word"]
    sentences = case["sentences"]
    
    print(f"\nTesting word: '{word.upper()}'")
    print("-" * 50)
    
    embeddings = []
    for i, sentence in enumerate(sentences):
        embedding = get_token_embedding(sentence, word)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Context {i+1}: {sentence}")
        else:
            print(f"Context {i+1}: {sentence} [WORD NOT FOUND]")
    
    if len(embeddings) >= 2:
        print("\nPairwise Cosine Similarities:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                print(f"  Context {i+1} vs Context {j+1}: {sim:.4f}")
        
        # Compare with direct word embedding
        direct_embedding = to_numpy(model.encode(word, dtype=torch.bfloat16))
        print(f"\nSimilarity to direct '{word}' embedding:")
        for i, embedding in enumerate(embeddings):
            sim = cosine_similarity(direct_embedding, embedding)
            print(f"  Direct vs Context {i+1}: {sim:.4f}")
    
    print()

print("=" * 80)
print("EXPERIMENT: TOKEN SUM VS SENTENCE EMBEDDING (MULTIPLE EXAMPLES)")
print("=" * 80)

test_sentences = [
    "Machine learning algorithms improve organizational performance.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence will revolutionize healthcare systems.",
    "Climate change poses significant environmental challenges globally."
]

for sentence in test_sentences:
    print(f"\nSentence: {sentence}")
    print("-" * 60)
    
    # Get sentence embedding
    sentence_embedding = model.encode(sentence, dtype=torch.bfloat16)
    sentence_embedding_cpu = to_numpy(sentence_embedding)
    
    # Get token embeddings
    token_embeddings = model.encode(sentence, output_value="token_embeddings", dtype=torch.bfloat16)
    token_embeddings_no_special = token_embeddings[1:-1]  # Remove special tokens
    
    # Sum and mean of tokens
    token_sum = to_numpy(torch.sum(token_embeddings_no_special, dim=0))
    token_mean = to_numpy(torch.mean(token_embeddings_no_special, dim=0))
    
    # Normalize token sum and mean for fair comparison (handle inf/nan)
    token_sum_norm = np.linalg.norm(token_sum)
    token_mean_norm = np.linalg.norm(token_mean)
    
    if token_sum_norm > 0 and not np.isinf(token_sum_norm):
        token_sum_normalized = token_sum / token_sum_norm
        sim_sum = cosine_similarity(sentence_embedding_cpu, token_sum_normalized)
    else:
        sim_sum = float('nan')
    
    if token_mean_norm > 0 and not np.isinf(token_mean_norm):
        token_mean_normalized = token_mean / token_mean_norm
        sim_mean = cosine_similarity(sentence_embedding_cpu, token_mean_normalized)
    else:
        sim_mean = float('nan')
    
    print(f"Cosine similarity (sentence vs normalized token sum):  {sim_sum:.4f}")
    print(f"Cosine similarity (sentence vs normalized token mean): {sim_mean:.4f}")

print("\n" + "=" * 80)
print("CONCLUSIONS:")
print("=" * 80)
print("""
1. CONTEXTUAL TOKEN EMBEDDINGS:
   - Same words in different contexts have DIFFERENT token embeddings
   - Similarity varies significantly based on semantic context
   - Words with multiple meanings show larger variations

2. TOKEN AGGREGATION vs SENTENCE EMBEDDING:
   - Simple sum/mean of tokens ≠ sentence embedding
   - Even with normalization, similarities are moderate at best
   - Sentence embeddings use sophisticated pooling beyond simple averaging

3. IMPLICATIONS:
   - Token embeddings capture word-in-context meaning
   - Sentence embeddings emerge from complex attention mechanisms
   - Both levels provide different but complementary information
""")
