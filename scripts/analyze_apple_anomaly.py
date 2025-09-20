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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def analyze_token_context(sentence, target_word):
    """Detailed analysis of token and surrounding context"""
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(sentence)
    token_embeddings = model.encode(sentence, output_value="token_embeddings", dtype=torch.bfloat16)
    
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}")
    
    # Find the target word and get surrounding context
    target_idx = None
    for i, token in enumerate(tokens):
        if target_word.lower() in token.lower().replace('ġ', ''):
            target_idx = i
            break
    
    if target_idx is not None:
        # Get surrounding context (2 tokens before and after)
        start = max(0, target_idx - 2)
        end = min(len(tokens), target_idx + 3)
        context_tokens = tokens[start:end]
        
        print(f"Target word '{target_word}' found at position {target_idx}")
        print(f"Context window: {context_tokens}")
        print(f"Context: '{' '.join([t.replace('Ġ', ' ') for t in context_tokens])}'")
        
        return to_numpy(token_embeddings[target_idx + 1])  # +1 for special start token
    
    return None

print("=" * 80)
print("DETAILED ANALYSIS: APPLE ANOMALY")
print("=" * 80)

apple_sentences = [
    "I ate a red apple for breakfast.",      # Context 1: fruit
    "Apple released a new iPhone model.",    # Context 2: company  
    "The apple tree in our garden is blooming."  # Context 3: fruit/tree
]

print("STEP 1: Analyzing each context in detail")
print("-" * 50)

embeddings = []
for i, sentence in enumerate(apple_sentences):
    print(f"\nContext {i+1}:")
    embedding = analyze_token_context(sentence, "apple")
    if embedding is not None:
        embeddings.append(embedding)
    print()

print("STEP 2: Pairwise similarity analysis")
print("-" * 50)

labels = ["Fruit (eating)", "Company (iPhone)", "Tree (garden)"]
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{labels[i]} vs {labels[j]}: {sim:.4f}")

print("\nSTEP 3: Testing alternative sentences")
print("-" * 50)

# Test with more explicit fruit contexts
alternative_sentences = [
    "The juicy apple tasted sweet and crisp.",           # More explicit fruit context
    "Apple Inc. is a technology company.",              # More explicit company context
    "Apple trees produce delicious fruit in autumn.",   # More explicit tree/fruit context
    "I picked a ripe apple from the tree.",            # Direct fruit picking
]

print("Testing with more explicit contexts:")
alt_embeddings = []
alt_labels = ["Juicy fruit", "Apple Inc.", "Fruit trees", "Picked fruit"]

for i, sentence in enumerate(alternative_sentences):
    print(f"\nAlternative {i+1}: {sentence}")
    tokenizer = model.tokenizer
    tokens = tokenizer.tokenize(sentence)
    token_embeddings = model.encode(sentence, output_value="token_embeddings", dtype=torch.bfloat16)
    
    # Find apple token
    for j, token in enumerate(tokens):
        if "apple" in token.lower().replace('ġ', ''):
            embedding = to_numpy(token_embeddings[j + 1])
            alt_embeddings.append(embedding)
            break

print(f"\nAlternative similarities:")
for i in range(len(alt_embeddings)):
    for j in range(i+1, len(alt_embeddings)):
        sim = cosine_similarity(alt_embeddings[i], alt_embeddings[j])
        print(f"{alt_labels[i]} vs {alt_labels[j]}: {sim:.4f}")

print("\nSTEP 4: Hypothesis testing")
print("-" * 50)

print("""
POSSIBLE EXPLANATIONS FOR THE ANOMALY:

1. LINGUISTIC PATTERNS:
   - "Apple released" vs "apple tree" might share similar grammatical structures
   - Both use "apple" as a subject/modifier in similar syntactic positions

2. TRAINING DATA BIAS:
   - The model might have seen more "apple tree" contexts in tech/business discussions
   - Or "Apple" (company) and "apple tree" co-occur in certain contexts

3. SEMANTIC OVERLAP:
   - "Apple tree" and "Apple company" both represent entities/organizations
   - While "ate apple" is more about consumption/action

4. CONTEXTUAL WORDS:
   - "tree" and "released" might have unexpected semantic relationships
   - "garden" and "iPhone" might share some conceptual space (products, cultivation)

Let's test these hypotheses...
""")

# Test sentence-level embeddings to see if the pattern persists
print("STEP 5: Sentence-level embedding comparison")
print("-" * 50)

sentence_embeddings = []
for sentence in apple_sentences:
    emb = to_numpy(model.encode(sentence, dtype=torch.bfloat16))
    sentence_embeddings.append(emb)

print("Sentence-level similarities:")
for i in range(len(sentence_embeddings)):
    for j in range(i+1, len(sentence_embeddings)):
        sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
        print(f"Sentence {i+1} vs Sentence {j+1}: {sim:.4f}")

print("\nCONCLUSION: The anomaly might be due to:")
print("- Syntactic similarity between 'Apple released' and 'apple tree'")
print("- Semantic relationships in the embedding space that aren't intuitive")
print("- Training data patterns that create unexpected associations")
print("- The model capturing abstract relationships beyond simple word meaning")
