
# ---------------------------------------------------------
# Semantic Search Demo with Pinecone & Sentence-Transformers
# ---------------------------------------------------------
#
# Goal: Search for sentences based on their Meaning (Semantics), not just keywords.
# To run this:
#   pip install pinecone-client sentence-transformers
#

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. Setup & Config
# -----------------
load_dotenv() # Load API keys from .env file
api_key = os.environ.get("PINECONE_API_KEY")

if not api_key:
    print("❌ Error: PINECONE_API_KEY not found in environment variables.")
    exit(1)

INDEX_NAME = "semantic-search-demo"

# 2. Load the "Brain" (The Embedding Model)
# ----------------------------------------
# We use 'all-MiniLM-L6-v2'. It's small, fast, and free to run locally.
# It converts text into a list of 384 numbers.
print("🧠 Loading model (this might take a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
DIMENSION = 384 

# 3. Initialize Pinecone (The Memory)
# ----------------------------------
pc = Pinecone(api_key=api_key)

# Check if index exists, if not create it
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"📂 Creating index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION, # Must match the model output!
        metric="cosine",     # Cosine Similarity is best for text
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

# Connect to the Index
index = pc.Index(INDEX_NAME)
print(f"✅ Connected to Pinecone Index: {INDEX_NAME}")

# 4. Our Data (The Corpus)
# -----------------------
# A mix of topics to prove the search works
data = [
    {"id": "vec1", "text": "Apple released a new iPhone with a better camera."}, # Tech
    {"id": "vec2", "text": "The stock market crashed today due to inflation."}, # Finance
    {"id": "vec3", "text": "Lions are social animals that live in prides."},    # Animals
    {"id": "vec4", "text": "My laptop keyboard is broken and needs repair."},   # Tech
    {"id": "vec5", "text": "Dogs are known as man's best friend."},            # Animals
    {"id": "vec6", "text": "Interest rates were raised by the central bank."},  # Finance
]

# 5. Embed & Upsert (Teaching the Database)
# ----------------------------------------
print("\n📤 Upserting data...")
vectors_to_upsert = []

for item in data:
    # Convert Text -> Numbers (Vector)
    vector_values = model.encode(item["text"]).tolist()
    
    # Pack it for Pinecone: (ID, Vector, Metadata)
    vectors_to_upsert.append((
        item["id"], 
        vector_values, 
        {"text": item["text"]} # Store original text as metadata so we can read it later!
    ))

index.upsert(vectors=vectors_to_upsert)
print(f"✨ Uploaded {len(data)} vectors.")
time.sleep(2) # Give Pinecone a moment to index

# 6. The "Search" (Semantic Querying)
# ----------------------------------
# Notice: We don't use keywords like "Apple" or "Lion". 
# We use concepts.

queries = [
    "tell me about wild cats",       # Should match "Lions"
    "tech news",                     # Should match "iPhone" or "Laptop"
    "money and economy"              # Should match "Stock market" or "Interest rates"
]

print("\n🔍 Starting Semantic Search...\n")

for query_text in queries:
    print(f"❓ Question: '{query_text}'")
    
    # 1. Convert Question -> Vector
    query_vector = model.encode(query_text).tolist()
    
    # 2. Ask Pinecone for the 2 closest matches
    results = index.query(
        vector=query_vector, 
        top_k=2, 
        include_metadata=True
    )
    
    # 3. Print Results
    for match in results.matches:
        print(f"   👉 Match ({match.score:.2f}): {match.metadata['text']}")
    print("-" * 40)

print("\n🎉 Done! You just built a Semantic Search Engine.")
