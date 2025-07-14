import os
import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

# Qdrant setup
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_knowledge"

# Embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"

# Data paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEDQUAD_PATH = os.path.join(BASE_DIR, 'data', 'medquad', 'processed', 'processed_medquad.json')
PUBMED_DIR = os.path.join(BASE_DIR, 'data', 'pubmed', 'processed')

# Load data
print("Loading MedQuAD data...")
with open(MEDQUAD_PATH, 'r') as f:
    medquad_data = json.load(f)

print("Loading PubMed data...")
pubmed_data = []
for fname in os.listdir(PUBMED_DIR):
    if fname.endswith('.json'):
        with open(os.path.join(PUBMED_DIR, fname), 'r') as f:
            pubmed_data.extend(json.load(f))

# Prepare documents for indexing
medquad_docs = [
    {
        "id": i,
        "text": f"Q: {item['question']}\nA: {item['answer']}",
        "metadata": {
            "type": "faq",
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "focus": item.get("focus", ""),
            "file": item.get("file", "")
        }
    }
    for i, item in enumerate(medquad_data)
]

pubmed_docs = [
    {
        "id": i,
        "text": f"{item.get('title', '')}\n{item.get('abstract', '')}",
        "metadata": {
            "type": "pubmed",
            "pmid": item.get("pmid", ""),
            "journal": item.get("journal", ""),
            "year": item.get("year", ""),
            "authors": item.get("authors", [])
        }
    }
    for i, item in enumerate(pubmed_data)
]

documents = medquad_docs + pubmed_docs
print(f"Total documents to index: {len(documents)}")

# Load embedding model
print(f"Loading embedding model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)

# Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Create collection if it doesn't exist
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=qdrant_models.Distance.COSINE
        )
    )

# Batch upload
BATCH_SIZE = 64
for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Indexing to Qdrant"):
    batch = documents[i:i+BATCH_SIZE]
    texts = [doc["text"] for doc in batch]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    payloads = [doc["metadata"] for doc in batch]
    ids = [doc["id"] for doc in batch]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qdrant_models.PointStruct(
                id=ids[j],
                vector=embeddings[j],
                payload=payloads[j]
            )
            for j in range(len(batch))
        ]
    )

print("Indexing complete!") 