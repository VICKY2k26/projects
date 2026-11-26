import os
import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

client = genai.Client(api_key=API_KEY)

TEST_QUERIES = [
    "When do I need a doctor's certificate for sick leave?",
    "How much can I claim for home internet and how often?",
    "What happens to unused annual leave at the end of the year?",
    "How quickly must I return my laptop after resigning?",
    "Are drinks or cigarettes covered during client travel meals?",
]


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 70) -> list[str]:
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"
    chunks = []
    start = 0
    cleaned = " ".join(text.split())
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        start += chunk_size - overlap
    return chunks


# TODO 1: Add @retry decorator (3 attempts, exponential backoff 1s to 8s, reraise=True)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def get_embedding(text: str) -> list[float]:
    # TODO 2: Call client.models.embed_content(model=EMBED_MODEL, contents=text)
    # and return result.embeddings[0].values
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def index_pdf(pdf_path: str, collection, chunk_size: int = 350, overlap: int = 70):
    doc = fitz.open(pdf_path)
    ids, documents, metadatas, embeddings = [], [], [], []

    # TODO 3: Loop through each page (1-indexed) using enumerate(doc, start=1),
    # extract text with page.get_text("text"), split using chunk_text(),
    # generate an embedding for each chunk, and populate ids, documents,
    # metadatas ({"source": pdf_path, "page": page_num}), and embeddings.
    # Then call collection.upsert(ids=..., documents=..., metadatas=..., embeddings=...).
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(pdf_path)}_p{page_num}_c{chunk_idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": pdf_path, "page": page_num})
            embeddings.append(get_embedding(chunk))

    doc.close()

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def search_policy(collection, query: str, top_k: int = 3):
    # TODO 4: Embed `query`, run collection.query(query_embeddings=[...], n_results=top_k),
    # and print rank, cosine distance score, source page number, and snippet preview.
    # Remember: Chroma returns nested lists (results["documents"][0], etc.).
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []

    for rank, (doc_text, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        page_num = meta.get("page", "?") if meta else "?"
        snippet = doc_text[:160].replace("\n", " ")
        print(f"  {rank}. [Page {page_num} | dist={dist:.4f}] {snippet}...")


def main():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="company_policy",
        metadata={"hnsw:space": "cosine"},
    )

    index_pdf("company_policy.pdf", collection, chunk_size=350, overlap=70)
    print(f"Total chunks indexed in ChromaDB: {collection.count()}\n")

    for q in TEST_QUERIES:
        print(f"Query: {q}")
        search_policy(collection, q, top_k=3)
        print("-" * 60)


if __name__ == "__main__":
    main()