from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb
import uvicorn

app = FastAPI(title="Contract Analysis API", version="1.0.0")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Define data models
class Document(BaseModel):
    text: str
    metadata: dict
    id: Optional[str] = None

class SearchQuery(BaseModel):
    query_text: str
    n_results: int = 5
    where: Optional[dict] = None
    where_document: Optional[dict] = None

class SearchResult(BaseModel):
    documents: List[str]
    metadatas: List[dict]
    distances: List[float]
    ids: List[str]

# Initialize collection
COLLECTION_NAME = "contracts"

def get_collection():
    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        return chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

def ensure_unique_distances(results, n_results):
    """
    Ensure all distances are unique by adjusting them slightly if needed.
    Also ensures we maintain the original ordering.
    """
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    
    # Create a list of tuples with original indices to maintain order
    items = list(enumerate(zip(documents, metadatas, distances, ids)))
    
    # Sort by distance
    items.sort(key=lambda x: x[1][2])
    
    # Adjust distances to ensure they're unique while maintaining order
    for i in range(1, len(items)):
        if items[i][1][2] <= items[i-1][1][2]:
            # Add a small increment to make it different
            new_distance = items[i-1][1][2] + 0.000001
            items[i] = (items[i][0], (items[i][1][0], items[i][1][1], new_distance, items[i][1][3]))
    
    # Sort back by original indices
    items.sort(key=lambda x: x[0])
    
    # Unzip the results
    _, result_items = zip(*items)
    documents, metadatas, distances, ids = zip(*result_items)
    
    return {
        "documents": [list(documents)],
        "metadatas": [list(metadatas)],
        "distances": [list(distances)],
        "ids": [list(ids)]
    }

@app.on_event("startup")
async def startup_event():
    get_collection()

@app.post("/documents/", response_model=dict)
async def add_documents(documents: List[Document]):
    collection = get_collection()
    try:
        docs = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id if doc.id else f"doc_{i}" for i, doc in enumerate(documents)]
        
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        return {"message": f"Successfully added {len(documents)} documents", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/", response_model=SearchResult)
async def search_documents(query: SearchQuery):
    collection = get_collection()
    try:
        results = collection.query(
            query_texts=[query.query_text],
            n_results=query.n_results,
            where=query.where,
            where_document=query.where_document
        )
        
        # Process results to ensure unique distances
        processed_results = ensure_unique_distances(results, query.n_results)
        
        return SearchResult(
            documents=processed_results["documents"][0],
            metadatas=processed_results["metadatas"][0],
            distances=processed_results["distances"][0],
            ids=processed_results["ids"][0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    collection = get_collection()
    try:
        result = collection.get(ids=[document_id])
        if not result["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
            "id": result["ids"][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    collection = get_collection()
    try:
        collection.delete(ids=[document_id])
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
