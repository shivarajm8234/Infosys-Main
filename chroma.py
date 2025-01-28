import chromadb
import pandas as pd

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Initialize collection
collection_name = "DatasetEx"
try:
    collections = chroma_client.list_collections()
    collection_names = [col.name for col in collections]
    if collection_name not in collection_names:
        collection = chroma_client.create_collection(name=collection_name)
    else:
        collection = chroma_client.get_collection(name=collection_name)
except Exception as e:
    print(f"Error creating collection: {str(e)}")
    exit(1)

# Load Excel data
file = r"cleaned_dataset.csv"
df = pd.read_csv(file)


# Add an ID column if it doesn't exist
if 'id' not in df.columns:
    df['id'] = range(len(df))

# Convert ID to string
df['id'] = df['id'].astype(str)

# Specify the columns for documents and metadata
document_columns = ["Category", "Parties", "Agreement Date" ,"Effective Date", "Expiration Date", "Renewal Term" , "Governing Law" , "Law Explanation"]  
metadata_columns = ["Category", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term", "Governing Law", "Law Explanation"]  

# Prepare documents, metadata, and IDs
# Create documents using f-strings
documents = df.apply(
    lambda row: (
        f"Category: {row['Category']} | "
        f"Parties: {row['Parties']} | "
        f"Agreement Date: {row['Agreement Date']} | "
        f"Effective Date: {row['Effective Date']} | "
        f"Expiration Date: {row['Expiration Date']} | "
        f"Renewal Term: {row['Renewal Term']} | "
        f"Governing Law: {row['Governing Law']} | "
        f"Law Explanation: {row['Law Explanation']}"
    ),
    axis=1
).tolist()

# Extract metadata
metadata = df[metadata_columns].to_dict(orient="records")

ids = df["id"].tolist()

try:
    # Add data to the ChromaDB collection
    collection.add(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    print(f"Added {len(documents)} records to the ChromaDB collection '{collection_name}'.")
except Exception as e:
    print(f"Error adding documents: {str(e)}")
    exit(1)

query_text = """
1. Objective of the Agreement: The first party intends to transfer money to people residing in areas outside of the control of the Government of Syria; the second party has the proven capacity, experience, and highest reasonable standard of diligence to facilitate the transfers.
2. Obligations of the First Party: Apply the highest reasonable standard of diligence to ensure that the money transferred under this agreement is not transferred to any individual or entity listed as a Designated Terrorist; coordinate public statements/agree comms strategy; inform the second party of the amount of money to be transferred to Syria; etc.
3. Obligations of the Second Party: Apply the highest reasonable standard of diligence to ensure that the money transferred under this agreement is not transferred to any individual or entity listed as a Designated Terrorist; provide first party's representatives with Syrian Pounds, Turkish Liras, or US Dollars; use an exchange rate equal to or more competitive than the Money Changers in the black currency exchange market; etc.
4. Duration: This contract is valid for a period of ______ months commencing from the signature date.
5. Modification and Cancellation: The Parties may only amend this agreement through mutual written consent; in case the second party cannot provide the required amount, he should submit something in writing to care stating his reason, based on which the first party can ask another money dealer to provide the needed amount onwards; etc.
6. Dispute Resolution: Both parties shall use their best efforts to resolve any dispute in a friendly manner, through consultation and clear communication; any dispute which cannot be resolved in such a way will be taken to a mutually agreed mediator on a cost-sharing basis.
"""

try:
    # Perform similarity search
    top_k = 2  # Number of similar documents to retrieve
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    # Process and display results
    # Display results
    print("Search Results:")
    for i in range(len(results['ids'][0])):
        print(f"\nResult {i + 1}:")
        print(f"Similarity Score: {float(results['distances'][0][i]):.4f}")
        print(f"Document ID: {results['ids'][0][i]}")
        print(f"Category: {results['metadatas'][0][i].get('Category', 'N/A')}")
        print(f"Governing Law: {results['metadatas'][0][i].get('Governing Law', 'N/A')}")
        print(f"Document: {results['documents'][0][i]}")
except Exception as e:
    print(f"Error querying collection: {str(e)}")
    exit(1)
