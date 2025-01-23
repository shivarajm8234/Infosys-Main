import pandas as pd
import chromadb

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection_name = "DatasetEx"

if collection_name not in chroma_client.list_collections():
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

# Load Excel data
file = r"C:\Users\sachi\Downloads\Dataset - updated_file_with_contracts_final (2).csv"
df = pd.read_csv(file)

# Add an ID column if it doesn't exist
df["ID"] = ["doc_" + str(i) for i in range(len(df))]

# Specify the columns for documents and metadata
document_columns = ["Category", "Parties", "Agreement Date" ,"Expiration Date", "Renewal Term" , "Governing Law" , "Exclusivity","contract"]  # Replace with your chosen document columns
metadata_columns = ["Document Name", "Effective Date"]  # Replace with your chosen metadata columns

# Prepare documents, metadata, and IDs
# Create documents using f-strings
documents = df.apply(
    lambda row: (
        f"Document Name: {row['Document Name']} | "
        f"Effective Date: {row['Effective Date']} | "
        f"Category: {row['Category']} | "
        f"Parties Involved: {row['Parties']} | "
        f"Agreement Date: {row['Agreement Date']} | "
        f"Expiration Date: {row['Expiration Date']} | "
        f"Renewal Term: {row['Renewal Term']} | "
        f"Governing Law: {row['Governing Law']} | "
        f"Exclusivity: {row['Exclusivity']} | "
        f"Contract Details: {row['contract']}"
    ),
    axis=1
).tolist()

# Extract metadata
metadata = df[["Document Name", "Effective Date", "Category"]].to_dict(orient="records")

ids = df["ID"].tolist()

# Add data to the ChromaDB collection
collection.add(documents=documents, metadatas=metadata, ids=ids)

print(f"Added {len(documents)} records to the ChromaDB collection '{collection_name}'.")


# # Specify the ID you want to search for
# specific_id = "doc_0"  # Replace with the desired ID

# # Retrieve data for the given ID
# data_by_id = collection.get(ids=[specific_id])

# # Check if data exists for the given ID
# if data_by_id["documents"]:
#     print(f"Data for ID '{specific_id}':")
#     print("Document:", data_by_id["documents"][0])
#     print("Metadata:", data_by_id["metadatas"][0])
#     print("ID:", data_by_id["ids"][0])
# else:
#     print(f"No data found for ID '{specific_id}'.")

query_text = """
1. Objective of the Agreement: The first party intends to transfer money to people residing in areas outside of the control of the Government of Syria; the second party has the proven capacity, experience, and highest reasonable standard of diligence to facilitate the transfers.
2. Obligations of the First Party: Apply the highest reasonable standard of diligence to ensure that the money transferred under this agreement is not transferred to any individual or entity listed as a Designated Terrorist; coordinate public statements/agree comms strategy; inform the second party of the amount of money to be transferred to Syria; etc.
3. Obligations of the Second Party: Apply the highest reasonable standard of diligence to ensure that the money transferred under this agreement is not transferred to any individual or entity listed as a Designated Terrorist; provide first party's representatives with Syrian Pounds, Turkish Liras, or US Dollars; use an exchange rate equal to or more competitive than the Money Changers in the black currency exchange market; etc.
4. Duration: This contract is valid for a period of ______ months commencing from the signature date.
5. Modification and Cancellation: The Parties may only amend this agreement through mutual written consent; in case the second party cannot provide the required amount, he should submit something in writing to care stating his reason, based on which the first party can ask another money dealer to provide the needed amount onwards; etc.
6. Dispute Resolution: Both parties shall use their best efforts to resolve any dispute in a friendly manner, through consultation and clear communication; any dispute which cannot be resolved in such a way will be taken to a mutually agreed mediator on a cost-sharing basis.
"""

# Perform similarity search
top_k = 2  # Number of similar documents to retrieve
results = collection.query(
    query_texts=[query_text],
    n_results=top_k
)

# Process and display results
# Display results
print("Search Results:")
for i, (doc_id, metadata, score) in enumerate(zip(results["ids"], results["metadatas"], results["distances"])):
    print(f"\nResult {i + 1}:")
    print(f"Similarity Score: {score:}")
    print(f"Document ID: {doc_id}")
    print(f"Contract Details: {metadata.get('Document Name', 'N/A')}")
