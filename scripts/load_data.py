import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from FlagEmbedding import FlagModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE")
ASTRA_DB_COLLECTION = os.environ.get("ASTRA_DB_COLLECTION")

model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vstore = AstraDBVectorStore(
    embedding=embedding,
    namespace=ASTRA_DB_KEYSPACE,
    collection_name="test",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
print("An example entry:")
print(philo_dataset[16])

docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        # Add metadata tags to the metadata dictionary
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    # Add a LangChain document with the quote and metadata tags
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)
    
inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")

results = vstore.similarity_search("Our life is what we make of it", k=3)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")