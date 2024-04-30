import os
import pandas as pd
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
# vstore = AstraDBVectorStore(
#     embedding=embedding,
#     namespace=ASTRA_DB_KEYSPACE,
#     collection_name="medicalinfo",
#     token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
#     api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
# )

# vstore2 = AstraDBVectorStore(
#     embedding=embedding,
#     namespace=ASTRA_DB_KEYSPACE,
#     collection_name="hallazgos",
#     token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
#     api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
# )


# estudios = pd.read_csv("data/estudios.csv", encoding="latin1")
hallazgos = pd.read_csv("data/patron-hallazgos.csv", encoding="utf-8-sig")

# docs = []
# for index, row in estudios.iterrows():
#     metadata = {"Estudio": row["Estudio"],  "Diagnostico": row["Sector/Diagnostico"]}
#     # Add a LangChain document with the quote and metadata tags
#     doc = Document(page_content=row["Resultados"], metadata=metadata)
#     docs.append(doc)

docs2 = []
for index, row in hallazgos.iterrows():
    metadata = {"Tipo": row["tipo"],  "Patron": row["patron"]}
    # Add a LangChain document with the quote and metadata tags
    doc = Document(page_content=row["hallazgo"], metadata=metadata)
    docs2.append(doc)

# inserted_ids = vstore.add_documents(docs)
# print(f"\nInserted {len(inserted_ids)} documents.")

inserted_ids = vstore2.add_documents(docs2)
print(f"\nInserted {len(inserted_ids)} documents.")

# results = vstore.similarity_search("Cardiaca de forma normal", k=3)
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")