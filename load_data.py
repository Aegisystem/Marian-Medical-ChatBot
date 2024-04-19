import os
import sys
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_SECURE_BUNDLE_PATH = os.environ.get('ASTRA_DB_SECURE_BUNDLE_PATH')
ASTRA_DB_CLIENT_ID = os.environ.get('ASTRA_DB_CLIENT_ID')
ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')

text_splitter = RecursiveCharacterTextSplitter(separators=["\ARTICULO","\n\n", "\n", " ", ""], chunk_size=1250, chunk_overlap=100)

file_path = os.path.abspath(sys.argv[1])

print('Loading document')
documents = TextLoader(file_path).load()
splittedDocs = text_splitter.split_documents(documents)

for doc in splittedDocs:
  doc.metadata['source'] = 'Estatuto Tributario 2023'

print('Connecting to the model')
model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

cluster = Cluster(
  cloud= {
    "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH
  },
  auth_provider=PlainTextAuthProvider(
    ASTRA_DB_CLIENT_ID,
    ASTRA_DB_APPLICATION_TOKEN,
  ),
)
print("Connecting to AstraDB")
session = cluster.connect()

print("Generating and storing embeddings on AstraDB")
doc_search = Cassandra.from_documents(
  documents=splittedDocs,
  embedding=embeddings,
  session=session,
  keyspace='keyspace_1',
  table_name='data_embeddings',
)