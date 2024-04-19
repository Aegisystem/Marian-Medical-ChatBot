import os
from langchain.chains import RetrievalQAWithSourcesChain 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Cassandra
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import chainlit
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_SECURE_BUNDLE_PATH = os.environ.get('ASTRA_DB_SECURE_BUNDLE_PATH')
ASTRA_DB_CLIENT_ID = os.environ.get('ASTRA_DB_CLIENT_ID')
ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
CHAIN_SESSION_KEY = 'user_chain'

model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def open_db_connection():
  cluster = Cluster(
    cloud= {
      "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH
    },
    auth_provider=PlainTextAuthProvider(
      ASTRA_DB_CLIENT_ID,
      ASTRA_DB_APPLICATION_TOKEN,
    ),
  )
  return cluster.connect()

def get_retrieval_qa_chain(llm, doc_search: Cassandra):
  prompt_template = """
  Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

  {summaries}

  Question: {question}
  Answer in Spanish:"""
  chat_prompt = ChatPromptTemplate(
    messages=[
      SystemMessagePromptTemplate.from_template(
        "Your name is Taxa you are a helpful accounting assistant with expertise on Colombian tax rules."
      ),
      HumanMessagePromptTemplate.from_template(prompt_template)
    ],
    input_variables=["summaries", "question"],
  )

  return RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=doc_search.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={'prompt': chat_prompt},
  )

@chainlit.on_chat_start
async def start():
  doc_search = Cassandra(
    embedding=embeddings,
    session=open_db_connection(),
    keyspace='keyspace_1',
    table_name='data_embeddings'
  )
  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5, streaming=True)
  chain = get_retrieval_qa_chain(llm, doc_search)

  chainlit.user_session.set(CHAIN_SESSION_KEY, chain)
  await chainlit.Message('Hola soy Taxa! Soy una asistente virtual contable con experiencia en las normas fiscales colombianas.\nEstoy aprendiendo así que ten paciencia conmigo puedo equivocarme.').send()

@chainlit.on_message
async def on_message(message):
  chain = chainlit.user_session.get(CHAIN_SESSION_KEY)
  callback = chainlit.AsyncLangchainCallbackHandler(
    stream_final_answer=True, answer_prefix_tokens=['FINAL', 'ANSWER']
  )
  # callback.answer_reached = True
  res = await chain.acall({ 'question': message }, callbacks=[callback])

  await chainlit.Message(res['answer']).send()