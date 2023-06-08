import os
import time
import datetime

import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv

load_dotenv()


# Load documents

directory = './data'

def load_docs(directory):
  loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(f"Number of documents: {len(documents)}")


# Split documents into chunks

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(f"Number of splited documents: {len(docs)}")


# Save result in a file to check the result manually

# create folder with logs if not exists
logs_dir = './logs'
if not os.path.exists(logs_dir):
  os.makedirs(logs_dir)

now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
with open(f'{logs_dir}/{now}-temp.md', 'w') as f:
  for d in docs:
    f.write(d.page_content)
    f.write('\n\n\n!--------------------------------------------\n\n\n')


# Prepare embeddings

embeddings = OpenAIEmbeddings()

query_result = embeddings.embed_query("Hello world")
dimension = len(query_result)
print(f"Dimension in embeddings: {dimension}")


# Pinecone init

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.environ.get('PINECONE_ENVIRONMENT'),
)

index_name = os.environ.get('PINECONE_INDEX_NAME')


# Recreate the index
# Can take 15 minutes or more

NotFoundException = pinecone.exceptions.NotFoundException

try:
    pinecone.delete_index(name=index_name)
    print(f"Index '{index_name}' deleted.")
except NotFoundException:
    print(f"Index '{index_name}' not found. Skipping deletion.")

pinecone.create_index(
    name=index_name,
    dimension=dimension,
    metric="cosine",
)

print(f"pinecone.describe_index(index_name):, {pinecone.describe_index(index_name)}")


# Upload the embeddings to the index
# Can take 10 minutes or more

def upload_documents_with_retry(retries=3, delay=5):
    for i in range(retries):
        try:
            # create index and upload documents
            index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
            return index
        except pinecone.core.client.exceptions.ServiceException as e:
            if i < retries - 1:
                print(f"Upload failed. Retrying in {delay} seconds.")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

index = upload_documents_with_retry()
# index = Pinecone.from_existing_index(index_name, embeddings) # connect to existing index
print(f"Pinecone index created: {index}")

query = "test"
k = 2
res = index.similarity_search_with_score(query, k=k)
print(f"Result of test query: {res}")
