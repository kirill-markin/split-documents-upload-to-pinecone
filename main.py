import os
import time
import datetime
from typing import List, TypeAlias

import pinecone
import pinecone.core.client
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv, find_dotenv


def load_docs(__directory: str) -> List[Document]:
    """
    Loads all .md documents from the directory and stores them into a list

    Parameters:
        __directory: directory from which all files will be taken

    Returns:
        List of documents with "md" extension
    """
    __loader = DirectoryLoader(__directory, glob="**/*.md", loader_cls=TextLoader)
    __documents = __loader.load()
    return __documents


def split_docs(__documents: List[Document], __chunk_size: int = 1000, __chunk_overlap: int = 20) -> List[Document]:
    """
    Takes in a list of documents and splits them into chunks of a specified size with a specified overlap.

    Parameters:
        __documents: List of files, specifically with "md" extension
        __chunk_size: How many chunks will be created
        __chunk_overlap: Specified overlap for files

    Returns:
        List of files that can be stored to Pinecone
    """
    __text_splitter = RecursiveCharacterTextSplitter(chunk_size=__chunk_size, chunk_overlap=__chunk_overlap)
    return __text_splitter.split_documents(__documents)


def upload_documents_with_retry(__retries: int = 3, __delay: int = 5):
    """
    Upload documents to a Pinecone index with the option to retry a specified number of times if the upload fails

    Parameters:
        __retries: number of retries the program will try to upload documents
        __delay: Time the program will wait before retrying to upload

    Returns:
        A VectorStore initialized from documents and embeddings

    Raises:
        ServiceException error if number of retries exceeds
    """
    for __i in range(__retries):
        try:
            # create index and upload documents
            __index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
            return __index
        except pinecone.core.client.exceptions.ServiceException as __e:
            if __i < __retries - 1:
                print(f"Upload failed. Retrying in {__delay} seconds.")
                time.sleep(__delay)
                __delay *= 2
            else:
                raise __e


load_dotenv(find_dotenv())

# Load documents
directory = "./data"
if not os.path.exists(directory):
    os.makedirs(directory)

# create folder with logs if not exists
logs_dir = "./logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Load documents from the directory
documents = load_docs(directory)
print(f"Number of documents: {len(documents)}")

# Save result in a file to check the result manually
docs = split_docs(documents)
print(f"Number of split documents: {len(docs)}")

now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
with open(f"{logs_dir}/{now}-temp.md", "w") as f:
    for d in docs:
        f.write(d.page_content)
        f.write("\n\n\n!--------------------------------------------\n\n\n")

# Prepare embeddings
embeddings = OpenAIEmbeddings()

query_result = embeddings.embed_query("Hello world")
dimension = len(query_result)
print(f"Dimension in embeddings: {dimension}")

# Pinecone init
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv('PINECONE_ENVIRONMENT'),
)

index_name = os.getenv('PINECONE_INDEX_NAME')

# Recreate the index
# Can take 15 minutes or more
NotFoundException: TypeAlias = pinecone.exceptions.NotFoundException
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
index = upload_documents_with_retry()
# index = Pinecone.from_existing_index(index_name, embeddings) # connect to existing index
print(f"Pinecone index created: {index}")

query = "test"
k = 2
res = index.similarity_search_with_score(query, k=k)
print(f"Result of test query: {res}")
