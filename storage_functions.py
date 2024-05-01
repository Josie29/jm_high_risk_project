import os
from langchain_community.document_loaders import  PyMuPDFLoader
from langchain_astradb.document_loaders import AstraDBLoader
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from dotenv import load_dotenv
load_dotenv()

def create_astra_vstore(embedding, collection_name):
    """
    Create an astra vector store using the provided embedding and collection name.

    Args:
    embedding (str): The embedding to be used for the vector store.
    collection_name (str): The name of the collection for the vector store.

    Returns:
    vstore (AstraDBVectorStore): The created vector store.

    """
    vstore = AstraDBVectorStore(
        embedding = embedding,
        collection_name = collection_name,
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    return vstore

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a pdf from the provided pdf path, storing each page as a Document.

    Args:
    pdf_path (str): The path to the pdf to be loaded.

    Returns:
    docs (List[Document]): The loaded document.

    """
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def chunk_docs(docs: List[Document], chunks_per_page: int, embedding) -> List[Document]:
    """
    Splits a list of documents into multiple chunks using a semantic chunker.

    Args:
        docs (List[Document]): The list of documents to be split into chunks.
        chunks_per_page (int): The number of chunks to create per page.
        embedding: The embedding model used by the semantic chunker.

    Returns:
        List[Document]: The list of chunks created from the input documents.
    """
    chunker = SemanticChunker(embedding, number_of_chunks=chunks_per_page)
    chunks = chunker.split_documents(docs)
    return chunks

def load_docs_from_db(collection_name: str) -> List[Document]:
    """
    Load documents from the specified collection in the database.

    Args:
        collection_name (str): The name of the collection to load documents from.

    Returns:
        List[Document]: A list of documents loaded from the collection.
    """
    loader = AstraDBLoader(
        collection_name = collection_name,
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    docs = loader.load()
    return docs