from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Loading the PDF

def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls= PyPDFLoader
    )

    document = loader.load()

    return document

#Extract Reference Metadata and Content
def extract_meta_data(docs: List[Document]) -> List[Document]:

    min_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page") 
        min_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source" : src, 
                            "page": page}

            )
        )
    return min_docs


#Chunking the Document
def chunking(minimal_doc):
    text_chunk = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    split_chunks = text_chunk.split_documents(minimal_doc)

    return split_chunks


#Downloading the Embedding Model
def download_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
        #kwargs={"device" : "cpu"}
    )

    return embeddings
