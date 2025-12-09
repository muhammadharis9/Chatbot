from dotenv import load_dotenv
import os


from src.helper import load_pdf, extract_meta_data, chunking, download_model
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

extracted_doc = load_pdf("dataset")
minimal_doc = extract_meta_data(extracted_doc)
chunked_doc = chunking(minimal_doc)
embedded_model = download_model()

pine_api = PINECONE_API_KEY
openrouter_api_key = OPENROUTER_API_KEY

pc = Pinecone(api_key=pine_api)

index_name = "chatbot-medical-free"
if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension=384,
        metric="cosine",
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(index_name)

storing = PineconeVectorStore.from_documents(
    documents= chunked_doc,
    embedding=embedded_model,
    index_name=index_name
)
