from flask import Flask, render_template, jsonify, request
from src.helper import download_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from src.prompt import *

#Initialize the Flask app
app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embedded_model = download_model()

index_name = "chatbot-medical-free"
storing = PineconeVectorStore.from_existing_index(
    embedding=embedded_model,
    index_name=index_name
)

retriveing = storing.as_retriever(search_type = "similarity",
                     search_kwrags = {"k" : 3})

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="alibaba/tongyi-deepresearch-30b-a3b:free",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human" , "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm , prompt)
rag_chain = create_retrieval_chain(retriveing , qa_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get" , methods = ["GET" , "POST"])
def chat():
    msg =  request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input" : msg})
    print("The Response is : " , response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0" , 
            port = 8080,
            debug=True)