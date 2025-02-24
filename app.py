from flask import Flask, render_template, jsonify, request
import sys
from langchain import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from src.helper import download_hugging_face_embeddings

app = Flask(__name__)
key = input("Enter Pinecone Api key: ")
os.environ['PINECONE_API_KEY'] = key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "medical-chatbot"

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

#Loading the index
docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

# Define the prompt template
prompt_template = """
Use the given information context to give an appropriate answer for the user's question.
If you don't know the answer, just say that you don't know the answer, but don't make up an answer.
Context: {context}
Question: {question}
Only return the appropriate answer and nothing else.
Helpful answer:
"""

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

# Define the configuration for the language model
config = {'max_new_tokens': 512, 'temperature': 0.8}
llm = CTransformers(model = 'TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q4_0.bin', model_type = 'llama', config = config)

# Initialize qa globally
qa = None

# Function to initialize RetrievalQA
def initialize_qa():
    global qa
    try:
        qa = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            retriever = docsearch.as_retriever(search_kwargs={'k': 2}),
            return_source_documents = True,
            chain_type_kwargs = {"prompt": PROMPT}  # Pass prompt template as chain_type_kwargs
        )
        print("QA system initialized successfully.")
    except Exception as e:
        print(f"Error initializing RetrievalQA: {e}")

# Initialize QA when the application starts
initialize_qa()

# Define routes for Flask application
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    if qa is None:
        return jsonify({"error": "QA system is not initialized"}), 500
    try:
        result = qa.invoke({'query' : msg})
        response = result.get("result", "No response found")
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

# Run Flask application
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8080, debug = True)