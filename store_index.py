from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#connect to pinecode
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Create a new index with the specified name and dimensionality
if 'medical-chatbot' not in pc.list_indexes().names():
    pc.create_index(
        name = 'medical-chatbot',
        dimension = 384,
        metric = 'cosine',
        spec = ServerlessSpec(
            cloud = "aws", 
            region = 'us-east-1'
        )
    )

index_host ="https://medical-chatbot-nvnpc5c.svc.aped-4627-b74a.pinecone.io"
index = pc.Index("medical-chatbot", host = index_host)
# Prepare vectors for upsert
vectors = []
for idx, embedding in enumerate(embeddings):
    vectors.append({"id": str(idx), "values": embedding})

namespace = "real"
pc = Pinecone(api_key = os.environ.get("PINECONE_API_KEY"))

index_name = "medical-chatbot"


try:
    docsearch = PineconeVectorStore.from_documents(
        documents = text_chunks,
        embedding = embeddings,
        index_name = index_name,
        namespace = namespace
    )
except Exception as e:
    print(f"Error initializing PineconeVectorStore: {e}")
    docsearch = None