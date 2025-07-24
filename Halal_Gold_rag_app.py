import streamlit as st
import os
from dotenv import load_dotenv
import re
import uuid

# LangChain and related imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
# Ensure the Google API Key is available
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file for local testing or in Streamlit secrets for deployment.")
    st.stop()

GOLD_ETF_FILE_PATH = "Halal_gold_info.txt"

# --- Caching Setup for Performance ---

@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace Embedding Model only once."""
    st.write("Cache miss: Loading embedding model...") # Log message for debugging
    return HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def create_retriever(file_path, _embeddings):
    """
    Creates an advanced MultiVectorRetriever to handle parent/child chunking.
    This ensures that the full context (like a complete table) is retrieved.
    """
    st.write("Cache miss: Creating MultiVectorRetriever...") # Log message for debugging
    
    # 1. Load the document
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text_content = f.read()
    
    # 2. Split the document into logical parent chunks using a simple and robust method.
    parent_chunks = []
    # Use Python's built-in split method for reliability
    for section in raw_text_content.split('\n---\n'):
        if section.strip(): # Ensure the section is not empty
            parent_chunks.append(Document(page_content=section.strip()))
    
    st.write(f"Created {len(parent_chunks)} parent chunks.") # Debugging line

    # 3. Setup the components for the MultiVectorRetriever
    id_key = "doc_id"
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    vectorstore = FAISS.from_texts([" "], _embeddings) # Initialize with a dummy document
    store = InMemoryStore()
    
    # 4. Create the retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    # 5. Generate unique IDs for each parent chunk
    doc_ids = [str(uuid.uuid4()) for _ in parent_chunks]
    
    # 6. Create smaller child chunks for searching
    child_chunks = []
    for i, doc in enumerate(parent_chunks):
        _id = doc_ids[i]
        _sub_docs = child_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        child_chunks.extend(_sub_docs)
    
    # 7. Add the documents to the retriever
    retriever.vectorstore.add_documents(child_chunks)
    retriever.docstore.mset(list(zip(doc_ids, parent_chunks)))
    
    return retriever

# --- Main App Logic ---

# Load models and create the retriever using the cached functions
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings_model = load_embedding_model()
retriever = create_retriever(GOLD_ETF_FILE_PATH, embeddings_model)

# The RAG chain is now built on every run, but this is very fast
# because the expensive parts (models and retriever) are already in the cache.
prompt_template = """
You are a helpful AI assistant specialized in Gold ETFs. Your task is to answer the user's question based strictly on the provided context.
Read the context carefully and synthesize the information that directly answers the question.
If the context provides a clear list of steps, methods, or types that are relevant to the user's question, it is helpful to present your answer as a numbered or bulleted list.
If the context does not contain information to answer the question, you must respond with: "I don't have enough information in the provided documents to answer this question." Do not use any external knowledge.

Context:
{context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- Streamlit UI ---
st.set_page_config(page_title="Halal Gold Investment Assistant", layout="centered")
st.title("💰 Halal Gold Investment Assistant")

user_query = st.text_area("Enter your question about Halal ways to invest in Gold here:", height=100)

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching and generating answer..."):
            try:
                result = qa_chain.invoke({"query": user_query})
                answer = result["result"]
                st.subheader("Answer:")
                st.info(answer)

                # --- Debugging Section ---
                st.subheader("Debugging Information")
                with st.expander("View Retrieved Documents"):
                    # The retriever now returns the parent chunks
                    if "source_documents" in result and result["source_documents"]:
                        st.write("The following PARENT chunks were retrieved and sent to the LLM:")
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Parent Chunk {i+1}**")
                            st.code(doc.page_content, language='text')
                            st.json(doc.metadata)
                    else:
                        st.write("No documents were retrieved for this query.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to get an answer.")

st.markdown("---")
