import streamlit as st
import os
from dotenv import load_dotenv
import re
import uuid
import hashlib  # For deterministic IDs

# LangChain and related imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Load environment variables
load_dotenv()

# --- Configuration ---
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please set it in .env or Streamlit secrets.")
    st.stop()

GOLD_ETF_FILE_PATH = "Halal_gold_info.txt"

# --- Caching Setup ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def create_retriever(file_path):
    """
    Uses entire Q&A block as a single chunk. No parent-child splitting.
    """
    _embeddings = load_embedding_model()

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text_content = f.read()

    parent_chunks = []
    for section in raw_text_content.split('\n---\n'):
        if section.strip():
            parent_chunks.append(Document(page_content=section.strip()))

    id_key = "doc_id"
    vectorstore = FAISS.from_texts([" "], _embeddings)
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    doc_ids = [hashlib.sha256(doc.page_content.encode()).hexdigest() for doc in parent_chunks]

    # Treat entire Q&A as single retrievable chunk
    child_chunks = []
    for i, doc in enumerate(parent_chunks):
        _id = doc_ids[i]
        doc.metadata[id_key] = _id
        child_chunks.append(doc)

    retriever.vectorstore.add_documents(child_chunks)
    retriever.docstore.mset(list(zip(doc_ids, parent_chunks)))

    return retriever

# --- Load Models & Retriever ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
retriever = create_retriever(GOLD_ETF_FILE_PATH)
retriever.search_kwargs = {"k": 8}  # ✅ Increase recall

# --- Prompt Template ---
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
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
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

                # Debugging
                st.subheader("Debugging Information")
                with st.expander("View Retrieved Documents"):
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
