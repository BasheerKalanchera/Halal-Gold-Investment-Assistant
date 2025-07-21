import asyncio
import sys

if sys.platform.startswith('linux') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import streamlit as st
import os
from dotenv import load_dotenv
import re
import uuid  # Import the uuid library

# LangChain and related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Imports for ParentDocumentRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

GOLD_ETF_FILE_PATH = "Halal_gold_info.txt"

raw_text_content = ""
try:
    with open(GOLD_ETF_FILE_PATH, "r", encoding="utf-8") as f:
        raw_text_content = f.read()
    if not raw_text_content.strip():
        st.warning(f"The file '{GOLD_ETF_FILE_PATH}' was loaded but appears to be empty.")
        st.stop()
except FileNotFoundError:
    st.error(f"Error: The file '{GOLD_ETF_FILE_PATH}' was not found.")
    st.info("Please make sure 'gold_etf_info.txt' is in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading '{GOLD_ETF_FILE_PATH}': {e}")
    st.stop()

# --- FINAL, ROBUST CHUNKING LOGIC ---
# This method splits the document into parent documents using "---" as a delimiter.
parent_documents_list = []
# Ensure the delimiter is on its own line for an accurate split
sections = raw_text_content.split('\n---\n')

for section_text in sections:
    if section_text.strip():
        # Use the first line of the section as its title for metadata
        first_line = section_text.strip().split('\n')[0]
        parent_documents_list.append(Document(
            page_content=section_text.strip(),
            metadata={"source": GOLD_ETF_FILE_PATH, "section_title": first_line}
        ))

if not parent_documents_list:
    st.warning("No sections found. Using the full document as a single parent.")
    parent_documents_list.append(Document(page_content=raw_text_content, metadata={"source": GOLD_ETF_FILE_PATH, "section_title": "Full Document"}))


# --- FAISS MultiVectorRetriever Setup ---
# This is the modern way to handle Parent-Child retrieval with FAISS.

# 1. Set up the vector store for the small child chunks
vectorstore = FAISS.from_texts(
    texts=[" "], # Start with a dummy document
    embedding=embeddings
)

# 2. Set up the store for the large parent documents
store = InMemoryStore()
id_key = "doc_id"

# 3. Create the MultiVectorRetriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# 4. Create unique IDs for each parent document
doc_ids = [str(uuid.uuid4()) for _ in parent_documents_list]

# 5. Create the child documents for searching
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
sub_docs = []
for i, doc in enumerate(parent_documents_list):
    _id = doc_ids[i]
    _sub_docs = child_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

# 6. Add the documents to the retriever
# This populates the vectorstore with the child chunks and the docstore with the parent chunks
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, parent_documents_list)))


# --- RAG Chain Definition ---
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

# --- Streamlit App ---
st.set_page_config(page_title="Halal Gold Investment Assistant", layout="centered")
st.title("💰 Halal Gold Investment Assistant")

user_query = st.text_area("Enter your question about Halal ways to invest in Gold here:", height=100)

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching and generating answer..."):
            try:
                result = qa_chain.invoke({"query": user_query})
                
                source_documents = result["source_documents"]
                answer = result["result"]

                st.subheader("Answer:")
                st.info(answer)

                                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to get an answer.")

st.markdown("---")
