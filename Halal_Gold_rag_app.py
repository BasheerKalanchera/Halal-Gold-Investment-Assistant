import streamlit as st
import os
from dotenv import load_dotenv
import hashlib
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

# --- Load .env for local development ---
load_dotenv()

# Fallback for local .env if Streamlit secrets are not defined
try:
    _ = st.secrets["gcp_service_account"]
except Exception:
    if st.secrets._secrets is None:
        st.secrets._secrets = {}
    st.secrets._secrets["gcp_service_account"] = {
        "type": os.getenv("GCP_TYPE"),
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GCP_PRIVATE_KEY").replace("\\n", "\n"),
        "client_email": os.getenv("GCP_CLIENT_EMAIL"),
        "client_id": os.getenv("GCP_CLIENT_ID"),
        "auth_uri": os.getenv("GCP_AUTH_URI"),
        "token_uri": os.getenv("GCP_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GCP_CLIENT_CERT_URL"),
    }

# --- Constants ---
GOLD_ETF_FILE_PATH = "Halal_gold_info.txt"

# --- Logging unanswered questions to Google Sheets ---
def log_unanswered_to_google_sheets(user_query, user_id="Anonymous"):
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("HalalGold_UnansweredLogs").sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, user_id, user_query])
    except Exception as e:
        st.warning(f"⚠️ Logging to Google Sheets failed: {e}")

# --- Embedding model ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )

# --- Retriever setup ---
@st.cache_resource
def create_retriever(file_path):
    _embeddings = load_embedding_model()
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text_content = f.read()
    parent_chunks = [Document(page_content=sec.strip())
                     for sec in raw_text_content.split('\n---\n') if sec.strip()]
    id_key = "doc_id"
    vectorstore = FAISS.from_texts([" "], _embeddings)
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    doc_ids = [hashlib.sha256(doc.page_content.encode()).hexdigest() for doc in parent_chunks]
    for i, doc in enumerate(parent_chunks):
        doc.metadata[id_key] = doc_ids[i]
    retriever.vectorstore.add_documents(parent_chunks)
    retriever.docstore.mset(list(zip(doc_ids, parent_chunks)))
    return retriever

# --- Load LLM and retriever ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
retriever = create_retriever(GOLD_ETF_FILE_PATH)
retriever.search_kwargs = {"k": 10}

# --- Prompt template ---
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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False, # Changed to False to remove debugging info
    chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
)

# --- Streamlit UI ---
st.set_page_config(page_title="Halal Gold Investment Assistant", layout="centered")

# --- Added Aisar to the title
st.title("💰 Aisar - Halal Gold Investment Assistant")

# --- Added Disclaimer to the top
st.warning("""
Disclaimer: This tool is not intended as investment or Shariah advice. Users should not rely on its responses for financial decisions and are advised to consult with a registered financial advisor or qualified Shariah scholar.
""")

# Optional user ID
if "user_id" not in st.session_state:
    st.session_state["user_id"] = ""
st.session_state["user_id"] = st.text_input("Enter your name or email (optional):", value=st.session_state["user_id"])

user_query = st.text_area("Enter your question about Halal ways to invest in Gold here:", height=100)

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching and generating answer..."):
            try:
                result = qa_chain.invoke({"query": user_query})
                answer = result["result"]
                st.subheader("Answer:")
                st.info(answer)

                fallback_message = "I don't have enough information in the provided documents to answer this question."
                if fallback_message in answer:
                    log_unanswered_to_google_sheets(user_query.strip(), st.session_state["user_id"] or "Anonymous")

                # The debugging information section is now removed.
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to get an answer.")

st.markdown("---")