# 💰 Halal Gold Investment Assistant

This is a smart Q&A application built with Streamlit and powered by Google's Gemini-2.5-Flash model. The assistant can answer questions about Shariah-compliant gold investments based on a provided knowledge base.

The application uses a sophisticated **Retrieval-Augmented Generation (RAG)** architecture to ensure that its answers are accurate, relevant, and strictly grounded in the source text, preventing the AI from making up information.

## ✨ Features

-   **Conversational Q&A:** Ask questions in natural language about halal gold investing.
-   **Context-Aware Answers:** The AI uses a provided text file (`Halal_gold_info.txt`) as its sole source of knowledge.
-   **Advanced RAG Pipeline:** Implements a Parent/Child retrieval strategy for more accurate and contextually rich answers.
-   **Built with Modern Tools:** Leverages LangChain for orchestrating the RAG pipeline, ChromaDB for in-memory vector storage, and Streamlit for the user interface.

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) pattern to provide accurate answers.

1.  **Load Knowledge Base:** The `Halal_gold_info.txt` file is loaded into the application.
2.  **Logical Chunking (Parent Chunks):** The text is split into large, logical sections based on a `---` delimiter. These "parent documents" contain full, coherent context.
3.  **Vectorization (Child Chunks):** Each parent chunk is further split into smaller "child chunks." These small chunks are converted into numerical representations (embeddings) using Google's embedding model and stored in a Chroma vector database.
4.  **Retrieval:** When you ask a question, the app searches the vector database to find the most relevant *child chunks*.
5.  **Context Augmentation:** The system then retrieves the full *parent chunks* corresponding to the best child chunks. This provides the LLM with the precise information it needs, along with the surrounding context.
6.  **Answer Generation:** The user's question and the retrieved parent chunks are sent to the Gemini LLM with a carefully engineered prompt, instructing it to generate an answer based only on the provided information.

## 🚀 Setup and Installation (Local)

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Make sure you have a `requirements.txt` file with the necessary libraries.
    ```
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key:**
    Create a file named `.env` in the root of the project directory and add your Google API key to it:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_GOES_HERE"
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run Halal_Gold_rag_app.py
    ```
    The application should now be running and accessible in your web browser.

## ☁️ Deployment

This application is designed to be easily deployed for free on **Streamlit Community Cloud**.

1.  **Push to GitHub:** Ensure your repository contains the following files:
    -   `Halal_Gold_rag_app.py`
    -   `requirements.txt`
    -   `Halal_gold_info.txt`

2.  **Sign up on Streamlit Community Cloud:** Use your GitHub account to sign up at [share.streamlit.io](https://share.streamlit.io).

3.  **Deploy:**
    -   Click "**New app**" and select your GitHub repository.
    -   In the "**Advanced settings...**" section, add your `GOOGLE_API_KEY` to the **Secrets**.
    -   Click "**Deploy!**"

## 📂 Files in this Repository

-   `Halal_Gold_rag_app.py`: The main Python script containing the Streamlit application logic and RAG pipeline.
-   `Halal_gold_info.txt`: The knowledge base file. Contains all the information the AI will use to answer questions.
-   `requirements.txt`: A list of all the Python libraries required to run the application.
-   `README.md`: This file, providing information about the project.
