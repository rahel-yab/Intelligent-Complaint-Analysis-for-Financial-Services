# Intelligent Complaint Analysis for Financial Services

CrediTrust is a digital finance leader in East Africa. As we scale to over 500,000 users, the volume of manual feedback analysis has become a bottleneck for our Product and Compliance teams. This project implements a Retrieval-Augmented Generation (RAG) pipeline to allow stakeholders to query thousands of unstructured customer complaints using natural language.

🚀 Project Status: Halfway Milestone
We have completed the core data engineering, exploratory analysis, and initial vector indexing. We are currently integrating the LLM generation layer and building the final user interface.

Task 1: EDA & Preprocessing (Complete)

Task 2: Chunking & Vector Indexing (Complete)

Task 3: RAG Core Logic & LLM Integration (In Progress)

Task 4: Interactive UI Development (Pending)

🛠️ Technical Stack
Language: Python 3.9+

Data Handling: Pandas, Parquet

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Database: ChromaDB / FAISS

Orchestration: LangChain

UI: Streamlit / Gradio (Upcoming)

📂 Repository Structure
Plaintext

```text
├── data/
│   ├── raw/                 # Original CFPB dataset
│   └── processed/           # Cleaned and filtered complaints
├── vector_store/            # Persisted ChromaDB/FAISS indices
├── src/
│   ├── preprocessing.py     # Data cleaning and filtering scripts
│   ├── indexing.py          # Chunking and embedding logic
│   └── engine.py            # (In Progress) RAG retrieval & generation
├── notebooks/
│   └── EDA_and_Sampling.ipynb # Exploratory analysis & quality checks
├── app.py                   # (Planned) Streamlit/Gradio interface
└── requirements.txt         # Project dependencies
```

🔍 Implementation Details (Tasks 1 & 2)
1. Data Cleaning & EDA
We processed the CFPB dataset to focus specifically on CrediTrust's core offerings.

Filtered Products: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

Cleaning: Removed narratives with missing text, handled "XXXX" masked data common in financial reports, and normalized text for better embedding performance.

Key Insight: Our EDA revealed that Credit Card billing disputes represent ~40% of the total complaint volume, informing our retrieval priority.

2. Strategic Chunking & Embedding
To ensure the LLM receives precise context without exceeding token limits:

Strategy: Used RecursiveCharacterTextSplitter.

Parameters: chunk_size=500, chunk_overlap=50.

Model: all-MiniLM-L6-v2 was selected for its balance of speed and semantic accuracy, generating 384-dimension vectors.

Indexing: Built a stratified sample index (15k records) for local testing and integrated the full-scale pre-built vector store (1.37M chunks) for production-grade retrieval.

🏗️ Current Focus: Building the "Brain"
We are currently developing the src/engine.py which handles:

Semantic Retrieval: Converting a user question (e.g., "Why are customers frustrated with loan interest rates?") into a vector.

Context Assembly: Fetching the top-5 most relevant complaint narratives from ChromaDB.

Grounded Generation: Passing the context to an LLM (Mistral/Llama) with a strict "don't hallucinate" system prompt.

🛠️ Installation & Setup
To set up the environment and run the current preprocessing scripts:

Clone the repo:

Bash

git clone https://github.com/your-username/rag-complaint-chatbot.git
cd rag-complaint-chatbot
Install Dependencies:

Bash

pip install -r requirements.txt
Run Preprocessing (Task 1):

Bash

python src/preprocessing.py
📈 Next Steps
LLM Fine-tuning: Refining the prompt template to ensure the assistant maintains a professional, financial-analyst tone.

Source Transparency: Implementing a feature in the UI to display exactly which complaint_id was used to generate an answer.

Evaluation: Running a qualitative test suite against 10 "Golden Questions" provided by the Product Management team.