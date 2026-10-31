# MediAssist AI 🏥

> **⚠️ DISCLAIMER**: MediAssist AI is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## Overview

MediAssist AI is a locally-running medical information assistant built on Retrieval-Augmented Generation (RAG). It combines a curated knowledge base of medical documents with a local large language model (Ollama) to answer medical questions, maintain patient profiles, and provide voice-enabled interaction — all while keeping your data private and on-device.

---

## Features

- 🧠 **RAG Pipeline** – Retrieve relevant medical context before generating answers
- 🏠 **100% Local** – All AI runs via Ollama; no data leaves your machine
- 👤 **Patient Digital Twin** – Save patient profiles to personalise answers
- 🎤 **Voice I/O** – Speech-to-text input and text-to-speech output
- 📚 **Knowledge Base Management** – Ingest, view, and reset your medical document library
- 💬 **Chat History** – All sessions persisted in SQLite
- 📋 **Source Citations** – Every answer cites the retrieved document chunks
- 👋 **Greeting Detection** – Casual greetings handled instantly without RAG overhead
- 🎯 **Relevance Checking** – Off-topic questions get a clear "not in knowledge base" message instead of hallucinated content
- 📝 **Structured Answers** – Responses use Markdown headings, bullet lists, and bold terms for readability
- 🔒 **Privacy-first** – ChromaDB and SQLite stored locally

---

## Architecture Overview

```
User ──► Streamlit UI ──► RAGEngine
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          Embedder      VectorStore     OllamaClient
    (SentenceTransformers) (ChromaDB)  (REST → Ollama)
              │
          SQLite DB
      (patients, sessions,
        messages, docs)
```

- **Embedder** (`core/embedder.py`): SentenceTransformer `all-MiniLM-L6-v2`
- **VectorStore** (`core/vector_store.py`): ChromaDB persistent client
- **OllamaClient** (`core/llm_client.py`): HTTP client to Ollama REST API
- **RAGEngine** (`core/rag_engine.py`): Orchestrates embed → retrieve → generate
- **SQLite** (`db/`): Stores patient profiles, chat sessions, and messages

---

## Prerequisites

### 1. Python 3.11+

```bash
python --version  # must be 3.11 or higher
```

### 2. Ollama (for LLM inference)

**Windows:**
1. Download the installer from https://ollama.com/download/windows
2. Run `OllamaSetup.exe` and complete the installation
3. Verify Ollama is running or start it in a terminal:
   ```bash
   ollama serve          # starts the Ollama server at http://localhost:11434
   ```
4. Pull the default model:
   ```bash
   ollama pull llama3.2  # downloads llama3.2 model (~2GB)
   ```
5. Optionally pull additional supported models:
   ```bash
   ollama pull phi3      # Microsoft Phi-3 (fast, lightweight)
   ollama pull mistral   # Mistral 7B
   ```

> **Note on Model Availability & Offline Behavior**:
> - If Ollama is running but `llama3.2` is not yet pulled, MediAssist AI detects this and guides you with the exact command `ollama pull llama3.2`.
> - If Ollama is not running, the application displays a clear connection error message and instructions to start `ollama serve`. Unrelated document fragments are never returned as an answer.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/meghshah630/MediAssist-AI.git
cd MediAssist-AI

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional)
copy .env.example .env
# Edit .env as needed

# 5. Run the app
streamlit run app.py
```

On first launch, the app creates its local SQLite database. In the sidebar, click **🔄 Ingest Documents** to build the local ChromaDB from the bundled sample documents. The embedding model is downloaded the first time it is needed. Generated databases are excluded from Git and are not uploaded.

---

## How to Run

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

---

## Knowledge Base

MediAssist AI ships with **18 curated medical topic documents** (94 focused chunks at chunk_size=200 words):

| # | File | Topic | Category |
|---|------|--------|----------|
| 1 | `anemia_and_blood_disorders.txt` | Anemia and Blood Disorders | haematology |
| 2 | `asthma.txt` | Asthma | respiratory |
| 3 | `cardiovascular_disease.txt` | Cardiovascular Disease | cardiology |
| 4 | `covid19.txt` | COVID-19 | infectious_disease |
| 5 | `diabetes_type2.txt` | Type 2 Diabetes | endocrinology |
| 6 | `emergency_warning_signs.txt` | Emergency Warning Signs | emergency |
| 7 | `fever_and_common_infections.txt` | Fever and Common Infections | infectious_disease |
| 8 | `general_first_aid.txt` | General First Aid | emergency |
| 9 | `hydration_and_water_intake.txt` | Hydration and Water Intake | lifestyle |
| 10 | `hypertension.txt` | Hypertension | cardiology |
| 11 | `medication_interactions.txt` | Medication Interactions | pharmacology |
| 12 | `mental_health_anxiety.txt` | Mental Health – Anxiety | mental_health |
| 13 | `mental_health_depression.txt` | Mental Health – Depression | mental_health |
| 14 | `nutrition_and_healthy_eating.txt` | Nutrition and Healthy Eating | lifestyle |
| 15 | `preventive_care_and_screening.txt` | Preventive Care and Screening | preventive |
| 16 | `respiratory_infections.txt` | Respiratory Infections | respiratory |
| 17 | `sleep_health_and_disorders.txt` | Sleep Health and Disorders | lifestyle |
| 18 | `weight_management_and_obesity.txt` | Weight Management and Obesity | lifestyle |

> **Source classification**: All documents are locally authored by the development team drawing on established clinical guidelines (NHS, CDC, WHO, ADA, AHA). They have **not** been independently validated by a licensed clinician and are intended for educational demonstration only.

> **Note**: After cloning, run `python tools/kb_coverage_report.py` to verify KB integrity.

---

## How to Ingest Documents

1. Launch the app
2. In the sidebar, click **🔄 Ingest Documents**, OR
3. Navigate to the **📚 Knowledge Base** tab and click **Re-ingest Documents**

The app will load all `.txt` files from `data/sample_docs/`, clean encoding artifacts, assign topic metadata, embed chunks with SentenceTransformers, and store them in ChromaDB (94 chunks from 18 documents).

To add your own documents:
1. Place `.txt` files in `data/sample_docs/`
2. Add an entry to the `TOPIC_CATALOGUE` dict in `core/document_ingestion.py`
3. Click "Re-ingest Documents" in the app

To verify KB integrity after ingestion:
```bash
python tools/kb_coverage_report.py
```

---

## How to Use Voice Features

### Speech-to-Text (Voice Input)
1. In the chat tab, click the **🎤 Voice Input** button next to the input box.
2. Choose your input method:
   - **Browser Microphone (Recommended)**: Click the recording widget, speak your medical question, and stop. The audio is automatically sent, transcribed, and populated into the chat.
   - **Direct Microphone Capture**: Click "🎙️ Direct Microphone Capture" for local hardware recording (with 5-second timeout).
3. If permissions are missing, or no speech is detected, an informative notification guides you without crashing the app.

### Real-Time Read-Along & Voice Output (Web Speech API)
1. Every assistant response includes an interactive **Real-Time Read-Along Player**:
   - **Browser-Native Web Speech (`speechSynthesis`)**: Audio is rendered natively in the browser without generating or downloading any audio files.
   - **Playback Controls**: Complete **▶ Listen**, **⏸ Pause**, **▶ Resume**, and **⏹ Stop** controls.
   - **Word-Level Highlighting**: Automatically highlights the exact word being spoken in real time and smoothly scrolls the viewport to follow along.
   - **Clean Spoken Content**: Skips repeated legal disclaimers and source metadata during speech while keeping them visible in the chat bubble.
   - **Voice & Speed Selector**: Allows selecting from available English browser voices and adjusting playback speed (0.85x to 1.4x).
   - **Cross-Message Coordination**: Starting speech on any response automatically halts any previously playing speech across the page.
   - **Unsupported Browser Fallback**: Displays a helpful notice if Web Speech API is unavailable.

---

## Project Structure

```
MediAssist-AI/
├── app.py                    # Streamlit entry point
├── config.py                 # Global constants and settings
├── requirements.txt
├── README.md
├── decision.md               # Architecture Decision Records
├── flow.md                   # Mermaid flow diagrams
├── .env.example
├── core/
│   ├── embedder.py           # SentenceTransformer embeddings
│   ├── vector_store.py       # ChromaDB wrapper (self-healing collection management)
│   ├── llm_client.py         # Ollama REST API client
│   ├── document_ingestion.py # Load/clean/chunk/deduplicate documents + topic metadata
│   ├── rag_engine.py         # RAG pipeline orchestrator with greeting normalization
│   └── summarizer.py         # Optional summarization helper
├── db/
│   ├── database.py           # SQLite schema & connection helpers
│   ├── patient_repo.py       # Patient CRUD
│   └── chat_repo.py          # Chat session/message CRUD
├── voice/
│   ├── stt.py                # Speech-to-text (browser bytes + hardware mic)
│   ├── tts.py                # Text-to-speech helpers & normalization
│   └── read_along.py         # Browser-native Web Speech API real-time read-along generator
├── ui/
│   ├── sidebar.py            # Sidebar UI component
│   ├── chat_window.py        # Chat display with real-time read-along player
│   ├── voice_controls.py     # Microphone input & read-along player component
│   └── history_panel.py      # History browser component
├── tools/
│   └── kb_coverage_report.py # KB audit: doc count, chunk count, gap analysis
├── data/
│   └── sample_docs/          # 18 medical text files (.txt)
└── tests/                    # pytest test suite (205 tests)
    ├── test_voice.py         # STT & read-along Web Speech tests (23 tests)
    ├── test_rag_evaluation.py # RAG retrieval quality tests (17 tests)
    ├── test_chat_repo.py     # Chat sessions, titles, search & messages (20 tests)
    └── ...                   # Core module unit tests (145 tests)
```

Only `data/sample_docs/` is committed. `data/chroma_db/` and `data/mediassist.db` are generated locally when the app is used and are excluded from Git.

---

## Running Tests

```bash
# Run all tests (use python -m pytest on Windows if pytest is not on PATH)
python -m pytest tests/ -v --tb=short

# Run with coverage report
python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Run voice and read-along tests
python -m pytest tests/test_voice.py -v

# Run RAG retrieval quality tests only (requires SentenceTransformer, ~8s)
python -m pytest tests/test_rag_evaluation.py -v

# Run main suite, skip slow RAG eval tests
python -m pytest tests/ --ignore=tests/test_rag_evaluation.py -q --tb=short
```

**Current test count**: **205 passing** (165 core & database + 23 voice & read-along + 17 RAG evaluation)

---

## Offline & Error Handling

MediAssist AI handles Ollama connection states and missing models gracefully:

1. **Ollama Offline (Service Unreachable)**:
   - Sidebar displays 🔴 **Ollama: Offline** with expandable start instructions (`ollama serve`).
   - The Chat tab displays an informative warning banner.
   - If a question is asked while Ollama is offline, the system returns a clear error explaining that Ollama is unreachable at `http://localhost:11434` with instructions on how to start it. Unrelated document text is never dumped as the response.

2. **Model Not Pulled (HTTP 404)**:
   - If Ollama is running but the selected model (e.g. `llama3.2`) is missing, the sidebar displays an alert, and chat queries return a clear, actionable instruction with the exact command to run:
     ```bash
     ollama pull llama3.2
     ```
   - No crash or confusing raw text occurs.

3. **LLM Online & Model Ready**:
   - The status dot turns green 🟢 **Ollama: Online**.
   - Answers are generated using the user's question, patient profile, and retrieved medical context from ChromaDB.
   - Source citations are displayed in a collapsible expander below each generated answer.

---

## Chat Behaviour

### Greetings
Casual greetings (`hello`, `hi`, `HII`, `hey`, `heyy`, `good morning`, `good afternoon`, `good evening`, `howdy`, etc.) are detected instantly **before** the RAG pipeline runs. Repeated characters (e.g. `HII`, `hiii`, `heyyyy`) are normalized automatically. Greetings execute with zero vector search and zero LLM latency, operating cleanly even if the vector store is uninitialized or offline.

### Medical Questions
For medical questions, the full RAG pipeline runs:
1. The vector store is verified (reconnecting automatically if collection was reset).
2. If the collection is uninitialized or empty, a helpful message is returned directing the user to ingest documents.
3. The question is embedded and the vector store is queried.
4. Retrieved chunks are checked against a **cosine-distance relevance threshold (0.85)**.
5. **If relevant context found**: the LLM receives the context and generates a structured Markdown answer with a heading, short paragraphs, and bullet lists.
6. **If no relevant context found** (all distances > 0.85): the LLM is instructed to state that the MediAssist knowledge base does not contain sufficient information about the topic. No medical facts are invented.

### Answer Formatting
LLM answers use Markdown formatting:
- **Bold heading** summarising the topic
- Short paragraphs (2–4 sentences)
- Bullet lists for enumerations (symptoms, medications, steps)
- Bold key medical terms on first use

### Source Citations
Source citations are shown in a collapsible expander for answers grounded in retrieved context. Citations are suppressed for greetings and off-topic (irrelevant-context) answers to avoid misleading references.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `🔴 Ollama: Offline` in sidebar | Start the Ollama background service: `ollama serve` in a terminal |
| `ollama : The term 'ollama' is not recognized` | Ollama was installed into your user profile. Restart your terminal, or use `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2` |
| `⚠️ Model 'llama3.2' is not installed` | Run `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2` |
| `No microphone found` | Check microphone connection and Windows microphone privacy permissions |
| Slow first response | SentenceTransformer embedding model loads on first start; subsequent queries are fast |
| `Collection [...] does not exist` | Auto-healed in v1.0.6: VectorStore automatically re-binds to active collection. You can also click "Re-ingest Documents" in the Knowledge Base tab |
| `Knowledge base not initialized` | Click "Re-ingest Documents" in the Knowledge Base tab or sidebar |
| ChromaDB errors | Click "Re-ingest Documents" in the Knowledge Base tab to restore without deleting files |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your active Python environment |
| Database errors | Delete `data/mediassist.db` — it will be recreated automatically on startup |
| Stale cached LLM client after code update | Streamlit caches `@st.cache_resource` objects in memory. Stop and restart Streamlit to refresh |

---

## Configuration

Edit `config.py` to change:
- `EMBEDDING_MODEL` – Switch to a different SentenceTransformer model
- `DEFAULT_MODEL` – Change the default Ollama model
- `CHUNK_SIZE` / `CHUNK_OVERLAP` – Adjust document chunking parameters (default: 200 / 30 words; smaller chunks yield more topically focused retrieval)
- `TOP_K_RESULTS` – Number of retrieved chunks per query (default: 5)
- `EXTRA_DOCS_PATH` – List of extra directories to scan for additional `.txt` documents

> **Changing `CHUNK_SIZE`**: After any `CHUNK_SIZE` change, click "Re-ingest Documents" in the app (or run the programmatic ingestion script) so ChromaDB is rebuilt with the new chunk boundaries.

---

## Privacy

All data stays on your machine:
- Documents → `data/chroma_db/` (ChromaDB)
- Patient profiles & chat history → `data/mediassist.db` (SQLite)
- No telemetry or analytics

---

## License

No license file is included yet. Add your chosen license before allowing others to reuse or redistribute this project.

---

## Disclaimer

⚠️ MediAssist AI is for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for personal health decisions.
