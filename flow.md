# MediAssist AI – System Flow Diagrams

This document contains Mermaid diagrams describing the key data flows and processes in MediAssist AI.

---

## 1. RAG Pipeline Flow

> **Updated (v1.0.6):** Added greeting normalization for repeated letters (e.g. "HII", "heyy"), self-healing VectorStore reconnection on stale collection IDs, and uninitialized/empty KB guidance.

```mermaid
flowchart TD
    A["User Question"] --> B{"Is Greeting?\n(hello, hi, HII, hey,\nrepeated chars,\ngood morning…)"}
    B -- "Yes" --> Z["👋 Friendly greeting response\n(no vector search, no LLM call)\nis_greeting=True"]
    Z --> DISC["Append Disclaimer"]
    B -- "No" --> CHK{"VectorStore empty\nor uninitialized?"}
    CHK -- "Yes" --> EMPTY["⚠️ Show guidance:\n'Knowledge base not initialized.\nPlease ingest documents.'"]
    EMPTY --> DISC
    CHK -- "No" --> C["Embedder\n(SentenceTransformer)"]
    C --> D["Query Vector\n(384-dim float list)"]
    D --> E["VectorStore.query()\n(Auto-refreshes stale collection ID)"]
    E --> REL{"All distances\n> 0.85?"}
    REL -- "Yes (irrelevant)" --> NOINFO["Prompt: KB lacks info\nDo NOT answer from general knowledge\nSources suppressed"]
    REL -- "No (relevant)" --> F["Prompt Builder\n(system prompt + context + history)"]
    NOINFO --> I["OllamaClient.generate()\n(POST /api/generate)"]
    G["Patient Profile\n(digital twin)"] --> F
    H["Chat History\n(last 6 messages)"] --> F
    F --> I
    I --> J{"Ollama Status"}
    J -- "HTTP 200 (Success)" --> K["LLM Response\n(raw generation)"]
    K --> NORM["Normalize Response Formatting\n(Convert Setext underlines to ## / ###;\nstrip raw divider rows)"]
    J -- "HTTP 404 (Model Missing)" --> L["Model Missing Message\n(show 'ollama pull <model>')"]
    J -- "Connection Error / Timeout" --> M["Offline Error Message\n(show 'ollama serve' setup)"]
    NORM --> N["Include Sources & Citations\n(only if relevant context used)"]
    L --> O["Suppress False Sources"]
    M --> O
    N --> DISC
    O --> DISC
    DISC["Append Medical Disclaimer"] --> Q["Final Answer Delivery"]
    Q --> R["Streamlit Chat UI\n(st.chat_message)"]
    Q --> S["SQLite\n(chat_messages table)"]
    Q --> T{"llm_online?"}
    T -- "Yes" --> U["Cache answer\n(session_state dict)"]
    T -- "No" --> V["Skip cache\n(retry on next query)"]
```

---

## 2. Document Ingestion Flow

> **Updated (v1.0.5):** Reduced chunk size to 200 words (30 overlap), added SHA-256 content-hash deduplication, deterministic chunk IDs, and topic/category metadata tagging.

```mermaid
flowchart TD
    A["User clicks\n'Ingest Documents'"] --> B["ingest_directory()\ndata/sample_docs/*.txt"]
    B --> C{"For each .txt file"}
    C --> D["load_document()\nread file content"]
    D --> E["clean_text()\nclean whitespace &\nfix encoding artefacts"]
    E --> F["chunk_text()\nsliding window chunking\n(200 words, 30 overlap)"]
    F --> DEDUP{"Check content_hash\n(SHA-256 deduplication)"}
    DEDUP -- "Duplicate" --> SKIP["Skip chunk"]
    DEDUP -- "Unique" --> TAG["Assign metadata:\ntopic, category, chunk_index, stable ID"]
    TAG --> G["List of Chunks\n{text, source, chunk_id,\ntopic, category, content_hash}"]
    G --> H["Embedder.embed_texts()\nbatch SentenceTransformer inference"]
    H --> I["Embedding Vectors\n(N × 384 matrix)"]
    I --> J["VectorStore.reset_collection()\nclear existing data"]
    J --> K["VectorStore.add_documents()\ndocs + embeddings + metadata + ids"]
    K --> L["ChromaDB\npersistent storage\ndata/chroma_db/ (94 chunks)"]
    K --> M["record_ingested_document()\nSQLite ingested_documents table"]
    L --> N["Success: 94 chunks indexed across 18 documents"]
```

---

## 3. User Interaction Flow

> **Updated (v1.0.8):** Upgraded to browser-native Web Speech API (`speechSynthesis`) read-along player with real-time word highlighting, auto-scroll, voice & speed selectors, and zero audio file generation.

```mermaid
flowchart TD
    A["User opens app\n(streamlit run app.py)"] --> B["init_db()\nSQLite tables created"]
    B --> C["Cache Resources\nEmbedder + VectorStore + OllamaClient"]
    C --> D["Streamlit UI\nDisclaimer banner + Sidebar + Tabs"]
    D --> E{"User Input\nMethod"}
    E -- "Text" --> F["st.chat_input\nUser types question"]
    E -- "Voice (Browser)" --> G1["🎤 Voice Input\nst.audio_input (browser mic)"]
    G1 --> H1["transcribe_audio_bytes()\nGoogle Web Speech API"]
    H1 --> F
    E -- "Voice (Hardware)" --> G2["🎙️ Direct Capture\nrecord_and_transcribe()"]
    G2 --> H2["Hardware mic capture\nwith timeout & permission checks"]
    H2 --> F
    F --> GREET{"Greeting?"}
    GREET -- "Yes" --> GRESP["Instant friendly response\n(no RAG, no LLM)"]
    GREET -- "No" --> I["RAGEngine.answer()\nEmbed → Retrieve → Relevance Check → Generate"]
    GRESP --> K["Display assistant response"]
    I --> K
    K --> L["add_message('assistant', ...)\nSQLite persist"]
    K --> RA["Browser-Native Read-Along Player\n(window.speechSynthesis)"]
    RA --> CLEAN["extract_spoken_content()\nOmit disclaimers, sources &\nstrip all divider rows (====, ----)"]
    CLEAN --> CTRL["Controls:\n▶ Listen | ⏸ Pause | ▶ Resume | ⏹ Stop\n🗣 Voice Selector | ⚡ Speed (0.85x-1.4x)"]
    CTRL --> SPEAK["Browser Speech Synthesis\n(natural English voice)"]
    SPEAK --> HIGHLIGHT["Real-Time Word Highlighting\n(utterance.onboundary) +\nSmooth Auto-Scroll Into View"]
    SPEAK --> SYNC["Cross-Message Sync\nBroadcastChannel cancels previous speech"]
    L --> M{"User Action"}
    M -- "New question" --> E
    M -- "+ New Chat (Sidebar)" --> NC["_start_new_chat()\nReset active session"]
    NC --> E
    M -- "Switch Chat (Sidebar)" --> SC["_switch_to_session()\nLoad messages from DB"]
    SC --> E
    M -- "View profile" --> O["Patient Profile Tab"]
    M -- "View KB" --> P["Knowledge Base Tab"]
    M -- "View history" --> Q["History Tab"]
    O --> R["CRUD: create/view/update/set active\nSQLite patients table"]
    P --> S["View chunks, re-ingest, reset"]
    Q --> T["list_sessions / get_messages\nSQLite chat_sessions/chat_messages"]
```

---

## 4. Data Storage Flow

```mermaid
flowchart LR
    subgraph app["MediAssist AI Application"]
        A["app.py\n(Streamlit)"]
        B["RAGEngine"]
        C["Embedder"]
        D["OllamaClient"]
    end

    subgraph storage["Local Storage"]
        subgraph chroma["ChromaDB\ndata/chroma_db/"]
            E["Collection:\nmedical_docs"]
            F["Document chunks\n+ embeddings\n+ metadata"]
        end
        subgraph sqlite["SQLite\ndata/mediassist.db"]
            G["patients\ntable"]
            H["chat_sessions\ntable"]
            I["chat_messages\ntable"]
            J["ingested_documents\ntable"]
        end
        subgraph files["File System\ndata/sample_docs/"]
            K["hypertension.txt"]
            L["diabetes_type2.txt"]
            M["asthma.txt"]
            N["...other docs..."]
        end
    end

    subgraph ollama["Ollama\n(localhost:11434)"]
        O["llama3.2 / phi3\n/ mistral"]
    end

    A -- "Patient CRUD" --> G
    A -- "Session CRUD" --> H
    A -- "Message CRUD" --> I
    A -- "Doc records" --> J
    A -- "Ingest trigger" --> K
    K & L & M & N -- "Read .txt" --> B
    B -- "Store embeddings" --> E
    E --> F
    B -- "Query top-K" --> E
    C -- "Embed query/texts" --> B
    D -- "POST /api/generate" --> O
    O -- "LLM response" --> D
```
