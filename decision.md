# Architecture Decision Records (ADR) – MediAssist AI

This document records the key architecture decisions made during the design and development of MediAssist AI, including the context, alternatives considered, and rationale for each choice.

---

## ADR-001: ChromaDB over FAISS or Pinecone

**Status**: Accepted  
**Date**: 2024-01-15

### Context
MediAssist AI requires a vector database to store document embeddings and perform approximate nearest-neighbour (ANN) searches for document retrieval.

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| **FAISS** (Facebook AI Similarity Search) | Extremely fast; battle-tested; minimal dependencies | No built-in persistence; manual serialisation; no metadata filtering; purely an index library, not a DB |
| **Pinecone** | Managed service; scalable; enterprise-grade | Cloud-only; requires API key; costs money; violates privacy-first design principle; internet dependency |
| **Weaviate** | Full-featured graph DB; supports multi-modal | Heavy resource requirements; complex setup; overkill for a local app |
| **Qdrant** | High performance; rich filtering; Rust-based | Requires running a separate service; Docker dependency adds complexity |
| **ChromaDB** ✅ | Pure Python; persistent client built-in; metadata filtering; simple API; designed for RAG apps; no external service needed | Not as performant as FAISS at scale; single-node only |

### Decision
**ChromaDB** was chosen because:
1. **Zero external dependencies** – `PersistentClient` stores data to disk out of the box
2. **Metadata filtering** – Supports `where` clauses for filtering by source document
3. **RAG-native API** – Designed specifically for embedding + retrieval workflows
4. **Simple setup** – One `pip install chromadb`; no Docker, no server process
5. **Privacy-first** – Data stays entirely on-device
6. **Testing support** – `EphemeralClient()` for in-memory testing

### Consequences
- Performance will not scale beyond hundreds of thousands of documents on a single machine
- If the project grows to production scale, migration to Qdrant or Weaviate would be required

---

## ADR-002: Sentence Transformers over OpenAI Embeddings

**Status**: Accepted  
**Date**: 2024-01-15

### Context
Embedding quality directly determines retrieval accuracy in the RAG pipeline. We needed a text embedding solution for converting medical documents and queries to dense vectors.

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| **OpenAI Embeddings** (text-embedding-3-small / text-embedding-ada-002) | State-of-the-art quality; easy API | Requires paid API key; sends data to OpenAI servers; internet dependency; ongoing cost per token |
| **Cohere Embed** | High quality; multilingual | Cloud-only; API key required; cost |
| **all-MiniLM-L6-v2** (SentenceTransformers) ✅ | Free; local; fast; good quality for semantic search; 22M parameters; 384-dim vectors | Slightly lower quality than OpenAI's latest models |
| **BAAI/bge-large-en-v1.5** | Higher quality than MiniLM | Slower; larger model (~1.3 GB) |
| **all-mpnet-base-v2** | Excellent semantic quality | Slower than MiniLM |

### Decision
**`all-MiniLM-L6-v2`** via the `sentence-transformers` library was chosen because:
1. **Privacy** – Embeddings computed locally; no data sent to third parties
2. **Zero cost** – No API key or token charges
3. **Speed** – Fast inference even on CPU (~14k sentences/second on modern hardware)
4. **Quality** – Strong performance on semantic similarity benchmarks for English text
5. **Offline operation** – Once downloaded, works without internet access
6. **Standard** – Widely used in RAG and semantic search applications

### Consequences
- Retrieval quality may be slightly lower than cloud embedding models on highly nuanced queries
- The model must be downloaded once (~90 MB) on first run
- Upgrading to a larger model (e.g., `all-mpnet-base-v2`) requires only changing `EMBEDDING_MODEL` in `config.py` and re-ingesting documents

---

## ADR-003: Ollama over Cloud LLMs (OpenAI, Claude, Gemini)

**Status**: Accepted  
**Date**: 2024-01-15

### Context
The LLM is the core reasoning engine of MediAssist AI. It receives retrieved medical context and generates coherent, grounded answers.

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| **OpenAI GPT-4o / GPT-3.5-turbo** | Best-in-class quality; simple API | Sends sensitive patient data to OpenAI; per-token cost; internet required; potential HIPAA concerns |
| **Anthropic Claude** | Excellent reasoning; large context window | Cloud-only; API cost; privacy concerns |
| **Google Gemini** | Multimodal; competitive quality | Cloud-only; API cost |
| **Hugging Face Inference API** | Free tier available | Rate limited; still sends data externally |
| **Local Transformers** (llama.cpp, transformers library) | Full control | Complex setup; memory-intensive; requires custom inference code |
| **Ollama** ✅ | Simple REST API; manages local models; wide model support; fast; privacy | Requires local installation; GPU recommended for best performance; not available in cloud environments |

### Decision
**Ollama** was chosen because:
1. **Privacy-first** – Patient data never leaves the device
2. **No cost** – Free to run (hardware costs only)
3. **Simple API** – Standard REST endpoint (`/api/generate`, `/api/tags`)
4. **Model variety** – Supports llama3.2, phi3, mistral, and hundreds more
5. **Graceful degradation** – MediAssist falls back to retrieved context when Ollama is unavailable
6. **Active development** – Well-maintained, widely adopted in local AI community

### Consequences
- Performance heavily depends on local hardware (GPU strongly recommended)
- LLM must be installed separately by the user
- Ollama unavailability is handled by the **keyword-sentence extraction fallback** in `OllamaClient._offline_answer()` — the app remains usable even with no Ollama installed

### Offline & Error Handling Strategy (Updated)

When Ollama is unreachable or a model is missing:

1. **Model Missing (HTTP 404)**:
   - When Ollama is running but the selected model has not been downloaded, `OllamaClient.generate()` catches HTTP 404 and returns a clear, actionable message stating that the model is missing and provides the exact command: `ollama pull <model>`.
   - It does **not** dump unrelated text from the knowledge base.
   - `RAGEngine.answer()` identifies this state (`model_missing=True`), excludes false source citations, and does not cache the response so the user can immediately retry once pulled.

2. **Ollama Service Offline (Connection Refused / Timeout)**:
   - When the Ollama daemon is not running at `http://localhost:11434`, `OllamaClient.generate()` catches `ConnectionError`/`Timeout` and returns an explicit status message indicating that Ollama is unreachable, with instructions to run `ollama serve`.
   - The UI surfaces the offline state in the sidebar (🔴 status indicator) and chat header.
   - Unrelated medical document fragments are never substituted as the answer.

3. **Grounded Generation & Source Citations**:
   - When the service is reachable and the model is present, the RAG prompt instructs the model to answer using only the retrieved medical context.
   - Source citations are rendered cleanly in an expander below the verified answer.
   - All responses include the medical disclaimer.

---

## ADR-004: SQLite over PostgreSQL for Patient Data

**Status**: Accepted  
**Date**: 2024-01-15

### Context
MediAssist AI needs to persist patient profiles, chat sessions, and messages. A relational database is appropriate given the structured nature of this data.

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| **PostgreSQL** | Production-grade; ACID; concurrent access; rich types | Requires running a database server; complex setup; overkill for single-user local app |
| **MySQL / MariaDB** | Widely used; good performance | Same server overhead as PostgreSQL |
| **MongoDB** | Flexible schema; JSON-native | NoSQL overhead; no native relational integrity; server required |
| **TinyDB** | Pure Python; no server | Very slow for any real workload; not SQL |
| **SQLite** ✅ | Built-in to Python; zero setup; ACID; serverless; file-based; excellent tooling | Not suitable for high-concurrency multi-user scenarios; WAL mode partially mitigates this |

### Decision
**SQLite** was chosen because:
1. **Zero setup** – Bundled with Python's standard library (`import sqlite3`)
2. **Serverless** – No separate database process to install or manage
3. **File-based** – Easy to backup, copy, or delete (`data/mediassist.db`)
4. **ACID compliant** – WAL mode enabled for better concurrent read performance
5. **Sufficient for scope** – Single-user local application; no need for multi-client concurrency
6. **Excellent tooling** – DB Browser for SQLite, SQLiteZeal for inspection

### Consequences
- Concurrent write access from multiple Streamlit processes would cause locking issues (acceptable for single-user local deployment)
- Migration to PostgreSQL in the future would require updating `db/database.py` connection logic (SQLAlchemy would make this trivial)

---

## ADR-005: Streamlit over FastAPI + React

**Status**: Accepted  
**Date**: 2024-01-15

### Context
MediAssist AI needs a user interface that is interactive, easy to develop, and suitable for a local desktop application.

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| **FastAPI + React** | Professional SPA; highly customisable; component ecosystem | Requires separate frontend and backend; Node.js setup; build toolchain; significantly more development time |
| **FastAPI + Jinja2** | Lighter than React; Python backend | Still requires HTML/CSS/JS knowledge; less interactive than React |
| **Gradio** | Very easy; designed for ML demos; Hugging Face integration | Less flexible UI; limited layout options; not ideal for multi-tab applications |
| **Flask + Bootstrap** | Simple; flexible | Manual AJAX for interactivity; more boilerplate |
| **Streamlit** ✅ | Pure Python; hot reload; chat primitives; tabs; sidebar; media support; caching decorators | Less control over styling; re-runs on every interaction (managed with session_state); not ideal for complex SPAs |

### Decision
**Streamlit** was chosen because:
1. **Pure Python** – No HTML/CSS/JS knowledge required; aligns with project scope
2. **Rapid development** – Rich component library (chat messages, tabs, sidebar, expanders)
3. **Built-in caching** – `@st.cache_resource` and `@st.cache_data` for model and data caching
4. **Chat UI primitives** – `st.chat_message`, `st.chat_input` map directly to the use case
5. **Audio support** – `st.audio()` for TTS playback
6. **Hot reload** – Fast iteration during development
7. **Session state** – `st.session_state` manages in-memory state across interactions

### Consequences
- Full re-render on every interaction (Streamlit's execution model); managed with `st.session_state`
- Limited CSS customisation without injecting raw HTML via `st.markdown(unsafe_allow_html=True)`
- Not suitable for production web deployment with many concurrent users
- For a future web product, migrating the backend to FastAPI and building a React frontend would be the recommended path


---

## ADR-006: Greeting Detection & Relevance Thresholding

**Status**: Accepted  
**Date**: 2026-10-29

### Context
When users typed casual greetings ("hello", "hi", "good morning") the RAG pipeline ran a full vector search and LLM generation — wasting resources, producing awkward responses, and potentially returning spurious chunk citations.

Separately, when users asked questions outside the knowledge base (e.g. finance, technology), ChromaDB returned results with high cosine distances (> 0.85), and the LLM was left to hallucinate an answer rather than clearly stating the KB lacks information.

### Decision

**Greeting Detection** – Added `RAGEngine._is_greeting()` static method:
- Lowercases and strips punctuation from the question.
- Compares token set against a whitelist of greeting words (`hi`, `hello`, `hey`, `howdy`, `greetings`, `hiya`, `yo`, `aloha`, `namaste`).
- Also matches common greeting phrases (`good morning`, `good afternoon`, `good evening`, `how are you`, etc.).
- Only triggers when the message is ≤ 5 (single-word greetings) or ≤ 7 tokens (phrase greetings), preventing long medical questions starting with "hello" from being mis-classified.
- Short-circuits the pipeline: no embedder, no vector store, no LLM call — returns an instant friendly response with the disclaimer appended.
- Result includes `is_greeting=True` so the UI can skip the Sources expander.

**Relevance Threshold** – After vector retrieval, distances are inspected:
- Constant `_RELEVANCE_THRESHOLD = 0.85` (cosine distance; ChromaDB returns values 0–2).
- If **all** retrieved distances exceed the threshold, `context_irrelevant = True`.
- `_build_prompt()` now accepts a `context_irrelevant` flag; when True it injects an explicit `NOTE` instructing the LLM **not** to answer from general knowledge and to state the KB lacks sufficient information.
- Sources list is suppressed when `context_irrelevant=True` (no false citations).

### Alternatives Considered
| Approach | Pros | Cons |
|----------|------|------|
| Dedicated intent classifier (e.g. fine-tuned BERT) | More accurate | Adds heavy dependency; overkill for a local app |
| Ask the LLM to classify intent first | Flexible | Double LLM call; latency cost; hallucination risk |
| Keyword regex only | Simple | Misses phrase variants; harder to maintain |
| **Word-set + phrase-set (chosen)** | Zero latency; no extra deps; easy to extend | Won't catch highly creative greetings |

### Consequences
- Greetings return in < 1 ms (no I/O).
- Medical questions with highly relevant context still get full LLM answers and source citations.
- Off-topic questions trigger a clear "knowledge base does not contain sufficient information" message with no hallucinated content.
- System prompt updated with explicit Markdown formatting rules (bold heading, short paragraphs, bullet lists).
- `_build_prompt()` signature updated with optional `context_irrelevant: bool = False` parameter — backward-compatible.
- `app.py` `_CACHE_VERSION` bumped to `v1.0.4` to flush stale Streamlit-cached RAGEngine instances.
- 32 new tests added (10 `TestGreetingDetection` + 9 `TestGreetingHandling` + 3 `TestMissingContext` + 10 `TestModelMissingMessage` / `TestOfflineMessage`).
- Total test count: **143** (was 111).

---

## ADR-007: KB Expansion — Chunk Size Reduction, Topic Metadata, and Deduplication

**Status**: Accepted  
**Date**: 2026-10-29

### Context

After completing ADR-006, retrieval quality was still poor:
- **Only 8 documents** in `data/sample_docs/` covering a narrow set of topics.
- **17 chunks** in ChromaDB (chunk_size=512 → 1–2 chunks per 75-line document).
- A chunk containing hypertension lifestyle + medication text was retrieved for weight-loss questions because both topics share "diet", "exercise", "weight" vocabulary at the 512-word granularity.
- 10 major topic areas were entirely absent (fever, weight management, nutrition, anemia, sleep, preventive care, respiratory infections, emergency signs, depression, hydration), causing "insufficient context" errors.
- Chunk IDs were random UUIDs — every re-ingestion created new IDs, so `upsert` was ineffective and ChromaDB accumulated duplicate records.

### Decision

**1. Reduce chunk size: 512 → 200 words (overlap 64 → 30)**  
Smaller chunks are more topically focused. A 200-word chunk on "hypertension medications" has low cosine similarity to a weight-loss question compared to a 512-word chunk that also includes lifestyle and weight content.  
Changed in `config.py`: `CHUNK_SIZE = 200`, `CHUNK_OVERLAP = 30`.

**2. Add TOPIC_CATALOGUE in `core/document_ingestion.py`**  
A filename-stem → `(topic_label, category)` mapping attaches human-readable topic and category labels to every chunk as ChromaDB metadata. This enables future filtering by category (`cardiology`, `lifestyle`, `emergency`, etc.) and cleaner source citations (`[Source: Weight Management and Obesity]` instead of raw filename).

**3. Stable deterministic chunk IDs: `{stem}_chunk_{idx:04d}`**  
Chunk IDs are now derived from the document filename and sequential index (e.g., `hypertension_chunk_0000`). This is deterministic across ingestion runs. ChromaDB `upsert` is now effective; full collection reset is no longer mandatory (though reset is still performed on re-ingest for simplicity).

**4. SHA-256 content-hash deduplication**  
Each chunk carries a 16-char hex `content_hash` (SHA-256 prefix). The `ingest_directory()` function tracks seen hashes within a single pass and skips duplicate chunks. This prevents a document accidentally included twice from doubling its chunk count in ChromaDB.

**5. Encoding artifact cleanup**  
Windows-1252 → UTF-8 misread artefacts (e.g., `â€"` → `–`, `â€˜` → `'`) are normalised in `clean_text()` using a Unicode escape lookup table, ensuring embedding quality is not degraded by garbled characters.

**6. 10 new KB documents added (18 total)**

| New Document | Topic |
|---|---|
| `fever_and_common_infections.txt` | Fever and Common Infections |
| `weight_management_and_obesity.txt` | Weight Management and Obesity |
| `nutrition_and_healthy_eating.txt` | Nutrition and Healthy Eating |
| `anemia_and_blood_disorders.txt` | Anemia and Blood Disorders |
| `sleep_health_and_disorders.txt` | Sleep Health and Disorders |
| `preventive_care_and_screening.txt` | Preventive Care and Screening |
| `respiratory_infections.txt` | Respiratory Infections |
| `emergency_warning_signs.txt` | Emergency Warning Signs |
| `mental_health_depression.txt` | Mental Health – Depression |
| `hydration_and_water_intake.txt` | Hydration and Water Intake |

All 18 documents are locally authored by the development team drawing on established clinical guidelines (NHS, CDC, WHO, ADA, AHA). They have **not** been independently validated by a licensed clinician.

**7. `tools/kb_coverage_report.py` added**  
New audit script prints: doc manifest (name, size, topic, category), classification by category, ChromaDB chunk count and sample, content quality checks, gap analysis (missing topics), and retrieval configuration note. Run with `python tools/kb_coverage_report.py`.

**8. `tests/test_rag_evaluation.py` added (17 tests)**  
RAG evaluation test file covering:
- Retrieval quality (7 tests): weight-loss → weight_management doc; fever → fever doc; diabetes symptoms → diabetes doc; first aid → first_aid doc; blood pressure → hypertension doc; CPR → first_aid doc; flu → respiratory or fever doc.
- Insufficient context (2 tests): uncovered topic returns KB-lacks-info; greetings bypass retrieval.
- Citation integrity (3 tests): sources are subset of retrieved-chunk sources; no sources for greetings; sources contain topic metadata.
- Duplicate ingestion (2 tests): repeated ingestion creates no duplicates; identical content hash detection.
- Metadata integrity (3 tests): chunks carry topic, category, content_hash.

Tests use **real SentenceTransformer embeddings** + **ephemeral in-memory ChromaDB** + **mocked OllamaClient** — no Ollama required.

### Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| Download documents from NHS/CDC via HTTP | Real authoritative text | Network dependency; licence ambiguity; complex scraping; changes over time |
| Keep chunk_size=512, add more documents | Simple config change | Root cause not addressed; topic bleed still occurs at 512 words |
| Add ChromaDB `where` filter by category | Targeted retrieval | Requires knowing category before asking; complex UX; over-engineering for this scope |
| Fine-tune embedding model on medical text | Best retrieval quality | Requires GPU, labelled data, significant compute; out of scope |
| **Reduce chunk_size + add metadata (chosen)** | Addresses root cause; no new dependencies; extensible | Must re-ingest ChromaDB after chunk_size change |

### Consequences

- ChromaDB **must be re-ingested** after chunk_size change (old 512-word IDs are completely different from new 200-word IDs). A programmatic re-ingestion script is provided in the session summary.
- Chunk count increased from **17 → 94** (18 docs × ~5 chunks each at 200 words).
- `app.py` `_CACHE_VERSION` bumped to `v1.0.5` to flush stale Streamlit-cached instances.
- RAG retrieval quality tests all pass (17/17), including the key regression test: weight-loss question retrieves `weight_management_and_obesity` document, not `hypertension`.
- Total test count: **161** (144 core + 17 RAG evaluation).

---

## ADR-008: Self-Healing ChromaDB Collection Management & Repeated Greeting Normalization

**Status**: Accepted  
**Date**: 2026-10-29

### Context

After re-indexing ChromaDB to 94 chunks, the running Streamlit app threw runtime errors:
`Knowledge base query error: Collection [3a17d8c8-7424-4126-97a5-0a6f073dc2a6] does not exist.`
This occurred on queries including `"HII"` and `"what is fever?"`.

**Root Cause Analysis:**
1. **Stale Collection Handle in VectorStore & Streamlit Cache**:
   - ChromaDB collections have an internal immutable UUID. When `reset_collection()` or an out-of-process re-ingest runs, ChromaDB deletes the old collection UUID and creates a new one.
   - The running Streamlit server had cached `VectorStore` via `@st.cache_resource` (which lacked a cache buster) and in `st.session_state["rag_engine"]`.
   - `VectorStore` cached `self._collection` permanently upon initialization and never checked whether the collection reference was still valid. When calling `self._collection.query(...)` or `count()`, ChromaDB looked up the deleted UUID in SQLite and raised `chromadb.errors.NotFoundError`.
2. **Greeting Detection Missed Repeated Characters**:
   - `_is_greeting` in `RAGEngine` looked for exact matches of a static set (`{"hi", "hello", ...}`).
   - The user query `"HII"` became `"hii"`, which was not in the set.
   - Consequently, `"HII"` was treated as a medical question and routed to the broken vector store rather than bypassing retrieval.

### Decision

**1. Self-Healing VectorStore (`core/vector_store.py`)**:
- Added `get_collection(force_refresh=False)` which probes `self._collection` and automatically re-binds if stale.
- Added automatic retry logic in `query()`, `add_documents()`, and `collection_count()`: if a `NotFoundError` or "does not exist" exception occurs, the collection reference is refreshed from `self._client.get_or_create_collection(...)` and retried once.
- Added `refresh_collection()` public method.
- Added `is_empty() -> bool` to inspect initialization status safely.

**2. Enhanced Greeting Detection & Bypass (`core/rag_engine.py`)**:
- Extended `_GREETING_WORDS` to include repeated/informal variants (`"hii"`, `"hiii"`, `"heyy"`, `"helloo"`, `"hola"`, etc.).
- Added character-collapse normalization (`re.sub(r"([a-z])\1+", r"\1", w)`) so any number of repeated characters (`"hiiii"`, `"heyyyy"`) map to base greetings.
- Guaranteed that greetings bypass `embedder` and `vector_store` completely, executing even if ChromaDB is offline, uninitialized, or throwing errors.

**3. Safe Vector Store Initialization & Error Handling**:
- When `vector_store.is_empty()` is True, `rag_engine.answer()` returns a clear guidance message directing the user to ingest documents without calling the LLM or hallucinating.
- If `vector_store.query()` fails, an actionable error message is returned with instructions on how to re-ingest and recover.

**4. Streamlit Cache & Session-State Management (`app.py`)**:
- Bumped `_CACHE_VERSION = "v1.0.6"`.
- Added `_cache_ver: str = _CACHE_VERSION` to `get_vector_store()`, ensuring cache invalidation on version bump.
- Updated `get_rag_engine()` to check for changed `vector_store` instances.
- Re-ingestion in `ingest_documents()` forces `vector_store.refresh_collection()` and rebinds `st.session_state["rag_engine"]`.

### Consequences

- Zero downtime or collection-ID mismatch crashes when re-ingesting or updating the knowledge base.
- Greetings like `"HII"` return in < 1 ms without touching ChromaDB.
- Uninitialized/empty vector store states display clear actionable instructions rather than cryptic database errors.
- Real ChromaDB on disk tested and verified: 94 chunks indexed across 18 documents.
- Total test count: **175 passing tests** (up from 161).

---

## ADR-009: Robust Speech-to-Text Input, Per-Message TTS with Clean Speech Normalization

**Status**: Accepted  
**Date**: 2026-10-29

### Context

The original voice integration had several critical user experience limitations:
1. **Host-vs-Browser Microphone Limitation**:
   - `record_and_transcribe` used `sr.Microphone()`, which relies on `pyaudio` capturing audio directly on the host machine. In browser-based Streamlit deployments or Windows environments without pre-compiled PyAudio wheels, this threw `AttributeError` or missing device exceptions.
2. **TTS Clutter & Reading Disclaimers Aloud**:
   - The assistant answer string ends with a full disclaimer block (`---\n\n⚠️ DISCLAIMER: MediAssist AI is for educational...`) and source citations.
   - When text was read aloud, TTS clumsily spoke characters, URLs, emojis, and repeated legal disclaimers.
3. **Limited Voice Output & No Stop Button**:
   - Only the very last assistant response had a speaker button; previous messages in the chat history could not be played.
   - Users had no way to stop or dismiss speech playback once started.

### Decision

**1. Dual-Path Speech Recognition (`voice/stt.py`)**:
- Added `transcribe_audio_bytes(audio_data, lang="en-US")` which processes in-memory audio buffers (`io.BytesIO`) using `sr.AudioFile`, eliminating the PyAudio dependency for browser users.
- In `voice_controls.py`, integrated Streamlit's native `st.audio_input` for browser microphone capture with HTML5 permissions, wave visualization, and automatic conversion to text.
- Retained `record_and_transcribe()` with explicit exception handling for missing PyAudio, microphone permission denial, and ambient timeout errors.

**2. Clean Speech Normalization (`voice/tts.py`)**:
- Added `clean_text_for_speech(text: str) -> str`:
  - Strips disclaimer blocks (`⚠️ DISCLAIMER...`) and source references (`📖 Sources...`).
  - Strips markdown formatting (headers, bold/italic markup, bullet markers, horizontal rules, links).
  - Strips inline backticks while retaining the word text (` `paracetamol` ` -> `paracetamol`).
  - Strips emojis and decorative symbols.
  - Normalizes whitespace for natural acoustic cadence.

**3. Dual TTS Engine & Playback Control**:
- `text_to_audio_bytes` attempts high-quality `gTTS` first; if offline or network is unavailable, it transparently falls back to `pyttsx3` (Windows SAPI5) generating offline WAV audio bytes.
- Added playback state management in `ui/voice_controls.py` with a **⏹️ Stop Audio** toggle button to stop and dismiss speech playback at any time.

**4. Per-Response TTS Controls (`ui/chat_window.py`)**:
- Updated `render_chat_window` so every assistant message in the conversation displays its own `🔊 Listen` button.

### Consequences

- Voice input functions directly in modern browsers without requiring PyAudio to be compiled locally.
- Voice output sounds clear, natural, and free of disclaimers, raw bullets, and source lists.
- Users can listen to any response in their chat history and stop playback at will.
- New test module [`tests/test_voice.py`](file:///c:/Users/Megh%20Shah/Desktop/MediAssist-AI/tests/test_voice.py) with 17 unit and integration tests.
- Total test count: **192 passing tests** (158 core + 17 voice + 17 RAG evaluation).

---

## ADR-010: Real-Time Read-Along Voice Output via Web Speech API (`speechSynthesis`)

**Status**: Accepted  
**Date**: 2026-10-29

### Context

Previously, the voice output feature generated audio files (MP3 via gTTS or WAV via pyttsx3) on the server, loaded them into Streamlit `st.audio()`, and played static audio.
This approach had several notable limitations:
1. **No Read-Along Visual Feedback**: Users could not see which word was being spoken in real-time.
2. **Audio File Latency & Overhead**: Audio generation required server-side processing, disk/memory buffering, and audio file serialization.
3. **No Dynamic Voice Switching in UI**: Users could not choose different system voices on the fly.
4. **Playback Synchronization**: Stopping speech when another response started playing required manual user intervention.

### Decision

**1. Replace Audio File Generation with Browser-Native Web Speech API (`voice/read_along.py`)**:
- Implemented `prepare_read_along_data(text)` and `generate_read_along_component_html(text, component_id)` using `window.speechSynthesis` and `SpeechSynthesisUtterance`.
- **Zero audio file creation**: Audio is synthesized directly in the client's browser with zero network or file I/O latency.
- Full playback controls: **▶ Listen**, **⏸ Pause**, **▶ Resume**, and **⏹ Stop**.

**2. Synchronized Real-Time Word Highlighting & Auto-Scroll**:
- Response text is tokenized into word spans: `<span class="speech-word" id="sw_{idx}" data-start="{start}" data-end="{end}">{word}</span>`.
- Connected to `SpeechSynthesisUtterance.onboundary` with `event.name === 'word'`:
  - Matches the spoken `event.charIndex` to the word span in real time.
  - Applies `.speaking-active` highlight class (`#fde047` background, rounded borders, shadow).
  - Smoothly auto-scrolls the reading viewport to follow speech: `target.scrollIntoView({ behavior: 'smooth', block: 'nearest' })`.

**3. Content Preservation & Medical Disclaimer Separation**:
- Preserves full document formatting (headings `<h4>`/`<h5>`, lists `<ul><li>`, paragraphs `<p>`).
- Skips repeated medical disclaimers (`---\n\n⚠️ DISCLAIMER...`) and source citations (`📖 Sources...`) during speech while keeping them fully visible in the chat bubble below.

**4. Voice Selector, Speed Control & Cross-Message Sync**:
- Dynamic English voice selector populated from `window.speechSynthesis.getVoices()`.
- Speed multiplier dropdown (`0.85x`, `1.0x`, `1.2x`, `1.4x`).
- Inter-message coordination via `window.speechSynthesis.cancel()` and `BroadcastChannel('mediassist_read_along_sync')`: starting playback on any message automatically halts and resets all other message players.
- Graceful unsupported browser detection banner.

### Consequences

- Fluid, real-time read-along experience for all medical responses.
- Eliminates server-side audio file generation for the chat Listen feature.
- Preserves all RAG, Ollama, citations, patient profile, and voice input features.
- Test coverage expanded with 5 dedicated read-along component tests in `tests/test_voice.py`.
- Total test count: **205 passing tests** (165 core & database + 23 voice & read-along + 17 RAG evaluation).

---

## ADR-012: Profile-Aware RAG Personalization, ChatGPT-Style Sidebar Navigation & Professional UI

**Status**: Accepted  
**Date**: 2026-10-29

### Context

The application needed a major upgrade in usability and personalization:
1. **Sidebar Navigation**: Previous sessions were only listed in Tab 4 ("History"). The sidebar lacked conversation switching, "+ New Chat", conversation search, and date-based grouping.
2. **Patient Profile Integration**: Profiles were stored in SQLite, but the RAG engine relied on a disconnected sidebar form rather than the database records. Personalization lacked allergy cross-checking and safety warnings.
3. **UI Polish**: The UI had excessive emojis in labels, buttons, spinners, and tabs that detracted from a professional medical assistant aesthetic.

### Decision

1. **Database Schema & Migration (`db/database.py`, `db/chat_repo.py`)**:
   - Added `title` column to `chat_sessions` table with automatic migration.
   - Added `rename_session()`, `update_session_title()`, and `search_sessions()` to `db/chat_repo.py`.
   - On the first user message of a chat, the session title is automatically generated from the question.

2. **ChatGPT-Style Sidebar Navigator (`ui/sidebar.py`)**:
   - Replaced old profile/KB forms with a dedicated conversation navigator.
   - **+ New Chat**: Immediately initializes a clean conversation session.
   - **Search Input**: Live filtering of sessions by title/topic.
   - **Date Grouping**: Grouped by "Today", "Previous 7 Days", and "Older".
   - **Per-Conversation Controls**: Active session highlighting, inline title renaming, and delete confirmation.

3. **Profile-Aware RAG Personalization (`core/rag_engine.py`, `app.py`)**:
   - Connected `Patient Profile` management directly to RAG: users can activate any saved patient profile.
   - Enhanced system prompt with strict safety guidelines:
     - Cross-references retrieved treatments against patient allergies and current medications.
     - Never prescribes, diagnoses, or changes dosages.
     - Falls back cleanly to general evidence-based educational content when no profile is active.
   - Active profile banner in chat with a one-click "Clear Profile" action.

4. **Professional UI & Clean Typography (`app.py`, `ui/chat_window.py`, `ui/voice_controls.py`, `ui/history_panel.py`)**:
   - Cleaned up emojis from main tabs ("Chat", "Patient Profile", "Knowledge Base", "History"), spinner notices, and button labels.
   - Retained dark theme styling, responsive columns, and real-time read-along highlighting.

### Consequences

- All 4 main tabs remain fully functional with zero breaking changes to existing SQLite databases or ChromaDB collections.
- Full regression suite passing with **205 tests** (`python -m pytest tests/ -q`).

---

## ADR-011: Markdown Setext Header Normalization and Divider Line Speech Suppression

**Status**: Accepted  
**Date**: 2026-10-29

### Context

When LLMs (such as `llama3.2`) generate medical responses, they frequently use Setext-style underlines for headers:
```markdown
Fever
================

Definition and Temperature Thresholds
----------------------------------------
```
This caused two distinct issues:
1. **TTS / Web Speech Vocalization**: Speech synthesizers vocalize `================` as "equals equals equals..." or "separator line".
2. **Display Inconsistency**: Setext divider rows displayed raw ASCII underlines (`================`) in some chat bubbles and read-along views rather than clean modern Markdown headings.

### Decision

1. **RAG Pipeline Response Normalization (`core/rag_engine.py`)**:
   - Added `_normalize_response_formatting(text: str) -> str`:
     - Detects Setext H1 (`Header\n====`) and converts it to standard Markdown `## Header`.
     - Detects Setext H2 (`Header\n----`) and converts it to standard Markdown `### Header`.
     - Strips standalone divider rows (`[-=_*~]{2,}`) so raw underline lines never persist into the response or UI bubbles.
     - Collapses excessive blank lines.

2. **Read-Along Spoken Content Extraction (`voice/read_along.py`)**:
   - Updated `extract_spoken_content(text: str)`:
     - Automatically normalizes Setext headings to Markdown `#` headings before processing.
     - Strips any standalone divider rows and symbol runs `[-=_*~]{2,}` so they can never be tokenized into speech words or plain speech text.
   - Retained lookahead Setext underline consumption and non-alphanumeric token filtering in `prepare_read_along_data()`.

3. **Fallback TTS Audio Normalization (`voice/tts.py`)**:
   - Updated `clean_text_for_speech(text: str)` to convert Setext heading underlines and strip repeated symbol dividers.

4. **Streamlit Cache Version Bump (`app.py`)**:
   - Bumped `_CACHE_VERSION` to `"v1.0.9"` to ensure all running client sessions reload the normalized RAG engine and read-along components.

### Consequences

- Neither `================` nor `----------------` divider lines are ever spoken aloud by the browser's Web Speech API or server-side TTS.
- Chat responses display clean, modern Markdown headings (`##`, `###`) instead of raw equals/dash underlines.
- Word token offsets in the read-along player remain 100% aligned with the spoken speech text.
- Added unit tests in `tests/test_rag_engine.py` and `tests/test_voice.py`.
- Total test count: **205 passing tests** (165 core & database + 23 voice & read-along + 17 RAG evaluation).

---

## ADR-012: Profile-Aware RAG Personalization, ChatGPT-Style Sidebar Navigation & Professional UI

**Status**: Accepted  
**Date**: 2026-10-29

### Context

The application needed a major upgrade in usability and personalization:
1. **Sidebar Navigation**: Previous sessions were only listed in Tab 4 ("History"). The sidebar lacked conversation switching, "+ New Chat", conversation search, and date-based grouping.
2. **Patient Profile Integration**: Profiles were stored in SQLite, but the RAG engine relied on a disconnected sidebar form rather than the database records. Personalization lacked allergy cross-checking and safety warnings.
3. **UI Polish**: The UI had excessive emojis in labels, buttons, spinners, and tabs that detracted from a professional medical assistant aesthetic.

### Decision

1. **Database Schema & Migration (`db/database.py`, `db/chat_repo.py`)**:
   - Added `title` column to `chat_sessions` table with automatic migration.
   - Added `rename_session()`, `update_session_title()`, and `search_sessions()` to `db/chat_repo.py`.
   - On the first user message of a chat, the session title is automatically generated from the question.

2. **ChatGPT-Style Sidebar Navigator (`ui/sidebar.py`)**:
   - Replaced old profile/KB forms with a dedicated conversation navigator.
   - **+ New Chat**: Immediately initializes a clean conversation session.
   - **Search Input**: Live filtering of sessions by title/topic.
   - **Date Grouping**: Grouped by "Today", "Previous 7 Days", and "Older".
   - **Per-Conversation Controls**: Active session highlighting, inline title renaming, and delete confirmation.

3. **Profile-Aware RAG Personalization (`core/rag_engine.py`, `app.py`)**:
   - Connected `Patient Profile` management directly to RAG: users can activate any saved patient profile.
   - Enhanced system prompt with strict safety guidelines:
     - Cross-references retrieved treatments against patient allergies and current medications.
     - Never prescribes, diagnoses, or changes dosages.
     - Falls back cleanly to general evidence-based educational content when no profile is active.
   - Active profile banner in chat with a one-click "Clear Profile" action.

4. **Professional UI & Clean Typography (`app.py`, `ui/chat_window.py`, `ui/voice_controls.py`, `ui/history_panel.py`)**:
   - Cleaned up emojis from main tabs ("Chat", "Patient Profile", "Knowledge Base", "History"), spinner notices, and button labels.
   - Retained dark theme styling, responsive columns, and real-time read-along highlighting.

### Consequences

- All 4 main tabs remain fully functional with zero breaking changes to existing SQLite databases or ChromaDB collections.
- Full regression suite passing with **205 tests** (`python -m pytest tests/ -q`).
