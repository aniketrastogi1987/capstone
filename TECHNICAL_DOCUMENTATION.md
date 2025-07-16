# Patent Analysis Pipeline: Complete Technical Documentation

## 🔬 Overview

This pipeline processes, analyzes, and serves patent data using a sophisticated combination of:
- **Filtering**: Extract G06 patents (AI/ML, computer vision) from raw data
- **RAG**: LightRAG for retrieval-augmented generation
- **LLM**: qwen2.5:14b-instruct via Ollama
- **Storage**: Multi-layer (JSON, Neo4j, SQLite)
- **Backup**: Real-time monitoring and backup systems
- **Guardrails**: Content safety and quality validation
- **Monitoring**: PostgreSQL + Grafana for real-time metrics
- **Evaluation**: ROUGE, relevance, coherence scoring

## 📚 Key Libraries and Packages

### Core Dependencies
```python
# Core system
os, sys, json, logging, argparse, requests, time, shutil, pathlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import hashlib
```

### Data Science & ML
```python
# Data processing
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

# LLM & NLP
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
sentence-transformers
rouge-score
```

### Database & Storage
```python
# Databases
neo4j>=5.0.0
sqlite3  # Built-in
psycopg2  # PostgreSQL
```

### Web & UI
```python
# Web interface
gradio>=3.40.0
requests>=2.31.0
```

### Monitoring & Utilities
```python
# Monitoring
prometheus-client
psutil
python-dotenv>=1.0.0

# Utilities
tqdm>=4.65.0
```

### LightRAG Integration
```python
# LightRAG
lightrag-hku[api]>=0.1.0
pipmaster
PyJWT
pyuca
```

## 🏗️ System Architecture

```
Raw Patent Data → Filtering → LightRAG → Chatbot → User
     ↓              ↓         ↓         ↓
   JSON Files   Optimized   RAG + LLM  Guardrails
     ↓              ↓         ↓         ↓
   Backup        SQLite    Neo4j      Evaluation
     ↓              ↓         ↓         ↓
   Monitoring   Grafana   Metrics    PostgreSQL
```

## 📁 Project Structure

```
patent_project/
├── main.py                          # Main orchestration script
├── requirements.txt                  # Python dependencies
├── .env                            # LightRAG configuration
│
├── filtering/                       # Step 1: Patent Filtering
│   ├── __init__.py
│   └── filter_g06_patents_optimized.py
│
├── lightrag_integration/            # Step 2: LightRAG Integration
│   ├── __init__.py
│   ├── lightrag_config.py
│   ├── start_lightrag_server.py
│   ├── integrate_lightrag_g06_patents_sequential.py
│   └── lightrag_uploader.py
│
├── chatbot/                         # Step 3: Interactive Chatbot
│   ├── __init__.py
│   ├── patent_chatbot.py
│   ├── neo4j_fallback.py           # Neo4j fallback for LightRAG
│   ├── guardrails_validator.py     # Content safety validation
│   └── patent_analyzer.py          # Patent analysis logic
│
├── monitoring/                      # Performance Monitoring
│   ├── __init__.py
│   ├── postgres_monitor.py         # PostgreSQL monitoring system
│   ├── simple_postgres_dashboard.json # Grafana dashboard config
│   ├── test_postgres_monitoring.py # PostgreSQL monitoring tests
│   └── verify_monitoring_flow.py   # Monitoring flow verification
│
├── backup/                          # Backup System
│   ├── manage_backup.py            # Backup system management
│   ├── backup_query_tool.py        # Query backed up data
│   ├── lightrag_backup_monitor.py  # Backup monitoring system
│   └── test_neo4j_backup.py       # Neo4j backup tests
│
├── evaluation/                      # Evaluation Metrics
│   ├── __init__.py
│   └── evaluate_responses.py       # Response evaluation system
│
├── hupd_extracted/                  # Source patent data
├── hupd_processed/                  # Filtered and optimized G06 patents
├── rag_storage/                     # LightRAG persistent storage
└── lightrag_backup.db               # Backup SQLite database
```

## 🔄 Pipeline Steps

### 1. **Filtering (G06 Patent Extraction)**

**Purpose**: Extract and optimize G06 patents (AI/ML, computer vision) from raw data.

**Key Class**: `OptimizedG06PatentFilter`  
**Location**: `filtering/filter_g06_patents_optimized.py`

**How it works**:
- Scans all JSON files in the input directory recursively
- Filters for patents with `main_ipcr_label` starting with `G06N` or `G06V`
- Keeps only essential fields, truncates long text for efficiency (90-99% size reduction)
- Creates optimized versions in `hupd_processed/`

**Critical Code**:
```python
def has_g06_main_ipcr_label(self, patent_data):
    main_ipcr_label = patent_data.get('main_ipcr_label', '')
    target_labels = ["G06N", "G06V"]  # AI/ML, Computer Vision
    return any(main_ipcr_label.startswith(target) for target in target_labels)

def optimize_patent_data(self, patent_data):
    # Keep only essential fields
    essential_fields = {
        'application_number', 'publication_number', 'title',
        'abstract', 'summary', 'main_ipcr_label', 'ipcr_labels'
    }
    # Truncate text fields for efficiency
    text_limits = {'title': 500, 'abstract': 1000, 'summary': 2000}
```

**Output**: Optimized JSON files in `hupd_processed/` with 90-99% size reduction.

### 2. **LightRAG Integration**

**Purpose**: Index filtered patents into LightRAG for RAG-based retrieval.

**Key Components**:
- `lightrag_integration/integrate_lightrag_g06_patents_sequential.py`
- `lightrag_integration/lightrag_uploader.py`
- `lightrag_integration/lightrag_config.py`

**How it works**:
- Sequentially uploads each filtered patent to LightRAG via REST API
- Monitors server health, retries on failure, tracks progress
- Uses sequential processing to avoid overwhelming the server

**Critical Code**:
```python
def upload_document(self, file_path):
    payload = {
        "text": text_content,
        "metadata": {
            "source": "Harvard USPTO Dataset (HUPD)",
            "patent_type": "G06 (Computing; Calculating; Counting)"
        }
    }
    response = requests.post(
        f"{LIGHTRAG_BASE_URL}/documents/text",
        json=payload,
        timeout=300
    )
```

**Configuration**:
- **LLM**: `qwen2.5:14b-instruct` via Ollama
- **Embeddings**: `bge-m3:latest`
- **Storage**: JSON (primary), Neo4j (graph), NanoVectorDB (vectors)
- **Server**: `localhost:9621`

### 3. **Storage & Backup Architecture**

#### a. LightRAG JSON Storage (Primary)
- **Location**: `rag_storage/`
- **Files**: 
  - `kv_store_full_docs.json` - Complete patent documents
  - `kv_store_text_chunks.json` - Document text chunks
  - `kv_store_doc_status.json` - Processing status
  - `vdb_chunks.json` - Vector embeddings
  - `vdb_entities.json` - Entity data
  - `vdb_relationships.json` - Entity relationships

#### b. SQLite Backup (Automatic)
- **File**: `lightrag_backup.db`
- **Monitored by**: `backup/lightrag_backup_monitor.py`
- **How it works**: Monitors JSON files for changes, backs up to SQLite every 30 seconds

**Critical Code**:
```python
def backup_json_file(self, file_path, table_name):
    # Read JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Insert/update into SQLite
    for doc_id, doc_data in data.items():
        cursor.execute('''
            INSERT OR REPLACE INTO full_docs 
            (doc_id, content, last_updated, file_hash)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, content, current_time, file_hash))
```

#### c. Neo4j Graph Storage (Entity Relationships)
- **Role**: Stores entity relationships, not full documents
- **Node Types**: `document`, `category`, `organization`, `person`, `technology`
- **Access**: Used for fallback queries if LightRAG is down
- **Current Status**: 26 document nodes, 7,566 total nodes, 6,750 relationships

### 4. **Chatbot System**

**Purpose**: Interactive patent Q&A with RAG, LLM, and fallback capabilities.

**Key File**: `chatbot/patent_chatbot.py`

**Features**:
- RAG retrieval from LightRAG (primary method)
- Fallback to Neo4j if LightRAG is unavailable
- Guardrails for content safety and quality
- Evaluation metrics (ROUGE, relevance, coherence)
- Gradio web UI and CLI modes
- Real-time streaming responses

**Critical Code**:
```python
def _get_rag_context(self, query):
    # First check if LightRAG is available
    if not self._check_lightrag_availability():
        print("📚 LightRAG unavailable, using Neo4j fallback...")
        return self.neo4j_fallback.generate_fallback_response(query)
    
    # Try LightRAG first
    response = requests.post(
        f"{self.lightrag_url}/api/chat",
        json=payload,
        timeout=120
    )
```

**Chatbot Options**:
- Enable/disable guardrails
- View monitoring stats
- Save/load chat history
- Evaluate responses
- Switch between Gradio UI and CLI modes

### 5. **Guardrails System**

**Purpose**: Ensure chatbot responses are safe, relevant, and polite.

**Key File**: `chatbot/guardrails_validator.py`

**Checks**:
- **Profanity**: Regex-based detection of profane words
- **Topic Relevance**: Keyword density analysis for patent-related content
- **Politeness**: Detection of impolite patterns and professional language

**Critical Code**:
```python
def check_profanity(self, text):
    profanity_patterns = [
        r'\b(fuck|shit|damn|hell|bitch|ass|dick|pussy|cunt|cock|whore|slut)\b',
        r'\b(f\*ck|s\*it|d\*mn|h\*ll|b\*tch|a\*s|d\*ck|p\*ssy|c\*nt|c\*ck|wh\*re|sl\*t)\b'
    ]
    for pattern in profanity_patterns:
        if re.search(pattern, text.lower()):
            return False, 0.0
    return True, 1.0

def check_topic_relevance(self, text):
    patent_keywords = [
        'patent', 'invention', 'claim', 'prior art', 'uspto',
        'technology', 'innovation', 'device', 'method', 'system'
    ]
    keyword_count = sum(1 for keyword in patent_keywords 
                       if keyword.lower() in text.lower())
    relevance_score = min(1.0, keyword_count / max(1, len(text.split()) / 10))
    return relevance_score > 0.1, relevance_score
```

**Scoring**: Each check returns a score from 0.0 to 1.0, with overall score as average.

### 6. **Evaluation Metrics**

**Purpose**: Quantitatively assess chatbot response quality.

**Key File**: `evaluation/evaluate_responses.py`

**Metrics**:
- **ROUGE-1, ROUGE-2, ROUGE-L**: Text overlap metrics
- **Semantic Relevance**: Cosine similarity using sentence-transformers
- **Coherence**: Text quality assessment
- **Guardrails Scores**: Profanity, topic relevance, politeness

**Critical Code**:
```python
def calculate_rouge_scores(self, reference, candidate):
    scores = self.rouge_scorer.score(reference, candidate)
    return {
        'rouge_1': scores['rouge1'].fmeasure,
        'rouge_2': scores['rouge2'].fmeasure,
        'rouge_l': scores['rougeL'].fmeasure
    }

def calculate_relevance_score(self, query, response):
    query_embedding = self.sentence_model.encode(query)
    response_embedding = self.sentence_model.encode(response)
    similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
    return similarity
```

### 7. **Monitoring & Metrics**

**Purpose**: Real-time system and chatbot performance monitoring.

**Key Files**:  
- `monitoring/postgres_monitor.py`
- `monitoring/simple_postgres_dashboard.json` (Grafana)

**What We Monitor**:
- **Response Performance**: Response times, throughput, success rates
- **Model Performance**: Token usage, model latency, accuracy
- **System Health**: LightRAG, Neo4j, Ollama status
- **User Experience**: Active sessions, query patterns, satisfaction
- **Business Metrics**: Patent coverage, query volume, popular topics

**Critical Code**:
```python
def record_lightrag_metric(self, operation, duration_ms, success, 
                          documents_retrieved=0, error_message=None):
    cursor.execute("""
        INSERT INTO performance_metrics 
        (component, operation, duration_ms, success, error_message, additional_data)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, ('lightrag', operation, duration_ms, success, error_message,
           json.dumps({'documents_retrieved': documents_retrieved})))
```

### 8. **LLM Configuration**

- **Model**: `qwen2.5:14b-instruct` (14 billion parameters)
- **Provider**: Ollama (local inference)
- **Host**: `http://localhost:11434`
- **Embeddings**: `bge-m3:latest` (1024 dimensions)
- **Max Tokens**: 32,768
- **Temperature**: 0.0 (deterministic responses)

### 9. **Neo4j Fallback**

**When used**: If LightRAG is down, times out, or returns errors

**How it works**: Direct Cypher queries for document and entity retrieval

**Critical Code**:
```python
def generate_fallback_response(self, query):
    # Search documents by content
    cypher_query = """
    MATCH (d:document)
    WHERE d.description CONTAINS $search_term 
       OR d.file_path CONTAINS $search_term
    RETURN d.entity_id as id, d.file_path as title, d.description as content
    LIMIT 10
    """
    result = session.run(cypher_query, search_term=query)
```

### 10. **PostgreSQL for Grafana**

- **Database**: `patent_monitoring`
- **Tables**: 
  - `chat_metrics` - Chatbot interactions
  - `performance_metrics` - Component performance
  - `system_metrics` - System health data
- **Integration**: Grafana dashboard for real-time visualization
- **Refresh Rate**: 5 seconds

## 🚀 Pipeline Flow Summary

1. **Filter**: Extract and optimize G06 patents (`filtering/`)
2. **Integrate**: Upload to LightRAG (`lightrag_integration/`)
3. **Backup**: Monitor and backup to SQLite (`backup/`)
4. **Graph**: Store entity relationships in Neo4j
5. **Chatbot**: Serve queries via RAG+LLM, fallback to Neo4j if needed (`chatbot/`)
6. **Guardrails**: Validate every response for safety and quality
7. **Evaluate**: Score responses with ROUGE, relevance, coherence
8. **Monitor**: Record all metrics to PostgreSQL, visualize in Grafana

## 📊 Example: End-to-End Usage

```bash
# 1. Filter patents
python main.py --mode filter --input-dir hupd_extracted

# 2. Integrate with LightRAG
python main.py --mode integrate

# 3. Start chatbot (with guardrails)
python main.py --mode chat

# 4. Monitor in Grafana
open http://localhost:3000
```

## 🎯 Conclusion

This pipeline is a robust, production-grade system for patent analysis, combining advanced filtering, RAG, LLM, entity graphs, multi-layer backup, guardrails, and real-time monitoring. All components are modular and can be extended or replaced as needed.

**Key Strengths**:
- ✅ **Modular Design**: Each component can be used independently
- ✅ **Robust Backup**: Multi-layer storage with real-time monitoring
- ✅ **Quality Assurance**: Guardrails and evaluation metrics
- ✅ **Production Ready**: Comprehensive monitoring and error handling
- ✅ **Scalable**: Can handle large patent datasets efficiently 