# PDF OCR + RAG Chat

A local PDF document search and chat application. Upload PDFs, extract text (with OCR support for scanned documents), and search through the content using natural language queries.

**No API keys required** - uses local ML models for semantic search.

## Features

- **PDF Text Extraction** - Extracts text from digital PDFs using PyPDF2
- **OCR Fallback** - Automatically uses Tesseract OCR for scanned/image-based PDFs
- **Semantic Search** - Uses sentence-transformers for intelligent document search
- **Hybrid Search** - Combines semantic similarity with keyword matching for better accuracy
- **Web Interface** - Clean, responsive UI for uploading and chatting
- **100% Local** - All processing happens on your machine, no data sent to external services

## Prerequisites

### System Dependencies

**For OCR support** (optional but recommended for scanned PDFs):

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr poppler-utils
```

**Arch Linux:**
```bash
sudo pacman -S tesseract poppler
```

**macOS:**
```bash
brew install tesseract poppler
```

### Python

- Python 3.10 or higher

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ocr
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > Note: First run will download the sentence-transformers model (~90MB)

## Usage

1. **Start the server:**
   ```bash
   source .venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. **Open your browser:**
   ```
   http://127.0.0.1:8000
   ```

3. **Upload a PDF** and start asking questions!

## Project Structure

```
ocr/
├── app/
│   ├── main.py          # FastAPI application & routes
│   ├── ocr_utils.py     # PDF text extraction & OCR
│   └── rag_utils.py     # Embedding, indexing & search
├── static/
│   └── index.html       # Web interface
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web interface |
| `/health` | GET | Health check |
| `/api/upload` | POST | Upload and index a PDF |
| `/api/chat` | POST | Query the indexed document |
| `/api/session/{id}` | DELETE | Delete a session |

### Upload PDF

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@document.pdf"
```

Response:
```json
{
  "session_id": "uuid",
  "filename": "document.pdf",
  "text_length": 12345,
  "message": "PDF processed successfully"
}
```

### Chat

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "uuid", "question": "What is this about?"}'
```

Response:
```json
{
  "answer": "📄 Match 1 (High relevance):\n..."
}
```

## How It Works

1. **Text Extraction**: When you upload a PDF, it first tries native text extraction. If that fails or returns minimal text, it falls back to OCR.

2. **Chunking**: The extracted text is split into overlapping chunks (500 chars) respecting sentence boundaries.

3. **Embedding**: Each chunk is embedded using the `all-MiniLM-L6-v2` sentence transformer model.

4. **Hybrid Search**: When you ask a question:
   - Computes semantic similarity (60% weight)
   - Computes keyword overlap score (40% weight)
   - Returns the most relevant passages

## Configuration

Edit `app/rag_utils.py` to adjust:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
CHUNK_SIZE = 500                       # Characters per chunk
CHUNK_OVERLAP = 100                    # Overlap between chunks
```

## Troubleshooting

### "No text extracted from PDF"
- Ensure the PDF contains actual text (not just images)
- For scanned PDFs, install Tesseract OCR

### Model download slow
- First run downloads ~90MB model
- Subsequent runs use cached model

### Search results not relevant
- Try rephrasing with different keywords
- Use specific terms from the document

## License

MIT
