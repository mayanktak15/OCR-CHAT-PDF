"""
RAG (Retrieval-Augmented Generation) utilities.
Uses local ML models with hybrid search - no API keys required.
"""

import logging
import re
from typing import Any
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Local embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking configuration - smaller chunks for better precision
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Model singleton
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Get or load the sentence transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Model loaded successfully")
    return _model


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    # First split into sentences
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk size, save current chunk
        if current_length + sentence_len > CHUNK_SIZE and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > CHUNK_OVERLAP:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            
            current_chunk = overlap_sentences
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sentence_len

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Also add paragraph-level chunks for longer context
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
    for para in paragraphs:
        if para not in chunks and len(para) < 1500:
            chunks.append(para)

    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks


def extract_keywords(text: str) -> set[str]:
    """Extract important keywords from text."""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'this', 'that', 'these', 'those', 'what',
        'which', 'who', 'whom', 'it', 'its', 'i', 'me', 'my', 'we', 'our',
        'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = {w for w in words if w not in stop_words}
    return keywords


def keyword_score(query: str, chunk: str) -> float:
    """Calculate keyword overlap score between query and chunk."""
    query_keywords = extract_keywords(query)
    chunk_keywords = extract_keywords(chunk)
    
    if not query_keywords:
        return 0.0
    
    # Count matching keywords
    matches = query_keywords & chunk_keywords
    
    # Also check for exact phrase matches
    query_lower = query.lower()
    chunk_lower = chunk.lower()
    
    phrase_bonus = 0.0
    query_words = query_lower.split()
    for i in range(len(query_words)):
        for j in range(i + 2, min(i + 5, len(query_words) + 1)):
            phrase = " ".join(query_words[i:j])
            if len(phrase) > 5 and phrase in chunk_lower:
                phrase_bonus += 0.1 * (j - i)
    
    base_score = len(matches) / len(query_keywords) if query_keywords else 0
    return min(1.0, base_score + phrase_bonus)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Create embeddings for text chunks using local model."""
    if not chunks:
        return np.array([])

    model = get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    return embeddings.astype(np.float32)


def build_index(text: str) -> dict[str, Any]:
    """
    Build a search index from text.
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks could be created")

    embeddings = embed_chunks(chunks)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "full_text": text,
    }


def search_index(index: dict[str, Any], query: str, top_k: int = 5) -> list[tuple[float, str]]:
    """
    Hybrid search: combines semantic similarity with keyword matching.
    """
    chunks = index["chunks"]
    embeddings = index["embeddings"]

    if len(chunks) == 0:
        return []

    model = get_model()
    
    # Semantic similarity
    query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    semantic_scores = embeddings @ query_vec

    # Keyword matching scores
    keyword_scores = np.array([keyword_score(query, chunk) for chunk in chunks])

    # Hybrid score: weighted combination
    # Semantic: 60%, Keyword: 40%
    hybrid_scores = 0.6 * semantic_scores + 0.4 * keyword_scores

    # Get top_k indices
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    return [(float(hybrid_scores[i]), chunks[i]) for i in top_indices]


def highlight_relevant_sentences(chunk: str, query: str) -> str:
    """Highlight the most relevant sentences within a chunk."""
    sentences = split_into_sentences(chunk)
    if len(sentences) <= 3:
        return chunk
    
    query_keywords = extract_keywords(query)
    
    # Score each sentence
    scored = []
    for sent in sentences:
        sent_keywords = extract_keywords(sent)
        overlap = len(query_keywords & sent_keywords)
        scored.append((overlap, sent))
    
    # Sort by relevance and take top sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Take most relevant sentences but maintain some context
    top_sentences = [s for _, s in scored[:4]]
    
    # Reorder by appearance in original text
    result = []
    for sent in sentences:
        if sent in top_sentences:
            result.append(sent)
    
    return " ".join(result) if result else chunk


def query_index(index: dict[str, Any], question: str) -> str:
    """
    Answer a question by finding and presenting the most relevant text passages.
    """
    results = search_index(index, question)

    if not results:
        return "No relevant information found in the document."

    response_parts = []
    seen_content = set()  # Avoid duplicate content
    
    for i, (score, chunk) in enumerate(results, 1):
        # Only include reasonably relevant results
        if score < 0.15:
            continue
        
        # Skip near-duplicate content
        chunk_key = chunk[:100].lower()
        if chunk_key in seen_content:
            continue
        seen_content.add(chunk_key)
        
        # Highlight most relevant parts
        highlighted = highlight_relevant_sentences(chunk, question)
        
        # Clean up and truncate if needed
        highlighted = highlighted.strip()
        if len(highlighted) > 600:
            highlighted = highlighted[:600] + "..."
        
        relevance_label = "High" if score > 0.5 else "Medium" if score > 0.3 else "Low"
        response_parts.append(f"📄 **Match {i}** ({relevance_label} relevance):\n{highlighted}")
        
        # Limit to 4 results
        if len(response_parts) >= 4:
            break

    if not response_parts:
        # Fallback: show best match even if low score
        if results:
            best_score, best_chunk = results[0]
            best_chunk = best_chunk[:500] + "..." if len(best_chunk) > 500 else best_chunk
            return f"Best match found (low confidence):\n\n{best_chunk}"
        return "Could not find relevant content. Try rephrasing your question or using different keywords."

    return "\n\n---\n\n".join(response_parts)
