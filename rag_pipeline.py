import os
import fitz
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import logging
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleRetriever(BaseRetriever):
    """Ensemble retriever that combines multiple retrievers."""
    
    retrievers: List[BaseRetriever]
    weights: List[float]
    
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        # Calculate weights if not provided
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)
        
        # Call parent __init__ with the fields as keyword arguments
        super().__init__(retrievers=retrievers, weights=weights)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from all retrievers."""
        all_docs = []
        doc_scores = {}
        
        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                try:
                    docs = retriever.get_relevant_documents(query)
                except AttributeError:
                    logger.warning(f"Retriever {type(retriever)} has no compatible method")
                    continue
            
            for i, doc in enumerate(docs):
                doc_id = doc.page_content
                score = weight * (1.0 / (i + 1))
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += score
                else:
                    doc_scores[doc_id] = {'doc': doc, 'score': score}
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
    
    def invoke(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

class RelevanceChecker:
    """
    RelevanceChecker: rerank, threshold, optional contextual compression.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        threshold: float = 0.60,
        min_docs: int = 2,
        max_docs: int = 6,
        enable_compression: bool = True,
        compression_top_sentences: int = 3,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        fallback_to_cosine: bool = True
    ):
        self.embedding_model = embedding_model
        self.cross_encoder = None
        self.cross_encoder_name = cross_encoder_name

        if cross_encoder_name:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_name)
                logger.info(f"Loaded CrossEncoder: {cross_encoder_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder '{cross_encoder_name}': {e}")
                self.cross_encoder = None

        self.threshold = threshold
        self.min_docs = min_docs
        self.max_docs = max_docs
        self.enable_compression = enable_compression
        self.compression_top_sentences = compression_top_sentences
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.fallback_to_cosine = fallback_to_cosine

    def filter_documents(
        self,
        question: str,
        docs: List,
        doc_embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[object, float]]:

        logger.info(f"Filtering {len(docs)} retrieved chunks for question: {question}")

        if not docs:
            logger.warning("No documents retrieved.")
            return []

        if self.cross_encoder is not None:
            scored = self._score_with_crossencoder(question, docs)
        else:
            scored = self._score_with_cosine(question, docs, doc_embeddings)

        filtered = [(d, s) for d, s in scored if s >= self.threshold]

        if len(filtered) < self.min_docs:
            filtered = scored[: self.min_docs]

        filtered = filtered[: self.max_docs]

        if self.enable_compression:
            compressed = []
            for doc, score in filtered:
                compressed_doc = self._compress_document(question, doc, top_k=self.compression_top_sentences)
                compressed.append((compressed_doc, score))
            filtered = compressed

        return filtered

    def _score_with_crossencoder(self, question: str, docs: List) -> List[Tuple[object, float]]:
        cross_input = [(question, doc.page_content) for doc in docs]
        try:
            scores = self.cross_encoder.predict(cross_input, batch_size=self.batch_size)
        except Exception as e:
            logger.warning(f"Cross-encoder failed: {e}, falling back to cosine.")
            return self._score_with_cosine(question, docs)

        scores = self._minmax_normalize(np.array(scores))
        return sorted(list(zip(docs, scores.tolist())), key=lambda x: x[1], reverse=True)


    def _score_with_cosine(self, question: str, docs: List, doc_embeddings: Optional[np.ndarray] = None):
        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)

        if doc_embeddings is None:
            texts = [d.page_content for d in docs]
            doc_embeddings = self.embedding_model.encode(texts, batch_size=self.batch_size,
                                                         show_progress_bar=False, convert_to_numpy=True)
        if self.normalize_embeddings:
            doc_embeddings = self._l2_normalize(doc_embeddings)

        sims = np.dot(doc_embeddings, q_emb.T).reshape(-1)
        sims = (sims + 1.0) / 2.0

        return sorted(list(zip(docs, sims.tolist())), key=lambda x: x[1], reverse=True)

    def _compress_document(self, question: str, doc, top_k: int = 3):
        text = doc.page_content
        sentences = self._split_sentences(text)

        if not sentences:
            return doc

        q_emb = self.embedding_model.encode([question], show_progress_bar=False, convert_to_numpy=True)
        sent_embs = self.embedding_model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)

        if self.normalize_embeddings:
            q_emb = self._l2_normalize(q_emb)
            sent_embs = self._l2_normalize(sent_embs)

        sims = np.dot(sent_embs, q_emb.T).reshape(-1)
        idx_scores = list(enumerate(sims))
        idx_scores_sorted = sorted(idx_scores, key=lambda x: x[1], reverse=True)

        top_idx = [i for i, _ in idx_scores_sorted[:top_k]]
        top_idx.sort()

        compressed_text = " ".join([sentences[i] for i in top_idx]).strip()

        compressed_doc = type(doc)(
            page_content=compressed_text,
            metadata={**getattr(doc, "metadata", {})}
        )
        return compressed_doc

    @staticmethod
    def _minmax_normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)

    @staticmethod
    def _l2_normalize(x):
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0] = 1e-12
        return x / denom

    @staticmethod
    def _split_sentences(text: str):
        parts = re.split(r'(?<=[\.\?\!])\s+', text)
        return [p.strip() for p in parts if p.strip()]

class PDFExtractor:
    """Handles PDF extraction with layout preservation"""
    
    def __init__(self):
        self.header_footer_margin = 50
        self.min_text_length = 50
        self.heading_font_threshold = 13
    
    def extract_pdf(self, pdf_path: str) -> List[Dict]:
        try:
            content_blocks = self._extract_with_layout(pdf_path)
            if not content_blocks:
                content_blocks = self._fallback_extraction(pdf_path)
            content_blocks = self._merge_blocks(content_blocks)
            return content_blocks
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return []
    
    def _extract_with_layout(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        all_content = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                bbox = block["bbox"]
                if (bbox[1] < self.header_footer_margin or 
                    bbox[3] > page_height - self.header_footer_margin):
                    continue
                text_lines = []
                font_sizes = []
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                    if line_text.strip():
                        text_lines.append(line_text.strip())
                if not text_lines:
                    continue
                text = " ".join(text_lines)
                if len(text.strip()) < self.min_text_length:
                    continue
                avg_font_size = np.mean(font_sizes) if font_sizes else 11
                content_type = "heading" if avg_font_size > self.heading_font_threshold else "paragraph"
                all_content.append({
                    "text": text,
                    "page": page_num + 1,
                    "type": content_type,
                    "bbox": bbox
                })
            tables = self._extract_tables(page, page_num + 1)
            all_content.extend(tables)
        doc.close()
        return all_content
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
        tables = []
        try:
            tabs = page.find_tables()
            for i, table in enumerate(tabs):
                df = table.to_pandas()
                if df.empty:
                    continue
                table_text = f"Table {i+1}:\n{df.to_string(index=False)}"
                tables.append({
                    "text": table_text,
                    "page": page_num,
                    "type": "table",
                    "bbox": table.bbox
                })
        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")
        return tables
    
    def _fallback_extraction(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        content = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "page": page_num + 1,
                    "type": "text"
                })
        doc.close()
        return content
    
    def _merge_blocks(self, content_blocks: List[Dict]) -> List[Dict]:
        if not content_blocks:
            return []
        merged = []
        current_block = None
        for block in content_blocks:
            if block["type"] == "table":
                if current_block:
                    merged.append(current_block)
                    current_block = None
                merged.append(block)
                continue
            if current_block is None:
                current_block = block.copy()
                continue
            if (block["page"] == current_block["page"] and 
                len(current_block["text"]) + len(block["text"]) < 800):
                current_block["text"] += " " + block["text"]
            else:
                merged.append(current_block)
                current_block = block.copy()
        if current_block:
            merged.append(current_block)
        return merged
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\"\'\%\$\#\@\!\?\/\&\+\=\*]', '', text)
        return text.strip()

class SemanticChunker:
    def __init__(self, embedding_model: SentenceTransformer, base_chunk_size: int = 512, base_overlap: int = 50, sim_threshold: float = 0.85):
        self.embedding_model = embedding_model
        self.base_splitter = CharacterTextSplitter(chunk_size=base_chunk_size, chunk_overlap=base_overlap, separator="\n\n")
        self.sim_threshold = sim_threshold

    def split_text(self, text: str) -> list[str]:
        mini_chunks = self.base_splitter.split_text(text)
        if len(mini_chunks) <= 1:
            return mini_chunks
        embs = self.embedding_model.encode(mini_chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        merged_chunks: list[str] = []
        current = mini_chunks[0]
        current_emb = embs[0]
        for i in range(1, len(mini_chunks)):
            sim = float(np.dot(current_emb, embs[i]))
            if sim >= self.sim_threshold:
                current = current + " " + mini_chunks[i]
                current_emb = (current_emb + embs[i]) / 2.0
            else:
                merged_chunks.append(current)
                current = mini_chunks[i]
                current_emb = embs[i]
        merged_chunks.append(current)
        return merged_chunks

class RAGPipeline:
    def __init__(self, pdf_folder: str, index_file: str, model_params: dict):
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.embedding_model.eval()
        self.relevance_checker = RelevanceChecker(embedding_model=self.embedding_model)
        self.pdf_extractor = PDFExtractor()
        self.text_splitter = SemanticChunker(embedding_model=self.embedding_model)
        
        self.documents = []
        self.embeddings = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

        api_key = model_params.get("google_api_key")
        if api_key:
            genai.configure(api_key=api_key)

    def expand_query(self, question: str) -> List[str]:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"Generate 3 different search queries to help answer this question: {question}. Return only the queries, one per line."
            response = model.generate_content(prompt)
            queries = [q.strip() for q in response.text.split('\n') if q.strip()]
            if question not in queries:
                queries.insert(0, question)
            return queries
        except Exception:
            return [question]

    def _reciprocal_rank_fusion(self, results_list: List[List[Document]], k=60):
        fused_scores = {}
        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", ""))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"score": 0.0, "doc": doc}
                fused_scores[doc_id]["score"] += 1.0 / (rank + k)
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]

    def build_index(self):
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        all_documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            content_blocks = self.pdf_extractor.extract_pdf(pdf_path)
            documents = self._create_chunks(content_blocks, pdf_file)
            all_documents.extend(documents)
        
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._build_retrievers(all_documents, texts, embeddings)
        self.documents = all_documents
        self.embeddings = embeddings
        self._save_index()
        return len(all_documents)

    def _create_chunks(self, content_blocks, pdf_name):
        documents = []
        for block in content_blocks:
            text = self.pdf_extractor.clean_text(block["text"])
            if len(text) < 50: continue
            if block["type"] == "table" or len(text) < 600:
                documents.append(Document(page_content=text, metadata={"source": pdf_name, "page": block["page"], "type": block["type"]}))
            else:
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"source": pdf_name, "page": block["page"], "type": block["type"]}))
        return documents

    def _build_retrievers(self, documents, texts, embeddings):
        faiss_index = FAISS.from_embeddings(text_embeddings=list(zip(texts, embeddings)), embedding=lambda x: self.embedding_model.encode(x, normalize_embeddings=True), metadatas=[doc.metadata for doc in documents])
        self.faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 20})
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20
        self.hybrid_retriever = EnsembleRetriever(retrievers=[self.faiss_retriever, self.bm25_retriever], weights=[0.85, 0.15])

    def _save_index(self):
        with open(self.index_file, "wb") as f:
            pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)

    def load_index(self):
        if not os.path.exists(self.index_file): return False
        with open(self.index_file, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        texts = [doc.page_content for doc in self.documents]
        self._build_retrievers(self.documents, texts, self.embeddings)
        return True

    def query(self, question: str, top_k: int = 8):
        queries = self.expand_query(question)
        all_results = [self.hybrid_retriever.invoke(q) for q in queries]
        fused_docs = self._reciprocal_rank_fusion(all_results)
        filtered = self.relevance_checker.filter_documents(question, fused_docs[:20])
        return [d for d, s in filtered[:top_k]]

if __name__ == "__main__":
    print("RAG Pipeline script updated to Gemini and advanced retrieval.")
