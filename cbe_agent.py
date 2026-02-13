import os
import fitz
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import logging
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple, Optional, Dict
import numpy as np
import heapq
import math

class RelevanceChecker:
    """
    RelevanceChecker: rerank, threshold, optional contextual compression.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.70,
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

        for i, (doc, score) in enumerate(scored):
            logger.info(f"[Score] Rank={i+1} Score={score:.4f} Content={doc.page_content[:150]}...")

        filtered = [(d, s) for d, s in scored if s >= self.threshold]
        logger.info(f"Chunks above threshold {self.threshold}: {len(filtered)}")

        if len(filtered) < self.min_docs:
            logger.info(f"Below min_docs={self.min_docs}, selecting top {self.min_docs} anyway.")
            filtered = scored[: self.min_docs]

        filtered = filtered[: self.max_docs]

        for doc, score in filtered:
            logger.info(f"[Selected] Score={score:.4f} Content={doc.page_content[:150]}...")

        if self.enable_compression:
            compressed = []
            for doc, score in filtered:
                logger.info(f"Compressing chunk (score={score:.4f}): {doc.page_content[:150]}...")
                compressed_doc = self._compress_document(question, doc, top_k=self.compression_top_sentences)
                logger.info(f"[Compressed Result] {compressed_doc.page_content[:200]}...")
                compressed.append((compressed_doc, score))
            filtered = compressed

        return filtered

    def _score_with_crossencoder(self, question: str, docs: List) -> List[Tuple[object, float]]:
        cross_input = [(question, doc.page_content) for doc in docs]
        try:
            scores = self.cross_encoder.predict(cross_input, batch_size=self.batch_size)
            logger.info("Cross-encoder scoring successful.")
        except Exception as e:
            logger.warning(f"Cross-encoder failed: {e}, falling back to cosine.")
            return self._score_with_cosine(question, docs)

        scores = self._minmax_normalize(np.array(scores))
        return sorted(list(zip(docs, scores.tolist())), key=lambda x: x[1], reverse=True)

    def _score_with_cosine(self, question: str, docs: List, doc_embeddings: Optional[np.ndarray] = None):
        logger.info("Using cosine similarity scoring...")

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

        logger.info(f"Splitting into {len(sentences)} sentences for compression.")

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
        for i, (idx, score) in enumerate(idx_scores_sorted[:top_k]):
            logger.info(f"[Compression Sentence #{i+1}] Score={score:.4f} Sentence={sentences[idx][:200]}")

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
                logger.warning(f"Layout extraction failed, using fallback for {pdf_path}")
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
        text = re.sub(r'\b\d{1,3}\s*$', '', text)
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\"\'\%\$\#\@\!\?\/\&\+\=\*]', '', text)
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'([(])\s+', r'\1', text)
        return text.strip()


class SemanticChunker:
    """
    Very simple semantic-ish chunker:
    1) First split into small fixed chunks.
    2) Then merge adjacent chunks whose embeddings are very similar.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        base_chunk_size: int = 512,
        base_overlap: int = 50,
        sim_threshold: float = 0.85,
    ):
        from langchain_text_splitters import CharacterTextSplitter

        self.embedding_model = embedding_model
        self.base_splitter = CharacterTextSplitter(
            chunk_size=base_chunk_size,
            chunk_overlap=base_overlap,
            separator="\n\n",
        )
        self.sim_threshold = sim_threshold

    def split_text(self, text: str) -> list[str]:
        mini_chunks = self.base_splitter.split_text(text)
        if len(mini_chunks) <= 1:
            return mini_chunks

        embs = self.embedding_model.encode(
            mini_chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )

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
    """Main RAG pipeline for PDF question answering - Tool-based version"""
    def __init__(
        self,
        pdf_folder: str,
        index_file: str,
        model_params: dict,
    ):
        # Paths
        self.pdf_folder = pdf_folder
        self.index_file = index_file
        
        # Store model params for later use
        self.model_params = model_params
        
        # Models
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.embedding_model.eval()
        
        self.relevance_checker = RelevanceChecker(
            embedding_model=self.embedding_model,
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            threshold=0.65,
            min_docs=2,
            max_docs=6,
            enable_compression=False,
            compression_top_sentences=3
        )
        
        self.pdf_extractor = PDFExtractor()
        
        self.text_splitter = SemanticChunker(
            embedding_model=self.embedding_model,
            base_chunk_size=512,
            base_overlap=50,
            sim_threshold=0.85,
        )
        
        # Storage
        self.documents = []
        self.embeddings = None
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
    
    def build_index(self, progress_callback=None, status_callback=None):
        """Build index from PDFs in folder"""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError("No PDF files found in folder")
        
        all_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            if status_callback:
                status_callback(f"Processing: {pdf_file} ({i+1}/{len(pdf_files)})")
            
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            
            try:
                content_blocks = self.pdf_extractor.extract_pdf(pdf_path)
                documents = self._create_chunks(content_blocks, pdf_file)
                all_documents.extend(documents)
                
                logger.info(f"Extracted {len(documents)} chunks from {pdf_file}")
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue
            
            if progress_callback:
                progress_callback((i + 1) / len(pdf_files))
        
        if not all_documents:
            raise ValueError("No content extracted from PDFs")
        
        if status_callback:
            status_callback(f"Encoding {len(all_documents)} chunks...")
        
        texts = [doc.page_content for doc in all_documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        self._build_retrievers(all_documents, texts, embeddings)
        
        self.documents = all_documents
        self.embeddings = embeddings
        self._save_index()
        
        if status_callback:
            status_callback(f"✅ Indexed {len(all_documents)} chunks from {len(pdf_files)} PDFs")
        
        return len(all_documents)
    
    def _create_chunks(self, content_blocks: List[Dict], pdf_name: str) -> List[Document]:
        documents = []
        
        for block in content_blocks:
            text = self.pdf_extractor.clean_text(block["text"])
            
            if len(text) < 50:
                continue
            
            if block["type"] == "table" or len(text) < 600:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_name,
                        "page": block["page"],
                        "type": block["type"]
                    }
                ))
            else:
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_name,
                            "page": block["page"],
                            "type": block["type"],
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))
        
        return documents
    
    def _build_retrievers(self, documents: List[Document], texts: List[str], embeddings: np.ndarray):
        faiss_index = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=lambda x: self.embedding_model.encode(x, normalize_embeddings=True),
            metadatas=[doc.metadata for doc in documents]
        )
        self.faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 20})
        
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20
        
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.faiss_retriever, self.bm25_retriever],
            weights=[0.85, 0.15]
        )
    
    def _save_index(self):
        """Save index to file(s)"""
        # Handle both string and list cases for index_file
        if isinstance(self.index_file, list):
            # Save to all files in the list
            for save_file in self.index_file:
                with open(save_file, "wb") as f:
                    pickle.dump({
                        "documents": self.documents,
                        "embeddings": self.embeddings,
                        "model": "sentence-transformers/all-mpnet-base-v2"
                    }, f)
                logger.info(f"Saved index to '{save_file}'")
        else:
            # Single file
            with open(self.index_file, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                    "model": "sentence-transformers/all-mpnet-base-v2"
                }, f)
    
    def load_index(self):
        """Load index from file(s)"""
        # Handle both string and list cases for index_file
        if isinstance(self.index_file, list):
            all_documents = []
            all_embeddings = []
            loaded_files = []
            
            for index_file in self.index_file:
                if os.path.exists(index_file):
                    try:
                        with open(index_file, "rb") as f:
                            data = pickle.load(f)
                        
                        # Collect documents and embeddings from this file
                        all_documents.extend(data["documents"])
                        all_embeddings.append(data["embeddings"])
                        loaded_files.append(index_file)
                        
                        logger.info(f"Loaded index from '{index_file}' with {len(data['documents'])} chunks")
                        
                    except Exception as e:
                        logger.error(f"Failed to load index file '{index_file}': {e}")
                        continue
            
            if not loaded_files:
                logger.error("Failed to load any of the index files")
                return False
            
            # Combine all embeddings
            self.documents = all_documents
            self.embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
            
            # Build retrievers with all combined data
            texts = [doc.page_content for doc in self.documents]
            self._build_retrievers(self.documents, texts, self.embeddings)
            
            logger.info(f"✅ Successfully merged {len(loaded_files)} index files. Total chunks: {len(self.documents)}")
            logger.info(f"Loaded files: {loaded_files}")
            return True
            
        else:
            # Original single file logic
            if not os.path.exists(self.index_file):
                return False
            
            try:
                with open(self.index_file, "rb") as f:
                    data = pickle.load(f)
                
                self.documents = data["documents"]
                self.embeddings = data["embeddings"]
                
                texts = [doc.page_content for doc in self.documents]
                self._build_retrievers(self.documents, texts, self.embeddings)
                
                logger.info(f"Loaded index with {len(self.documents)} chunks")
                return True
            
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                return False
    
    def retrieve_documents(self, question: str, top_k: int = 8) -> Dict:
        """
        Tool function: Retrieve relevant documents for a question.
        Returns a dictionary with documents and metadata.
        """
        if self.hybrid_retriever is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        try:
            # Retrieve documents
            retrieved_docs = self.hybrid_retriever.invoke(question)
            
            if not retrieved_docs:
                return {
                    "success": False,
                    "message": "No relevant documents found.",
                    "documents": [],
                    "count": 0
                }
            
            top_docs = retrieved_docs[:top_k]
            filtered = self.relevance_checker.filter_documents(question, top_docs)
            filtered_docs = [d for d, s in filtered]
            
            # Format documents for return
            formatted_docs = []
            for doc in filtered_docs:
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "type": doc.metadata.get("type", "text")
                })
            
            logger.info(f"Retrieved {len(formatted_docs)} relevant documents for question: {question[:100]}")
            
            return {
                "success": True,
                "message": f"Successfully retrieved {len(formatted_docs)} relevant documents.",
                "documents": formatted_docs,
                "count": len(formatted_docs),
                "question": question
            }
        
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return {
                "success": False,
                "message": f"Retrieval error: {str(e)}",
                "documents": [],
                "count": 0
            }
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.documents:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(self.documents),
            "content_types": {}
        }
        
        for doc in self.documents:
            doc_type = doc.metadata.get("type", "unknown")
            stats["content_types"][doc_type] = stats["content_types"].get(doc_type, 0) + 1
        
        return stats

    def debug_print_chunks_for_source(self, source_name: str, max_chunks: int = 20):
        """Print all (or first N) chunks for a given PDF source."""
        matched = [d for d in self.documents if d.metadata.get("source") == source_name]
        print(f"[DEBUG] Found {len(matched)} chunks for source='{source_name}'")
        for i, doc in enumerate(matched[:max_chunks], 1):
            print(f"\n--- Chunk {i} ---")
            print("metadata:", doc.metadata)
            preview = doc.page_content[:800].replace("\n", " ")
            print("text    :", preview, "...")