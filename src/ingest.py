"""
Document Ingestion Module

Handles document loading, chunking, and preprocessing for RAG.
Supports multiple file formats and intelligent chunking strategies.
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Iterator
import re

from tqdm import tqdm


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class Document:
    """Represents a source document."""
    doc_id: str
    title: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextSplitter:
    """Recursive text splitter for intelligent chunking."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " "]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive character splitting."""
        return self._split_recursive(text, self.separators)
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using progressively finer separators."""
        if not text:
            return []
        
        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        # Find the best separator
        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else []
        
        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split as last resort
            splits = list(text)
        
        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_splits = self._get_overlap_splits(current_chunk, separator)
                current_chunk = overlap_splits + [split]
                current_length = sum(len(s) + len(separator) for s in current_chunk)
            else:
                current_chunk.append(split)
                current_length += split_length
        
        # Handle remaining text
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif len(chunk_text) > 0 and next_separators:
                # Try finer splitting for small remaining chunks
                chunks.extend(self._split_recursive(chunk_text, next_separators))
        
        return chunks
    
    def _get_overlap_splits(self, splits: List[str], separator: str) -> List[str]:
        """Get splits that should be included for overlap."""
        if not splits or self.chunk_overlap == 0:
            return []
        
        overlap_text = ""
        overlap_splits = []
        
        for split in reversed(splits):
            if len(overlap_text) + len(split) + len(separator) <= self.chunk_overlap:
                overlap_splits.insert(0, split)
                overlap_text = separator.join(overlap_splits)
            else:
                break
        
        return overlap_splits


class DocumentIngestor:
    """Main class for document ingestion and processing."""
    
    SUPPORTED_FORMATS = {'.txt', '.md', '.json', '.jsonl'}
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None
    ):
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            separators=separators
        )
        self.metadata_fields = metadata_fields or []
    
    def process_file(self, filepath: str) -> List[Chunk]:
        """Process a single file into chunks."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
        
        # Load document(s)
        documents = self._load_file(filepath)
        
        # Process each document
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def process_directory(
        self, 
        dirpath: str, 
        recursive: bool = True
    ) -> List[Chunk]:
        """Process all documents in a directory."""
        dirpath = Path(dirpath)
        
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        
        # Find all files
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in dirpath.glob(pattern) 
            if f.is_file() and f.suffix in self.SUPPORTED_FORMATS
        ]
        
        all_chunks = []
        for filepath in tqdm(files, desc="Processing documents"):
            try:
                chunks = self.process_file(filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        return all_chunks
    
    def _load_file(self, filepath: Path) -> List[Document]:
        """Load document(s) from a file."""
        documents = []
        
        if filepath.suffix == '.jsonl':
            # JSONL format (e.g., NQ, HotpotQA)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    doc = self._json_to_document(data, filepath)
                    documents.append(doc)
        
        elif filepath.suffix == '.json':
            # Single JSON or JSON array
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    doc = self._json_to_document(item, filepath)
                    documents.append(doc)
            else:
                doc = self._json_to_document(data, filepath)
                documents.append(doc)
        
        else:
            # Plain text formats (.txt, .md)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc = Document(
                doc_id=self._generate_id(str(filepath)),
                title=filepath.stem,
                text=text,
                source=str(filepath),
                metadata={"format": filepath.suffix}
            )
            documents.append(doc)
        
        return documents
    
    def _json_to_document(self, data: Dict[str, Any], filepath: Path) -> Document:
        """Convert JSON data to Document."""
        # Handle different JSON schemas
        text = data.get('text') or data.get('context') or data.get('content', '')
        title = data.get('title') or data.get('name') or filepath.stem
        doc_id = data.get('id') or data.get('doc_id') or self._generate_id(text[:100])
        
        # Extract metadata
        metadata = {
            k: v for k, v in data.items() 
            if k in self.metadata_fields or k not in ['text', 'context', 'content', 'title', 'id']
        }
        metadata['format'] = 'json'
        
        return Document(
            doc_id=str(doc_id),
            title=title,
            text=text,
            source=str(filepath),
            metadata=metadata
        )
    
    def _chunk_document(self, doc: Document) -> List[Chunk]:
        """Split document into chunks."""
        texts = self.splitter.split_text(doc.text)
        
        chunks = []
        char_offset = 0
        
        for idx, text in enumerate(texts):
            # Find actual position in original document
            start = doc.text.find(text, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(text)
            char_offset = start
            
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_{idx}",
                text=text,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                chunk_index=idx,
                start_char=start,
                end_char=end,
                metadata=doc.metadata.copy()
            )
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def _generate_id(content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """Save chunks to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + '\n')
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
    
    @staticmethod
    def load_chunks(input_path: str) -> List[Chunk]:
        """Load chunks from JSONL file."""
        chunks = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                chunks.append(Chunk.from_dict(data))
        return chunks


def main():
    """CLI for document ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents for RAG")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--recursive", action="store_true", default=True)
    
    args = parser.parse_args()
    
    ingestor = DocumentIngestor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        chunks = ingestor.process_file(args.input)
    else:
        chunks = ingestor.process_directory(args.input, recursive=args.recursive)
    
    ingestor.save_chunks(chunks, args.output)
    print(f"Ingestion complete: {len(chunks)} chunks created")


if __name__ == "__main__":
    main()
