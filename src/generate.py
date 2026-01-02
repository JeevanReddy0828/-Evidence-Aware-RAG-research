"""
Answer Generation Module

Generates answers with inline citations using local or API-based LLMs.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .ingest import Chunk


@dataclass
class GenerationConfig:
    """Configuration for answer generation."""
    provider: str = "local"  # local, openai, anthropic
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    device: str = "auto"
    
    # Prompt settings
    system_prompt: str = """You are a helpful assistant that answers questions based on provided context.
Always cite your sources using [1], [2], etc. referring to the passage numbers.
If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
Be concise and accurate."""
    
    citation_style: str = "inline"  # inline, footnote, none


@dataclass  
class GenerationResult:
    """Container for generation results."""
    answer: str
    citations: List[int]  # Which passages were cited
    raw_output: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Generator:
    """LLM-based answer generator with citation support."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Determine device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device
    
    def _load_local_model(self):
        """Load local HuggingFace model."""
        if self.pipe is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers required for local generation")
            
            print(f"Loading model: {self.config.model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True
            )
            
            # Use pipeline for easier generation
            self.pipe = pipeline(
                "text-generation",
                model=self.config.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
    
    def _format_context(self, chunks: List[Chunk]) -> str:
        """Format retrieved chunks as numbered context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] {chunk.text}")
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        chunks: List[Chunk],
        chat_format: bool = True
    ) -> str:
        """Build the generation prompt."""
        context = self._format_context(chunks)
        
        user_message = f"""Context:
{context}

Question: {query}

Answer the question based on the context above. Cite your sources using [1], [2], etc."""
        
        if chat_format:
            # Format for instruction-tuned models
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Use tokenizer's chat template if available
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # Fallback format
            return f"""<s>[INST] {self.config.system_prompt}

{user_message} [/INST]"""
        
        return f"{self.config.system_prompt}\n\n{user_message}\n\nAnswer:"
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from generated text."""
        # Find all [N] patterns
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return sorted(set(int(m) for m in matches))
    
    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        max_new_tokens: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate an answer with citations.
        
        Args:
            query: User question
            chunks: Retrieved context chunks
            max_new_tokens: Override default max tokens
        
        Returns:
            GenerationResult with answer and citations
        """
        if self.config.provider == "local":
            return self._generate_local(query, chunks, max_new_tokens)
        elif self.config.provider == "openai":
            return self._generate_openai(query, chunks, max_new_tokens)
        elif self.config.provider == "anthropic":
            return self._generate_anthropic(query, chunks, max_new_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _generate_local(
        self,
        query: str,
        chunks: List[Chunk],
        max_new_tokens: Optional[int] = None
    ) -> GenerationResult:
        """Generate using local model."""
        self._load_local_model()
        
        prompt = self._build_prompt(query, chunks, chat_format=True)
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=self.config.temperature if self.config.do_sample else None,
            top_p=self.config.top_p if self.config.do_sample else None,
            do_sample=self.config.do_sample,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        raw_output = outputs[0]['generated_text']
        answer = raw_output.strip()
        citations = self._extract_citations(answer)
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            raw_output=raw_output,
            prompt=prompt,
            metadata={"provider": "local", "model": self.config.model}
        )
    
    def _generate_openai(
        self,
        query: str,
        chunks: List[Chunk],
        max_new_tokens: Optional[int] = None
    ) -> GenerationResult:
        """Generate using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for OpenAI generation")
        
        context = self._format_context(chunks)
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        user_message = f"""Context:
{context}

Question: {query}

Answer the question based on the context above. Cite your sources using [1], [2], etc."""
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature
        )
        
        answer = response.choices[0].message.content.strip()
        citations = self._extract_citations(answer)
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            raw_output=answer,
            prompt=str(messages),
            metadata={
                "provider": "openai",
                "model": self.config.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }
        )
    
    def _generate_anthropic(
        self,
        query: str,
        chunks: List[Chunk],
        max_new_tokens: Optional[int] = None
    ) -> GenerationResult:
        """Generate using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required for Anthropic generation")
        
        context = self._format_context(chunks)
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        user_message = f"""Context:
{context}

Question: {query}

Answer the question based on the context above. Cite your sources using [1], [2], etc."""
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        answer = response.content[0].text.strip()
        citations = self._extract_citations(answer)
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            raw_output=answer,
            prompt=user_message,
            metadata={
                "provider": "anthropic",
                "model": self.config.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        )


class MockGenerator:
    """Mock generator for testing (returns context summary)."""
    
    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        max_new_tokens: Optional[int] = None
    ) -> GenerationResult:
        """Generate a mock answer by extracting from context."""
        # Simple extractive approach for testing
        if not chunks:
            answer = "I don't have enough information to answer this question."
            citations = []
        else:
            # Use first chunk as answer
            answer = f"Based on the provided context, {chunks[0].text[:200]}... [1]"
            citations = [1]
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            raw_output=answer,
            prompt=query,
            metadata={"provider": "mock"}
        )


def main():
    """CLI for testing generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test answer generation")
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--context", "-c", nargs="+", required=True)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    
    args = parser.parse_args()
    
    # Create dummy chunks
    chunks = [
        Chunk(
            chunk_id=f"test_{i}",
            text=text,
            doc_id="test",
            doc_title="Test",
            chunk_index=i,
            start_char=0,
            end_char=len(text)
        )
        for i, text in enumerate(args.context)
    ]
    
    config = GenerationConfig(provider=args.provider, model=args.model)
    generator = Generator(config)
    
    result = generator.generate(args.query, chunks)
    
    print(f"\nQuery: {args.query}")
    print("=" * 60)
    print(f"\nAnswer: {result.answer}")
    print(f"\nCitations used: {result.citations}")


if __name__ == "__main__":
    main()
