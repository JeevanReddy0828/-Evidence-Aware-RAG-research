"""
Interactive Demo Application

Provides a Gradio-based interface for testing the Evidence-Aware RAG system.
"""

import os
import json
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Warning: Gradio not available. Install with: pip install gradio")

from src.pipeline import RAGPipeline, PipelineConfig
from src.ingest import DocumentIngestor, Chunk
from src.verify import VerificationDecision


# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


def load_demo_documents():
    """Load sample documents for demo."""
    demo_docs = [
        {
            "title": "Machine Learning Basics",
            "text": """Machine learning is a subset of artificial intelligence that enables computers 
            to learn and improve from experience without being explicitly programmed. The three main 
            types of machine learning are supervised learning, unsupervised learning, and reinforcement 
            learning. Supervised learning uses labeled data to train models, while unsupervised learning 
            finds patterns in unlabeled data. Reinforcement learning trains agents through rewards and 
            penalties. Deep learning, a subset of machine learning, uses neural networks with multiple 
            layers to learn complex patterns. Popular deep learning frameworks include TensorFlow, PyTorch, 
            and JAX."""
        },
        {
            "title": "Natural Language Processing",
            "text": """Natural Language Processing (NLP) is a field of AI focused on enabling computers 
            to understand, interpret, and generate human language. Key NLP tasks include text classification, 
            named entity recognition, sentiment analysis, machine translation, and question answering. 
            Modern NLP heavily relies on transformer architectures, introduced in the 'Attention is All 
            You Need' paper in 2017. Large language models like GPT, BERT, and LLaMA have revolutionized 
            NLP by achieving state-of-the-art results across many tasks. These models are pre-trained on 
            large text corpora and can be fine-tuned for specific applications."""
        },
        {
            "title": "Retrieval-Augmented Generation",
            "text": """Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based 
            and generation-based approaches for question answering and text generation. RAG systems first 
            retrieve relevant documents from a knowledge base, then use a language model to generate 
            answers based on the retrieved context. This approach helps reduce hallucinations by grounding 
            the model's outputs in factual information. Key components of RAG include the retriever 
            (often using dense embeddings), the knowledge base, and the generator (typically a large 
            language model). RAG is particularly useful for knowledge-intensive tasks where the model 
            needs access to up-to-date or domain-specific information."""
        },
        {
            "title": "Vector Databases",
            "text": """Vector databases are specialized databases designed to store and query high-dimensional 
            vectors efficiently. They are essential for modern AI applications including semantic search, 
            recommendation systems, and RAG pipelines. Popular vector databases include Pinecone, Weaviate, 
            Milvus, Qdrant, and Chroma. These databases use approximate nearest neighbor (ANN) algorithms 
            like HNSW, IVF, and PQ to enable fast similarity search. Vector databases typically support 
            both dense vectors (from neural embeddings) and sparse vectors (from TF-IDF or BM25). 
            Integration with embedding models from OpenAI, Cohere, or open-source alternatives is common."""
        },
        {
            "title": "The Apollo 11 Mission",
            "text": """Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil 
            Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 
            1969, at 20:17 UTC. Armstrong became the first person to step onto the lunar surface six hours 
            and 39 minutes later on July 21. Aldrin joined him 19 minutes later. They spent about two and 
            a quarter hours together outside the spacecraft. Michael Collins piloted the command module 
            Columbia alone in lunar orbit while they were on the Moon's surface. Armstrong and Aldrin 
            collected 47.5 pounds of lunar material to bring back to Earth."""
        }
    ]
    return demo_docs


def initialize_pipeline(documents: list = None):
    """Initialize the RAG pipeline with documents."""
    global pipeline
    
    if documents is None:
        documents = load_demo_documents()
    
    # Create chunks from documents
    ingestor = DocumentIngestor(chunk_size=400, chunk_overlap=50)
    chunks = []
    
    for i, doc in enumerate(documents):
        text_chunks = ingestor.splitter.split_text(doc["text"])
        for j, text in enumerate(text_chunks):
            chunks.append(Chunk(
                chunk_id=f"doc_{i}_chunk_{j}",
                text=text,
                doc_id=f"doc_{i}",
                doc_title=doc["title"],
                chunk_index=j,
                start_char=0,
                end_char=len(text)
            ))
    
    # Build pipeline with mock generator (to avoid loading heavy models)
    from src.generate import MockGenerator
    
    config = PipelineConfig()
    config.verification_enabled = True
    config.rerank_enabled = False  # Disable for demo speed
    
    pipeline = RAGPipeline.from_chunks(chunks, config)
    pipeline._generator = MockGenerator()  # Use mock for demo
    
    return f"✅ Pipeline initialized with {len(chunks)} chunks from {len(documents)} documents"


def query_pipeline(question: str, verify: bool = True):
    """Process a question through the pipeline."""
    global pipeline
    
    if pipeline is None:
        return "❌ Pipeline not initialized. Click 'Initialize' first.", "", ""
    
    if not question.strip():
        return "Please enter a question.", "", ""
    
    try:
        result = pipeline.query(question, skip_verification=not verify)
        
        # Format answer
        answer = result.answer
        
        # Format verification info
        if result.verification_result:
            vr = result.verification_result
            decision_emoji = {
                VerificationDecision.GROUNDED: "✅",
                VerificationDecision.PARTIALLY_GROUNDED: "⚠️",
                VerificationDecision.ABSTAIN: "🛑",
                VerificationDecision.CLARIFY: "❓"
            }
            
            verification_info = f"""
**Decision**: {decision_emoji.get(vr.decision, "")} {vr.decision.value.upper()}
**Groundedness Score**: {vr.groundedness_score:.2%}
**Claims Verified**: {len(vr.claim_verifications)}
**Claims Supported**: {sum(1 for cv in vr.claim_verifications if cv.is_supported)}
            """
            
            # Claim details
            claim_details = ""
            for i, cv in enumerate(vr.claim_verifications, 1):
                status = "✓" if cv.is_supported else "✗"
                claim_details += f"\n{i}. [{status}] \"{cv.claim.text[:60]}...\" (score: {cv.entailment_score:.2f})"
        else:
            verification_info = "Verification disabled"
            claim_details = ""
        
        # Format sources
        sources = ""
        for i, (chunk, score) in enumerate(result.retrieval_result, 1):
            sources += f"\n**[{i}] {chunk.doc_title}** (score: {score:.3f})\n{chunk.text[:200]}...\n"
        
        # Add latency info
        latency_info = f"\n\n**Latency**: {result.total_latency_ms:.0f}ms"
        
        return answer, verification_info + claim_details + latency_info, sources
        
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def create_demo_interface():
    """Create the Gradio interface."""
    if not GRADIO_AVAILABLE:
        print("Gradio not available. Running in CLI mode.")
        return None
    
    with gr.Blocks(title="Evidence-Aware RAG Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Evidence-Aware RAG Demo
        
        This demo showcases a RAG system with **groundedness verification** that:
        - Retrieves relevant passages using hybrid search
        - Generates answers with citations
        - Verifies claims against evidence using NLI
        - Abstains when evidence is insufficient
        
        *Note: This demo uses mock generation. Full system requires GPU for LLM inference.*
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                init_btn = gr.Button("🚀 Initialize Pipeline", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### Ask a Question")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g., What is RAG and how does it reduce hallucinations?",
                    lines=2
                )
                
                with gr.Row():
                    verify_checkbox = gr.Checkbox(label="Enable Verification", value=True)
                    submit_btn = gr.Button("🔎 Search & Answer", variant="primary")
                
                gr.Markdown("### Answer")
                answer_output = gr.Textbox(label="Generated Answer", lines=4, interactive=False)
                
            with gr.Column(scale=1):
                gr.Markdown("### Verification Details")
                verification_output = gr.Markdown()
                
        gr.Markdown("### Retrieved Sources")
        sources_output = gr.Markdown()
        
        gr.Markdown("""
        ---
        ### Example Questions
        - What is machine learning and what are its main types?
        - Who was the first person to walk on the moon?
        - What are vector databases used for?
        - What is the capital of Mars? *(should abstain - out of scope)*
        """)
        
        # Event handlers
        init_btn.click(
            fn=initialize_pipeline,
            inputs=[],
            outputs=[init_status]
        )
        
        submit_btn.click(
            fn=query_pipeline,
            inputs=[question_input, verify_checkbox],
            outputs=[answer_output, verification_output, sources_output]
        )
        
        question_input.submit(
            fn=query_pipeline,
            inputs=[question_input, verify_checkbox],
            outputs=[answer_output, verification_output, sources_output]
        )
    
    return demo


def run_cli_demo():
    """Run a simple CLI demo."""
    print("\n" + "="*60)
    print("Evidence-Aware RAG - CLI Demo")
    print("="*60)
    
    print("\nInitializing pipeline...")
    status = initialize_pipeline()
    print(status)
    
    print("\nExample questions:")
    print("1. What is machine learning?")
    print("2. Who walked on the moon first?")
    print("3. What is RAG?")
    print("4. What is the population of Jupiter? (should abstain)")
    print("\nType 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer, verification, sources = query_pipeline(question, verify=True)
        
        print(f"\n{'='*40}")
        print(f"Answer: {answer}")
        print(f"\nVerification: {verification}")
        print(f"\nSources: {sources[:500]}...")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evidence-Aware RAG Demo")
    parser.add_argument("--cli", action="store_true", help="Run CLI demo instead of Gradio")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    
    args = parser.parse_args()
    
    if args.cli or not GRADIO_AVAILABLE:
        run_cli_demo()
    else:
        demo = create_demo_interface()
        demo.launch(server_port=args.port, share=args.share)
