"""
AmbedkarGPT Evaluation Framework
Assignment 2 - Comprehensive RAG Evaluation System

This script implements comprehensive evaluation metrics for the RAG system:
- Retrieval Metrics: Hit Rate, MRR, Precision@K
- Answer Quality Metrics: Answer Relevance, Faithfulness, ROUGE-L
- Semantic Metrics: Cosine Similarity, BLEU Score


"""

import os

# Disable ALL telemetry across LangChain, Chroma, LangSmith, OpenAI compatibility
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LC_TELEMETRY"] = "false"
os.environ["LANGCHAIN_DISABLE_TELEMETRY"] = "true"
os.environ["LANGCHAIN_HUB_API_KEY"] = ""
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_ENDPOINT"] = ""
os.environ["LANGSMITH_TRACING"] = "false"

# Chroma telemetry disable (already done, but keep it)
os.environ["CHROMADB_DISABLE_TELEMETRY"] = "1"
os.environ["CHROMADB_DISABLE_EMBEDDING"] = "1"

import json
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from pathlib import Path

# LangChain imports
# NOTE: use langchain_community.TextLoader and manual directory iteration for Windows compatibility
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Evaluation metrics imports
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive RAG evaluation framework with multiple metrics
    """

    def __init__(self, corpus_path: str = "corpus", test_dataset_path: str = "test_dataset.json"):
        """
        Initialize the RAG Evaluator

        Args:
            corpus_path: Path to document corpus folder
            test_dataset_path: Path to test dataset JSON file
        """
        self.corpus_path = corpus_path
        self.test_dataset_path = test_dataset_path
        self.test_questions = self._load_test_dataset()

        # Initialize embeddings model (same as Assignment 1)
        logger.info("Initializing HuggingFace embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize LLM (Ollama with Mistral)
        logger.info("Initializing Ollama LLM with Mistral 7B...")
        self.llm = OllamaLLM(model="mistral", temperature=0.1)

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "chunking_strategies": {}
        }

    def _load_test_dataset(self) -> List[Dict]:
        """Load test dataset from JSON file"""
        logger.info(f"Loading test dataset from {self.test_dataset_path}")
        with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_questions']

    def _load_documents(self) -> List[Any]:
        """
        Load all documents from corpus folder (Windows-safe)
        Uses TextLoader on each .txt file in the corpus_path and returns a flat list of langchain Documents.
        """
        logger.info(f"Loading documents from {self.corpus_path} (Windows-safe loader)")
        docs = []

        if not os.path.isdir(self.corpus_path):
            logger.warning(f"Corpus path '{self.corpus_path}' does not exist or is not a directory.")
            return docs

        # Iterate files in corpus directory (non-recursive). If you want recursive support,
        # replace os.listdir with os.walk.
        for fname in sorted(os.listdir(self.corpus_path)):
            fpath = os.path.join(self.corpus_path, fname)
            if not os.path.isfile(fpath):
                continue
            # Accept .txt files (you can add more extensions if needed)
            if fname.lower().endswith(".txt"):
                try:
                    loader = TextLoader(fpath, encoding='utf-8')
                    file_docs = loader.load()
                    docs.extend(file_docs)
                except Exception as e:
                    logger.error(f"Failed to load {fpath}: {e}")
                    continue
        logger.info(f"Loaded {len(docs)} documents from corpus")
        return docs

    def _create_vector_store(self, documents: List[Any], chunk_size: int, 
                            chunk_overlap: int, strategy_name: str) -> Chroma:
        """
        Create vector store with specific chunking strategy

        Args:
            documents: List of loaded documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            strategy_name: Name of chunking strategy for persistence

        Returns:
            Chroma vector store
        """
        logger.info(f"Creating vector store with chunk_size={chunk_size}, overlap={chunk_overlap}")

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Split documents
        texts = text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} chunks")

        # Create vector store with unique persistence directory
        persist_directory = f"./chroma_db_{strategy_name}"

        # Remove existing directory if exists
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)

        vectorstore = Chroma.from_documents(
         documents=texts,
        embedding=self.embeddings,
        persist_directory=persist_directory,
        client_settings={
        "chroma_db_impl": "duckdb+parquet",
        "persist_directory": persist_directory
    }
)


        vectorstore.persist()
        logger.info(f"Vector store created and persisted to {persist_directory}")

        return vectorstore

    def _create_qa_chain(self, vectorstore: Chroma, k: int = 4) -> RetrievalQA:
        """
        Create RetrievalQA chain

        Args:
            vectorstore: Chroma vector store
            k: Number of documents to retrieve

        Returns:
            RetrievalQA chain
        """
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer or if the information is not in the context, say "I don't have enough information to answer this question based on the provided documents."

Context: {context}

Question: {question}

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return qa_chain

    # ==================== RETRIEVAL METRICS ====================

    def calculate_hit_rate(self, retrieved_docs: List[str], 
                          relevant_docs: List[str]) -> float:
        """
        Calculate Hit Rate: Whether any relevant document was retrieved

        Args:
            retrieved_docs: List of retrieved document names
            relevant_docs: List of ground truth relevant document names

        Returns:
            Hit rate (0 or 1)
        """
        if not relevant_docs:  # For unanswerable questions
            return 1.0 if not retrieved_docs else 0.0

        for doc in retrieved_docs:
            if any(rel_doc in doc for rel_doc in relevant_docs):
                return 1.0
        return 0.0

    def calculate_mrr(self, retrieved_docs: List[str], 
                     relevant_docs: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank

        Args:
            retrieved_docs: Ordered list of retrieved document names
            relevant_docs: List of ground truth relevant document names

        Returns:
            MRR score
        """
        if not relevant_docs:  # For unanswerable questions
            return 0.0

        for idx, doc in enumerate(retrieved_docs, 1):
            if any(rel_doc in doc for rel_doc in relevant_docs):
                return 1.0 / idx
        return 0.0

    def calculate_precision_at_k(self, retrieved_docs: List[str], 
                                 relevant_docs: List[str], k: int = 4) -> float:
        """
        Calculate Precision@K

        Args:
            retrieved_docs: List of retrieved document names
            relevant_docs: List of ground truth relevant document names
            k: Number of top documents to consider

        Returns:
            Precision@K score
        """
        if not relevant_docs:  # For unanswerable questions
            return 0.0

        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in retrieved_k 
            if any(rel_doc in doc for rel_doc in relevant_docs)
        )

        return relevant_retrieved / k if k > 0 else 0.0

    # ==================== ANSWER QUALITY METRICS ====================

    def calculate_answer_relevance(self, answer: str, question: str) -> float:
        """
        Calculate Answer Relevance using TF-IDF cosine similarity

        Args:
            answer: Generated answer
            question: User question

        Returns:
            Relevance score (0-1)
        """
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question, answer])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def calculate_faithfulness(self, answer: str, context_docs: List[str]) -> float:
        """
        Calculate Faithfulness: How much of answer content appears in context

        Args:
            answer: Generated answer
            context_docs: Retrieved context documents

        Returns:
            Faithfulness score (0-1)
        """
        if not context_docs or not answer:
            return 0.0

        # Combine all context
        full_context = " ".join(context_docs).lower()
        answer_lower = answer.lower()

        # Split answer into sentences/phrases
        answer_parts = [s.strip() for s in answer_lower.split('.') if s.strip()]

        if not answer_parts:
            return 0.0

        # Check how many parts appear in context
        faithful_parts = 0
        for part in answer_parts:
            # Check for significant word overlap
            words = [w for w in part.split() if len(w) > 3]  # Skip short words
            if not words:
                continue

            overlap = sum(1 for w in words if w in full_context)
            if overlap / len(words) > 0.5:  # At least 50% word overlap
                faithful_parts += 1

        return faithful_parts / len(answer_parts)

    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        Calculate ROUGE-L score

        Args:
            generated: Generated answer
            reference: Ground truth answer

        Returns:
            ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure

    # ==================== SEMANTIC METRICS ====================

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using embeddings

        Args:
            text1: First text (generated answer)
            text2: Second text (ground truth)

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Get embeddings
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                np.array(emb1).reshape(1, -1),
                np.array(emb2).reshape(1, -1)
            )[0][0]

            return float(similarity)
        except:
            return 0.0

    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU score

        Args:
            generated: Generated answer
            reference: Ground truth answer

        Returns:
            BLEU score
        """
        try:
            # Tokenize
            reference_tokens = [reference.lower().split()]
            generated_tokens = generated.lower().split()

            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction()
            score = sentence_bleu(
                reference_tokens,
                generated_tokens,
                smoothing_function=smoothing.method1
            )

            return float(score)
        except:
            return 0.0

    # ==================== EVALUATION PIPELINE ====================

    def evaluate_single_question(self, qa_chain: RetrievalQA, 
                                question_data: Dict) -> Dict:
        """
        Evaluate a single question with all metrics

        Args:
            qa_chain: RetrievalQA chain
            question_data: Test question data

        Returns:
            Dictionary with all evaluation metrics
        """
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        relevant_docs = question_data['source_documents']

        # Get answer from QA chain
        try:
            result = qa_chain({"query": question})
            generated_answer = result['result']
            source_docs = result['source_documents']

            # Extract document names
            retrieved_doc_names = [
                os.path.basename(doc.metadata.get('source', '')) 
                for doc in source_docs
            ]

            # Extract document contents
            retrieved_doc_contents = [doc.page_content for doc in source_docs]

        except Exception as e:
            logger.error(f"Error processing question {question_data['id']}: {e}")
            generated_answer = ""
            retrieved_doc_names = []
            retrieved_doc_contents = []

        # Calculate all metrics
        metrics = {
            "question_id": question_data['id'],
            "question": question,
            "question_type": question_data['question_type'],
            "answerable": question_data['answerable'],
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "retrieved_documents": retrieved_doc_names,
            "relevant_documents": relevant_docs,

            # Retrieval Metrics
            "hit_rate": self.calculate_hit_rate(retrieved_doc_names, relevant_docs),
            "mrr": self.calculate_mrr(retrieved_doc_names, relevant_docs),
            "precision_at_k": self.calculate_precision_at_k(retrieved_doc_names, relevant_docs),

            # Answer Quality Metrics
            "answer_relevance": self.calculate_answer_relevance(generated_answer, question),
            "faithfulness": self.calculate_faithfulness(generated_answer, retrieved_doc_contents),
            "rouge_l": self.calculate_rouge_l(generated_answer, ground_truth),

            # Semantic Metrics
            "cosine_similarity": self.calculate_cosine_similarity(generated_answer, ground_truth),
            "bleu_score": self.calculate_bleu_score(generated_answer, ground_truth)
        }

        return metrics

    def evaluate_chunking_strategy(self, strategy_name: str, 
                                   chunk_size: int, 
                                   chunk_overlap: int) -> Dict:
        """
        Evaluate complete RAG system with specific chunking strategy

        Args:
            strategy_name: Name of the strategy
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            Complete evaluation results for this strategy
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating strategy: {strategy_name}")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        logger.info(f"{'='*60}")

        # Load documents
        documents = self._load_documents()

        # Create vector store
        vectorstore = self._create_vector_store(
            documents, chunk_size, chunk_overlap, strategy_name
        )

        # Create QA chain
        qa_chain = self._create_qa_chain(vectorstore)

        # Evaluate all questions
        question_results = []
        for idx, question_data in enumerate(self.test_questions, 1):
            logger.info(f"Evaluating question {idx}/{len(self.test_questions)}")
            metrics = self.evaluate_single_question(qa_chain, question_data)
            question_results.append(metrics)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(question_results)

        strategy_results = {
            "strategy_name": strategy_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "aggregate_metrics": aggregate_metrics,
            "question_results": question_results
        }

        return strategy_results

    def _calculate_aggregate_metrics(self, question_results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all questions"""

        # Filter answerable questions for most metrics
        answerable_results = [r for r in question_results if r['answerable']]

        if not answerable_results:
            return {}

        aggregate = {
            "total_questions": len(question_results),
            "answerable_questions": len(answerable_results),
            "unanswerable_questions": len(question_results) - len(answerable_results),

            # Retrieval Metrics (on answerable questions)
            "avg_hit_rate": np.mean([r['hit_rate'] for r in answerable_results]),
            "avg_mrr": np.mean([r['mrr'] for r in answerable_results]),
            "avg_precision_at_k": np.mean([r['precision_at_k'] for r in answerable_results]),

            # Answer Quality Metrics
            "avg_answer_relevance": np.mean([r['answer_relevance'] for r in answerable_results]),
            "avg_faithfulness": np.mean([r['faithfulness'] for r in answerable_results]),
            "avg_rouge_l": np.mean([r['rouge_l'] for r in answerable_results]),

            # Semantic Metrics
            "avg_cosine_similarity": np.mean([r['cosine_similarity'] for r in answerable_results]),
            "avg_bleu_score": np.mean([r['bleu_score'] for r in answerable_results]),

            # By question type
            "metrics_by_type": self._metrics_by_type(question_results)
        }

        return aggregate

    def _metrics_by_type(self, question_results: List[Dict]) -> Dict:
        """Calculate metrics grouped by question type"""
        types = {}

        for result in question_results:
            q_type = result['question_type']
            if q_type not in types:
                types[q_type] = []
            types[q_type].append(result)

        metrics_by_type = {}
        for q_type, results in types.items():
            if not results:
                continue

            metrics_by_type[q_type] = {
                "count": len(results),
                "avg_hit_rate": np.mean([r['hit_rate'] for r in results]),
                "avg_rouge_l": np.mean([r['rouge_l'] for r in results]),
                "avg_cosine_similarity": np.mean([r['cosine_similarity'] for r in results])
            }

        return metrics_by_type

    def run_comparative_analysis(self) -> Dict:
        """
        Run evaluation on all three chunking strategies

        Returns:
            Complete results for all strategies
        """
        logger.info("Starting comparative chunking analysis...")

        # Define chunking strategies as per assignment
        strategies = [
            ("small_chunks", 250, 50),   # 200-300 characters
            ("medium_chunks", 550, 100),  # 500-600 characters
            ("large_chunks", 900, 150)    # 800-1000 characters
        ]

        for strategy_name, chunk_size, chunk_overlap in strategies:
            strategy_results = self.evaluate_chunking_strategy(
                strategy_name, chunk_size, chunk_overlap
            )
            self.results["chunking_strategies"][strategy_name] = strategy_results

        return self.results

    def save_results(self, output_file: str = "test_results.json"):
        """Save evaluation results to JSON file"""
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved successfully")

    def print_summary(self):
        """Print summary of evaluation results"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        for strategy_name, strategy_data in self.results["chunking_strategies"].items():
            agg = strategy_data["aggregate_metrics"]

            print(f"\n{strategy_name.upper().replace('_', ' ')}")
            print(f"Chunk Size: {strategy_data['chunk_size']}, Overlap: {strategy_data['chunk_overlap']}")
            print("-" * 60)

            print("\nRetrieval Metrics:")
            print(f"  Hit Rate:        {agg['avg_hit_rate']:.3f}")
            print(f"  MRR:             {agg['avg_mrr']:.3f}")
            print(f"  Precision@K:     {agg['avg_precision_at_k']:.3f}")

            print("\nAnswer Quality Metrics:")
            print(f"  Answer Relevance: {agg['avg_answer_relevance']:.3f}")
            print(f"  Faithfulness:     {agg['avg_faithfulness']:.3f}")
            print(f"  ROUGE-L:          {agg['avg_rouge_l']:.3f}")

            print("\nSemantic Metrics:")
            print(f"  Cosine Similarity: {agg['avg_cosine_similarity']:.3f}")
            print(f"  BLEU Score:        {agg['avg_bleu_score']:.3f}")

        print("\n" + "="*80)


def main():
    """Main execution function"""
    print("="*80)
    print("AmbedkarGPT - Comprehensive RAG Evaluation Framework")
    print("Assignment 2 - Kalpit Pvt Ltd, UK")
    print("="*80)

    # Initialize evaluator
    evaluator = RAGEvaluator(
        corpus_path="corpus",
        test_dataset_path="test_dataset.json"
    )

    # Run comparative analysis
    results = evaluator.run_comparative_analysis()

    # Save results
    evaluator.save_results("test_results.json")

    # Print summary
    evaluator.print_summary()

    logger.info("\nEvaluation complete! Check test_results.json for detailed results.")
    logger.info("Run analysis on results to generate results_analysis.md")


if __name__ == "__main__":
    main()
