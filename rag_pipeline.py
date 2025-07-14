import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from typing import List, Optional, Dict, Any
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedBotRAG:
    def __init__(self):
        """Initialize the RAG pipeline components"""
        self.rag_chain = None
        self.embeddings = None
        self.retriever = None
        self.chat_model = None
        self.index_name = "medicalbot"
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
    def initialize_embeddings(self):
        """Initialize the embedding model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return self

    def initialize_vectorstore(self):
        """Initialize or connect to Pinecone vector store"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        return self

    def initialize_retriever(self):
        """Configure the document retriever"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 50,
                "score_threshold": 0.7,
                "filter": {"status": "verified"}  # Optional content filtering
            }
        )
        return self

    def initialize_llm(self):
        """Initialize the language model"""
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,
            max_output_tokens=2000,
            top_k=40,
            top_p=0.95
        )
        return self

    def create_prompt_template(self):
        """Create the advanced prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are NthanziLanga+ AI assistant, created by TecNix to help with health-related questions and information.
            You are designed to provide detailed helpful, accurate health guidance and answers using the context.

            Guidelines:
            1. Answer comprehensively without recommending doctors unless:
               - Explicitly asked about referrals
               - The condition is life-threatening
               - Context indicates need for professional care
            2. Structure responses:
               - Direct Answer (concise)
               - Supporting Evidence (bulleted)
               - Self-Care Options (when applicable)
            3. Cite sources when available
            
            Context: {context}
            """),
            ("human", "{input}")
        ])

    def build_chains(self):
        """Construct the processing chains"""
        prompt = self.create_prompt_template()
        
        document_chain = create_stuff_documents_chain(
            llm=self.chat_model,
            prompt=prompt
        )
        
        self.rag_chain = create_retrieval_chain(
            self.retriever,
            document_chain
        )
        return self

    def initialize(self):
        """Complete pipeline initialization with validation"""
        try:
            (self.initialize_embeddings()
                .initialize_vectorstore()
                .initialize_retriever()
                .initialize_llm()
                .build_chains())
            
            # Validate all components
            assert all([
                self.embeddings,
                self.vectorstore,
                self.retriever,
                self.chat_model,
                self.rag_chain
            ]), "Component initialization failed"
            
            logger.info("RAG pipeline fully initialized")
            return self
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"RAG pipeline initialization error: {str(e)}")

    def refresh_documents(self, document_path: Optional[str] = None):
        """Refresh documents in the vector store"""
        from langchain.document_loaders import (
            DirectoryLoader,
            PyPDFLoader,
            TextLoader,
            WebBaseLoader
        )
        
        try:
            if document_path:
                if document_path.startswith('http'):
                    loader = WebBaseLoader(document_path)
                elif document_path.endswith('.pdf'):
                    loader = PyPDFLoader(document_path)
                else:
                    loader = TextLoader(document_path)
                documents = loader.load()
            else:
                # Reload all existing documents
                documents = self.vectorstore.similarity_search("", k=10000)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True
            )
            
            processed_docs = text_splitter.split_documents(documents)
            
            # Clear and refresh index
            self.pc.delete_index(self.index_name)
            self.initialize_vectorstore()
            
            PineconeVectorStore.from_documents(
                documents=processed_docs,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            logger.info(f"Processed {len(processed_docs)} document chunks")
            return {"status": "success", "processed": len(processed_docs)}
            
        except Exception as e:
            logger.error(f"Document refresh failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def query(self, question: str, medical_context: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced query handling with medical context"""
        try:
            if not self.rag_chain:
                logger.warning("RAG chain not initialized - auto-initializing")
                self.initialize()
            
            # Prepare input
            augmented_input = (
                f"Medical Context: {medical_context}\n\nQuestion: {question}"
                if medical_context else question
            )
            
            # Execute query
            result = self.rag_chain.invoke({"input": augmented_input})
            
            # Process results
            answer = result.get("answer", "")
            if not answer or "don't know" in answer.lower():
                answer = self._fallback_search(question)
            
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("context", [])
                ],
                "confidence": self._calculate_confidence(result.get("context", []))
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return {
                "error": "Service unavailable",
                "fallback_answer": self._emergency_response(question),
                "details": str(e)
            }

    def _fallback_search(self, question: str) -> str:
        """Secondary search strategy when primary fails"""
        try:
            docs = self.vectorstore.max_marginal_relevance_search(
                question, k=10, lambda_mult=0.8
            )
            return "\n".join([doc.page_content for doc in docs][:3])
        except Exception:
            return "I couldn't find enough information to answer this question."

    def _calculate_confidence(self, documents: List) -> float:
        """Calculate response confidence score"""
        if not documents:
            return 0.0
        return min(1.0, len(documents) / 10)  # Scale based on document count

    def _emergency_response(self, question: str) -> str:
        """Fallback when system fails"""
        medical_keywords = ["heart attack", "stroke", "bleeding", "unconscious"]
        if any(kw in question.lower() for kw in medical_keywords):
            return ("This appears to be a medical emergency. "
                   "Please call emergency services immediately.")
        return ("I'm experiencing technical difficulties. "
               "For reliable medical information, please consult a healthcare provider.")

# Singleton instance
medbot_rag = MedBotRAG().initialize()