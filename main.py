"""
Main application entry point for the Manzil Chatbot.
"""

import os
import argparse
from pathlib import Path

# Import configuration and setup logging first
from config import settings
from utils.logging_config import setup_logging, get_logger

# Initialize logging
logger = setup_logging()
logger = get_logger("main")

# Import components
from crawler.crawler import WebCrawler
from knowledge_base.chunking import TextChunker
from knowledge_base.vector_store import VectorStore
from knowledge_base.knowledge_graph import KnowledgeGraph
from nlp.embeddings import EmbeddingGenerator
from nlp.query_processor import QueryProcessor
from nlp.prompt_engineering import PromptEngineering
from rag.retriever import Retriever
from rag.generator import ResponseGenerator
from ui.gradio_app import create_ui

class ManzilChatbot:
    """
    Main chatbot application class that integrates all components.
    """
    
    def __init__(self):
        """Initialize the chatbot with all necessary components."""
        # Check critical settings
        self._validate_settings()
        
        # Initialize components
        logger.info("Initializing Manzil Chatbot components")
        
        self.crawler = WebCrawler()
        self.chunker = TextChunker()
        self.vector_store = VectorStore()
        self.embedding_generator = EmbeddingGenerator()
        self.query_processor = QueryProcessor()
        self.prompt_engineering = PromptEngineering()
        self.retriever = Retriever(self.vector_store)
        self.generator = ResponseGenerator()
        
        # Initialize knowledge graph if enabled
        if settings.ENABLE_KNOWLEDGE_GRAPH:
            self.knowledge_graph = KnowledgeGraph()
        else:
            self.knowledge_graph = None
        
        logger.info("Manzil Chatbot initialized successfully")
    
    def _validate_settings(self):
        """Validate critical settings before startup."""
        issues = settings.validate_settings()
        
        if issues:
            for issue in issues:
                logger.error(f"Configuration issue: {issue}")
            
            if any(issue.startswith("OPENAI_API_KEY") for issue in issues):
                raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
        
        logger.info("Configuration validated successfully")
    
    def initialize_knowledge_base(self, force_refresh=False):
        """
        Initialize or refresh the knowledge base.
        
        Args:
            force_refresh (bool): Whether to force a refresh even if data exists
        
        Returns:
            bool: Whether the initialization was successful
        """
        try:
            # Check if we need to refresh
            if not force_refresh:
                collection_info = self.vector_store.get_collection_info()
                if collection_info.get("count", 0) > 0:
                    logger.info(f"Using existing knowledge base with {collection_info['count']} documents")
                    return True
            
            # Clear existing data if refreshing
            if force_refresh:
                logger.info("Forcing refresh of knowledge base")
                self.vector_store.clear_collection()
            
            # Crawl the website
            logger.info(f"Crawling website: {settings.BASE_URL}")
            pages_content = self.crawler.crawl()
            
            if not pages_content:
                logger.error("No content extracted from website")
                return False
            
            logger.info(f"Extracted content from {len(pages_content)} pages")
            
            # Process and chunk content
            all_chunks = []
            for page in pages_content:
                # Use header-based chunking for better semantic division
                chunks = self.chunker.chunk_by_headers(page)
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from extracted content")
            
            # Store chunks in vector database
            self.vector_store.store_documents(all_chunks)
            
            logger.info(f"Knowledge base initialized successfully with {len(all_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            return False
    
    def process_query(self, query, conversation_history=None):
        """
        Process a user query and generate a response.
        
        Args:
            query (str): The user's question
            conversation_history (list, optional): Previous conversation turns
            
        Returns:
            str: Generated response
        """
        try:
            # Process the query to identify type and enhance it
            query_type, enhanced_query = self.query_processor.process_query(query)
            
            logger.info(f"Query type: {query_type}, Enhanced query: {enhanced_query}")
            
            # Retrieve relevant contexts
            retrieved_documents = self.retriever.retrieve(enhanced_query)
            
            if not retrieved_documents:
                logger.warning("No relevant documents found for query")
                prompt = self.prompt_engineering.create_no_context_prompt(query)
            else:
                # Format context
                context = self.prompt_engineering.format_context(retrieved_documents)
                
                # Create RAG prompt
                prompt = self.prompt_engineering.create_rag_prompt(
                    query_type, 
                    query, 
                    context, 
                    conversation_history
                )
            
            # Generate response
            logger.debug(f"Generating response with prompt length: {len(prompt)}")
            response = self.generator.generate(prompt)
            
            # Update knowledge graph if enabled
            if self.knowledge_graph:
                entities = self.query_processor.extract_entities(query, response)
                if entities:
                    self.knowledge_graph.update_from_query(query, response, entities)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your question. Please try again or contact {settings.COMPANY_NAME} directly for assistance."
    
    def run_web_interface(self):
        """Launch the web interface."""
        ui = create_ui(self)
        ui.launch(
            server_name=settings.SERVER_NAME,
            server_port=settings.SERVER_PORT,
            share=settings.GRADIO_SHARE
        )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Manzil Help Center Chatbot')
    parser.add_argument('--refresh', action='store_true', help='Force refresh of the knowledge base')
    parser.add_argument('--initialize-only', action='store_true', help='Only initialize the knowledge base, then exit')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize the chatbot
        chatbot = ManzilChatbot()
        
        # Initialize the knowledge base
        success = chatbot.initialize_knowledge_base(force_refresh=args.refresh)
        
        if not success:
            logger.error("Failed to initialize knowledge base. Exiting.")
            return 1
        
        # Exit if only initializing
        if args.initialize_only:
            logger.info("Knowledge base initialized. Exiting as requested.")
            return 0
        
        # Run the web interface
        logger.info("Starting web interface")
        chatbot.run_web_interface()
        
        return 0
        
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)