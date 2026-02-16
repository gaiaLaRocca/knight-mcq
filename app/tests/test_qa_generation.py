#!/usr/bin/env python

import os
import sys
import logging
import time
from app.core.common.neo4j_connection import Neo4jConnection
from app.generation.qa_generation import generate_qa_from_graph
from app.core.common.config import (
    GPT_NEO4J_URI as NEO4J_URI,
    GPT_NEO4J_USER as NEO4J_USER,
    GPT_NEO4J_PASSWORD as NEO4J_PASSWORD,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("qa_generation_test")

class SimpleLLMClient:
    """Simple wrapper for LangChain LLM to match the expected interface"""
    
    def __init__(self, model_name=OPENAI_MODEL, api_key=OPENAI_API_KEY):
        kwargs = dict(model=model_name, api_key=api_key, max_tokens=2000, temperature=0.1)
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        self.llm = ChatOpenAI(**kwargs)
        logger.info(f"Initialized LLM client with model: {model_name}")
    
    def generate(self, prompt):
        """Generate text using the LLM with a prompt"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None

def main():
    """Main function to run the QA generation test"""
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test QA generation")
    parser.add_argument("--limit", type=int, default=3, help="Limit per type")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    parser.add_argument("--validation-rate", type=float, default=1.0, help="Validation sample rate (0.0-1.0)")
    args = parser.parse_args()
    
    logger.info(f"Starting QA generation test with limit={args.limit}, "
                f"skip_validation={args.skip_validation}, "
                f"validation_rate={args.validation_rate}")
    
    # Check if Neo4j credentials are available
    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error("Neo4j credentials missing. Make sure your .env file contains GPT_NEO4J_* or REBEL_NEO4J_* variables.")
        sys.exit(1)
    
    # Initialize the LLM client
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY missing. Make sure your .env file contains OPENAI_API_KEY.")
        sys.exit(1)
        
    llm_client = SimpleLLMClient()
    
    # Initialize the Neo4j connection
    start_time = time.time()
    try:
        with Neo4jConnection(uri=NEO4J_URI, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as conn:
            # Run the QA generation with the specified parameters
            qa_pairs = generate_qa_from_graph(
                neo4j_conn=conn,
                llm_client=llm_client,
                limit_per_type=args.limit,
                skip_validation=args.skip_validation,
                validation_sample_rate=args.validation_rate
            )
            
            # Print the results
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            total_time = time.time() - start_time
            logger.info(f"Total execution time: {total_time:.2f}s")
            
            # Display some example pairs
            print("\nExample generated QA pairs:")
            for i, pair in enumerate(qa_pairs[:5]):  # Show up to 5 examples
                print(f"\nPair {i+1}:")
                print(f"Q: {pair.get('question')}")
                print(f"A: {pair.get('answer')}")
                
    except Exception as e:
        logger.error(f"Error in QA generation test: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 