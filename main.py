# ============================================================================
# MAIN_V4 - Advanced Financial AI Agent with Full Integration
# Entry point for V4 with all features
# ============================================================================

import asyncio
import sys
import logging
import torch
from typing import Optional, Dict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import (
    Config, TaskType, setup_environment, check_dependencies, validate_config
)
from llm_handler import LLMHandler
from orchestrator import FinancialAgentOrchestratorV4


async def run_single_query(agent: FinancialAgentOrchestratorV4,
                          query: str,
                          user_context: Dict = None):
    """Run a single query with V4 agent"""
    
    try:
        print(f"\n{'#'*80}")
        print(f"Query: {query}")
        print(f"{'#'*80}")
        
        response = await agent.process_query(query, user_context)
        
        print(f"\n{'='*80}")
        print("ANSWER:")
        print(f"{'='*80}")
        print(response.answer)
        
        print(f"\n{'='*80}")
        print("METADATA:")
        print(f"{'='*80}")
        print(f"Task Type: {response.task_type.value}")
        # print(f"Complexity: {response.metadata.get('complexity', 'Unknown')}")
        # print(f"Live Data Included: {response.metadata.get('live_data_included', False)}")
        # print(f"PDF Generated: {response.metadata.get('pdf_generated', False)}")
        # print(f"Email Sent: {response.metadata.get('email_sent', False)}")
        
        if response.sources:
            print(f"Sources: {response.sources[:3]}")
        
        print(f"\n{'='*80}\n")
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"\n❌ Error: {e}\n")
        return None


async def run_interactive_mode(agent: FinancialAgentOrchestratorV4):
    """Run agent in interactive mode"""
    
    print(f"\n{'#'*80}")
    print("INTERACTIVE MODE - V4 (Advanced Financial AI Agent)")
    print("Features: DragonLLM + RAG + Live APIs + PDF/Email + 3 Task Handlers")
    print(f"{'#'*80}")
    print("Type 'exit' or 'quit' to stop")
    print("Type 'info' to see system information")
    print("Type 'budget' to start budget planning")
    print("Type 'risk' to start risk profiling\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'info':
                print("\n" + "="*60)
                print("System Information:")
                print("="*60)
                device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
                print(f"Device: {device}")
                print(f"Model: DragonLLM/Llama-Open-Finance-8B")
                print(f"RAG: Pinecone + Hybrid Search + BGE Reranking")
                print(f"Live Data: Finnhub, Alpha Vantage, Numbeo, NewsAPI")
                print(f"Tasks: QA, Budget Planning, Risk Profiling")
                print(f"Outputs: PDF Generation, Email Sending")
                print("="*60 + "\n")
                continue
            
            # Process query
            await run_single_query(agent, user_input)
            
        except KeyboardInterrupt:
            print(f"\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}\n")


async def run_test_queries(agent: FinancialAgentOrchestratorV4):
    """Run test queries demonstrating all features"""
    
    test_queries = [
        # QA with live data
        ("What is the current price of Apple stock (AAPL) and should I invest?",
         {"task": "qa_with_live_data"}),
        
        # Budget planning
        ("I earn Rs 1,00,000 per month and want to create a detailed budget. Please send me a PDF.",
         {"task": "budget", "monthly_income": 100000}),
        
        # Risk profiling
        ("I'm 40 years old with Rs 20 lakhs to invest. What's my risk profile? Send me the report to my email.",
         {"task": "risk", "age": 40, "investment_amount": 2000000}),
    ]
    
    print(f"\n{'#'*80}")
    print("RUNNING TEST QUERIES - V4 (Advanced Features)")
    print(f"{'#'*80}\n")
    
    results = []
    
    for i, (query, context) in enumerate(test_queries):
        print(f"\n{'─'*80}")
        print(f"Test {i+1}/{len(test_queries)}")
        print(f"{'─'*80}")
        print(f"Query: {query}")
        print(f"Context: {context}")
        print(f"{'─'*80}\n")
        
        response = await run_single_query(agent, query, context)
        if response:
            results.append(response)
    
    # Summary
    print(f"\n{'#'*80}")
    print("TEST SUMMARY")
    print(f"{'#'*80}")
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(test_queries) - len(results)}")
    print(f"{'#'*80}\n")

async def main():
    """Main entry point"""
    
    print(f"\n{'#'*80}")
    print("🚀 FINANCIAL AI AGENT V4 - ADVANCED SYSTEM INITIALIZATION")
    print("DragonLLM + RAG + Live APIs + PDF/Email + 3 Task Handlers")
    print(f"{'#'*80}\n")
    
    # Step 1: Check dependencies
    print("1️⃣  Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Missing dependencies")
        print("Install: pip install -r requirements_v4.txt")
        sys.exit(1)
    print("✓ All dependencies installed\n")
    
    # Step 2: Setup environment
    print("2️⃣  Setting up environment...")
    setup_environment()
    print("✓ Environment configured\n")
    
    # Step 3: Validate configuration
    print("3️⃣  Validating configuration...")
    if not validate_config():
        print("\n❌ Configuration invalid")
        print("Please update config.py with your API keys")
        sys.exit(1)
    print("✓ Configuration valid\n")
    
    # Step 4: Initialize components
    print("4️⃣  Initializing V4 Agent Components...\n")
    
    try:
        # LLM Handler
        print("   - Initializing LLM Handler...")
        llm_handler = LLMHandler(llm_type=Config.LLM_TYPE)
        print(f"     ✓ LLM ({Config.LLM_TYPE}) ready")
        
        # Agent V4
        print("   - Initializing Agent V4 (Advanced System)...")
        print("     * DragonLLM Task Detector")
        print("     * Pinecone + Hybrid RAG")
        print("     * Live Data Providers (Finnhub, Alpha Vantage, Numbeo)")
        print("     * PDF Generator & Email Sender")
        print("     * 3 Specialized Task Handlers")
        
        agent = FinancialAgentOrchestratorV4(llm_handler)
        print("     ✓ Agent V4 ready\n")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)
    
    print(f"{'#'*80}")
    print("✓ INITIALIZATION COMPLETE")
    print(f"{'#'*80}\n")
    
    # Step 5: Run interface
    print("5️⃣  Starting Financial AI Agent...\n")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == '--interactive':
            await run_interactive_mode(agent)
        elif sys.argv[1].lower() == '--test':
            await run_test_queries(agent)
        else:
            query = ' '.join(sys.argv[1:])
            await run_single_query(agent, query)
    else:
        # Show usage
        print("Usage:")
        print("  python main.py --interactive     # Interactive mode")
        print("  python main.py --test             # Run test queries")
        print("  python main.py <query>            # Process specific query\n")
        
        print("Running demo...\n")
        
        demo_query = "I am leaving in the Newyork, I am earning 10000 dollar per week, give me the budget palnner in pdf format."
        # await run_single_query(agent, demo_query)

        await run_interactive_mode(agent)
    
    print(f"\n{'#'*80}")
    print("✓ SESSION COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye! 👋")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
