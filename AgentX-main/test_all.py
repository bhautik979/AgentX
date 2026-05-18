import sys
sys.path.insert(0, '.')

# Test 1: Test RAG Pipeline
print("🧪 TEST 1: RAG Pipeline")
print("="*50)
try:
    from config import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL
    from core.rag_pipeline import FinancialRAGPipeline
    
    rag = FinancialRAGPipeline(
        pinecone_api_key=PINECONE_API_KEY,
        embed_model_name=EMBED_MODEL,
        index_name=PINECONE_INDEX
    )
    print("✅ RAG Pipeline initialized successfully")
    
    # Test retrieval
    query = "emergency fund"
    results = rag.retrieve_with_reranking(query, top_k=3)
    print(f"✅ Retrieved {len(results)} results for query: '{query}'")
    
except Exception as e:
    print(f"❌ RAG Pipeline test failed: {e}")

# Test 2: Test LLM Handler
print("\n🧪 TEST 2: LLM Handler")
print("="*50)
try:
    from core.llm_handler import LLMHandler
    from config import LLM_TYPE, LLM_MODEL
    
    llm = LLMHandler(llm_type=LLM_TYPE, model_name=LLM_MODEL)
    print("✅ LLM Handler initialized successfully")
    
    # Test generation
    prompt = "What is an emergency fund? Answer in one sentence."
    response = llm.generate(prompt, max_tokens=100)
    print(f"✅ Generated response (length: {len(response)} chars)")
    
except Exception as e:
    print(f"❌ LLM Handler test failed: {e}")

# Test 3: Test Agent
print("\n🧪 TEST 3: Agent")
print("="*50)
try:
    from core.agent import FinancialAdvisorAgent, ConversationState
    from core.rag_pipeline import FinancialRAGPipeline
    from core.llm_handler import LLMHandler
    from config import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL, LLM_TYPE, LLM_MODEL
    
    rag = FinancialRAGPipeline(PINECONE_API_KEY, EMBED_MODEL, PINECONE_INDEX)
    llm = LLMHandler(LLM_TYPE, LLM_MODEL)
    agent = FinancialAdvisorAgent(rag, llm, {})
    print("✅ Agent initialized successfully")
    
    # Test conversation
    response = agent.process_query("test_user", "I earn Rs 50,000 per month")
    print(f"✅ Processed query successfully")
    print(f"   Response type: {response.get('type')}")
    print(f"   Confidence: {response.get('confidence', 0):.2f}")
    
except Exception as e:
    print(f"❌ Agent test failed: {e}")

# Test 4: Test Pydantic Models
print("\n🧪 TEST 4: Pydantic Models")
print("="*50)
try:
    from api.models import AskRequest, AskResponse, BudgetRequest, EmergencyFundRequest
    
    # Test AskRequest
    req = AskRequest(query="How to save money?", user_id="user1")
    print("✅ AskRequest model validated")
    
    # Test BudgetRequest
    budget_req = BudgetRequest(
        user_id="user1",
        income=50000,
        expenses={"rent": 12000, "food": 8000}
    )
    print("✅ BudgetRequest model validated")
    
    # Test EmergencyFundRequest
    emf_req = EmergencyFundRequest(user_id="user1", monthly_expenses=30000)
    print("✅ EmergencyFundRequest model validated")
    
except Exception as e:
    print(f"❌ Pydantic Models test failed: {e}")

# Test 5: Test Budget Planner
print("\n🧪 TEST 5: Budget Planner")
print("="*50)
try:
    from api.tools import BudgetPlanner
    
    planner = BudgetPlanner()
    print("✅ BudgetPlanner initialized")
    
    # Test DOCX generation
    docx_path = planner.generate(
        user_id="test_user",
        income=50000,
        expenses={"rent": 12000, "food": 8000, "transport": 3000},
        output_format="docx"
    )
    print(f"✅ Generated DOCX report: {docx_path}")
    
except Exception as e:
    print(f"❌ Budget Planner test failed: {e}")

print("\n" + "="*50)
print("✅ ALL TESTS COMPLETED")
print("="*50)