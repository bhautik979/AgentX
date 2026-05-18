import logging
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import (
        API_HOST, API_PORT, DEBUG, PINECONE_API_KEY,
        PINECONE_INDEX, EMBED_MODEL, LLM_TYPE, LLM_MODEL,
        OLLAMA_BASE_URL, OPENAI_API_KEY
    )
except Exception as e:
    logger.error(f"❌ Failed to load config: {e}")
    sys.exit(1)

# Import core components
try:
    from core_rag_pipeline import FinancialRAGPipeline
    from core_llm_handler import LLMHandler
    from core_agent import FinancialAdvisorAgent
    from api_tools import BudgetPlanner
except Exception as e:
    logger.error(f"❌ Failed to import core components: {e}")
    sys.exit(1)

# Import models
from api_models import AskRequest, AskResponse, ChatMessage, BudgetRequest, BudgetResponse, EmergencyFundRequest, EmergencyFundResponse, ConversationResponse, HealthResponse, ErrorResponse

logger.info("🚀 Starting Financial RAG System Initialization...")

# ==================== Global Objects ====================
rag_pipeline = None
llm_handler = None
agent = None
budget_planner = None

# ==================== Startup & Shutdown ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # STARTUP
    logger.info("🔄 Starting up...")
    try:
        global rag_pipeline, llm_handler, agent, budget_planner
        
        # Initialize RAG Pipeline
        logger.info(f"📚 Initializing RAG Pipeline...")
        rag_pipeline = FinancialRAGPipeline(
            pinecone_api_key=PINECONE_API_KEY,
            embed_model_name=EMBED_MODEL,
            index_name=PINECONE_INDEX
        )
        logger.info("✅ RAG Pipeline ready")
        
        # Initialize LLM Handler
        logger.info(f"🤖 Initializing LLM Handler ({LLM_TYPE})...")
        llm_handler = LLMHandler(
            llm_type=LLM_TYPE,
            model_name=LLM_MODEL,
            api_key=OPENAI_API_KEY if LLM_TYPE == "openai" else None
        )
        logger.info("✅ LLM Handler ready")
        
        # Initialize Tools
        logger.info("🔧 Initializing tools...")
        budget_planner = BudgetPlanner()
        logger.info("✅ Tools ready")
        
        # Initialize Agent
        logger.info("🤝 Initializing Agent...")
        agent = FinancialAdvisorAgent(
            rag_pipeline=rag_pipeline,
            llm_handler=llm_handler,
            tools={
                "budget_planner": budget_planner
            }
        )
        logger.info("✅ Agent ready")
        
        logger.info("✅ System fully initialized and ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # App runs here
    
    # SHUTDOWN
    logger.info("🔄 Shutting down...")
    logger.info("✅ Shutdown complete")

# ==================== Create FastAPI App ====================
app = FastAPI(
    title="Financial Literacy RAG System",
    description="Expert AI agent for budgeting and saving guidance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ==================== CORS Middleware ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Health Check ====================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health and component status"""
    components = {
        "rag_pipeline": "ready" if rag_pipeline else "not_ready",
        "llm_handler": "ready" if llm_handler else "not_ready",
        "agent": "ready" if agent else "not_ready",
        "budget_planner": "ready" if budget_planner else "not_ready"
    }
    
    all_ready = all(v == "ready" for v in components.values())
    
    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        components=components,
        version="1.0.0"
    )

# ==================== Q&A Endpoint ====================
@app.post("/api/v1/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    """Ask a financial question and get expert advice"""
    try:
        logger.info(f"📥 Received Q&A request: {request.query[:50]}...")
        
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Process query through agent
        response = agent.process_query(request.user_id, request.query)
        
        return AskResponse(
            answer=response["answer"],
            sources=response.get("sources", []),
            type=response.get("type", "qa"),
            confidence=response.get("confidence", 0.5),
            intent=response.get("intent")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# ==================== Chat Endpoint ====================
@app.post("/api/v1/chat/message", response_model=AskResponse, tags=["Chat"])
async def send_message(message: ChatMessage):
    """Send a message in multi-turn conversation"""
    try:
        logger.info(f"💬 Received chat message from {message.user_id}")
        
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        response = agent.process_query(message.user_id, message.content)
        
        return AskResponse(
            answer=response["answer"],
            sources=response.get("sources", []),
            type=response.get("type", "general"),
            confidence=response.get("confidence", 0.5),
            intent=response.get("intent")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in send_message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Conversation History ====================
@app.get("/api/v1/chat/history/{user_id}", response_model=ConversationResponse, tags=["Chat"])
async def get_conversation(user_id: str):
    """Get conversation history for a user"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        if user_id not in agent.conversations:
            raise HTTPException(status_code=404, detail=f"No conversation found for user {user_id}")
        
        conv = agent.conversations[user_id]
        
        return ConversationResponse(
            user_id=user_id,
            message_count=len(conv.messages),
            messages=[msg.to_dict() for msg in conv.messages],
            user_profile=conv.user_profile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Budget Report ====================
@app.post("/api/v1/reports/budget", response_model=BudgetResponse, tags=["Reports"])
async def generate_budget_report(request: BudgetRequest):
    """Generate personalized budget report"""
    try:
        logger.info(f"📊 Generating budget report for {request.user_id}")
        
        if not budget_planner:
            raise HTTPException(status_code=503, detail="Budget tool not available")
        
        # Generate report
        file_path = budget_planner.generate(
            user_id=request.user_id,
            income=request.income,
            expenses=request.expenses,
            output_format=request.output_format
        )
        
        logger.info(f"✅ Report generated: {file_path}")
        
        return BudgetResponse(
            message="Budget report generated successfully",
            file_path=file_path,
            file_format=request.output_format,
            income=request.income,
            expenses=request.expenses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating budget report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Emergency Fund ====================
@app.post("/api/v1/reports/emergency-fund", response_model=EmergencyFundResponse, tags=["Reports"])
async def calculate_emergency_fund(request: EmergencyFundRequest):
    """Calculate and provide emergency fund guidance"""
    try:
        logger.info(f"🆘 Calculating emergency fund for {request.user_id}")
        
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Use agent to generate advice
        query = f"Calculate emergency fund for Rs {request.monthly_expenses} monthly expenses"
        response = agent.process_query(request.user_id, query)
        
        emergency_fund = request.monthly_expenses * 6
        monthly_target = emergency_fund / 12
        
        return EmergencyFundResponse(
            monthly_expenses=request.monthly_expenses,
            emergency_fund_amount=emergency_fund,
            months_covered=6,
            monthly_savings_target=monthly_target,
            advice=response["answer"],
            timeline_months=12
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error calculating emergency fund: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Error Handlers ====================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "status_code": 400}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )

# ==================== Root Endpoint ====================
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Financial Literacy RAG System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "qa": "/api/v1/ask",
            "chat": "/api/v1/chat/message",
            "budget": "/api/v1/reports/budget"
        }
    }

# ==================== Main ====================
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting FastAPI server on {API_HOST}:{API_PORT}")
    logger.info(f"📚 Docs available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        reload=DEBUG
    )
