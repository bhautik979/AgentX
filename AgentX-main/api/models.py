from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

# ==================== Request Models ====================

class AskRequest(BaseModel):
    """Request model for Q&A endpoint"""
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    user_id: str = Field(default="default_user", description="Unique user identifier")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I create an emergency fund?",
                "user_id": "user123",
                "top_k": 5
            }
        }

class ChatMessage(BaseModel):
    """Request model for chat endpoint"""
    user_id: str = Field(..., description="Unique user identifier")
    content: str = Field(..., min_length=1, max_length=500, description="User message")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "content": "I earn Rs 50,000 per month"
            }
        }

class BudgetRequest(BaseModel):
    """Request model for budget generation"""
    user_id: str = Field(..., description="Unique user identifier")
    income: float = Field(..., gt=0, description="Monthly income in rupees")
    expenses: Dict[str, float] = Field(..., description="Expense breakdown")
    goals: Optional[List[str]] = Field(default=None, description="Financial goals")
    output_format: str = Field(default="pdf", pattern="^(pdf|docx)$")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "income": 50000,
                "expenses": {
                    "rent": 12000,
                    "food": 8000,
                    "transport": 3000
                },
                "goals": ["emergency_fund", "investment"],
                "output_format": "pdf"
            }
        }

class EmergencyFundRequest(BaseModel):
    """Request model for emergency fund calculation"""
    user_id: str = Field(..., description="Unique user identifier")
    monthly_expenses: float = Field(..., gt=0, description="Total monthly expenses")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "monthly_expenses": 30000
            }
        }

# ==================== Response Models ====================

class AskResponse(BaseModel):
    """Response model for Q&A endpoint"""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default=[], description="Retrieved source chunks")
    type: str = Field(..., description="Response type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    intent: Optional[str] = Field(default=None, description="Detected user intent")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "An emergency fund should cover 3-6 months of expenses...",
                "sources": ["Source text 1", "Source text 2"],
                "type": "qa",
                "confidence": 0.85,
                "intent": "qa"
            }
        }

class BudgetResponse(BaseModel):
    """Response model for budget generation"""
    message: str = Field(..., description="Status message")
    file_path: str = Field(..., description="Path to generated file")
    file_format: str = Field(..., description="File format (pdf or docx)")
    income: float = Field(..., description="Monthly income used")
    expenses: Dict[str, float] = Field(..., description="Expenses breakdown")
    generated_at: datetime = Field(default_factory=datetime.now)

class EmergencyFundResponse(BaseModel):
    """Response model for emergency fund calculation"""
    monthly_expenses: float = Field(..., description="Monthly expenses")
    emergency_fund_amount: float = Field(..., description="Recommended emergency fund")
    months_covered: int = Field(default=6, description="Months of coverage")
    monthly_savings_target: float = Field(..., description="Monthly target to reach goal")
    advice: str = Field(..., description="Emergency fund advice")
    timeline_months: int = Field(default=12, description="Timeline to reach goal")

class ConversationResponse(BaseModel):
    """Response model for conversation history"""
    user_id: str = Field(..., description="User identifier")
    message_count: int = Field(..., description="Total messages")
    messages: List[Dict] = Field(..., description="Message history")
    user_profile: Dict = Field(..., description="User profile")

class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")
    components: Dict[str, str] = Field(default={})
