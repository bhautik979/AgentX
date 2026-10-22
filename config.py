# ============================================================================
# COMPLETE FINANCIAL AI AGENT IMPLEMENTATION
# Production-Ready, Fully Integrated System
# ============================================================================
# This is the MAIN configuration file for the entire system
# All settings and imports are here
# ============================================================================

import os
import json
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Central configuration for the entire system"""
    
    # API Keys (REPLACE WITH YOUR OWN)
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    PINECONE_API_KEY = "PINECONE_API_KEY"
    
    # Pinecone Configuration
    PINECONE_INDEX = "financial-knowledge"
    PINECONE_DIMENSION = 768
    
    # LLM Configuration
    LLM_TYPE = "gemini"  # gemini, ollama, or huggingface
    LLM_MODEL = "gemini-2.5-flash"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "mistral"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "FinLang/finance-embeddings-investopedia"
    EMBEDDING_DIMENSION = 768
    
    # Agent Configuration
    MAX_VERIFICATION_ITERATIONS = 3
    VERIFICATION_CONFIDENCE_THRESHOLD = 0.75

    SENDGRID_API_KEY = "SG.xxxxxxxxx..."
    SENDGRID_FROM_EMAIL = "karanppatel2025@gmail.com"

    ANVIL_API_KEY = "ANVIL_API_KEY"

    
    # Storage paths
    DATA_DIR = "./data"
    REPORTS_DIR = "./data/reports"
    LOGS_DIR = "./logs"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, REPORTS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# ENUMS
# ============================================================================

class TaskType(Enum):
    """Task types the agent can handle"""
    GENERAL_QA = "general_qa"
    BUDGET_PLANNING = "budget_planning"
    RISK_PROFILING = "risk_profiling"
    DEBT_MANAGEMENT = "debt_management"
    SAVINGS_OPTIMIZATION = "savings_optimization"


class IssueSeverity(Enum):
    """Severity levels for issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueType(Enum):
    """Types of issues that can be detected"""
    FACTUAL_ERROR = "factual_error"
    LOGICAL_ERROR = "logical_error"
    DOMAIN_ERROR = "domain_error"
    COMPLETENESS_ERROR = "completeness_error"
    SOURCE_ERROR = "source_error"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Issue:
    """Represents a single issue detected in answer"""
    issue_type: IssueType
    severity: IssueSeverity
    reason: str
    confidence: float
    details: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of verification check"""
    has_issues: bool
    issues: List[Issue] = field(default_factory=list)
    primary_issue: Optional[Issue] = None
    confidence: float = 0.0
    specialist_needed: Optional[str] = None


@dataclass
class IterationLog:
    """Log of a single verification iteration"""
    iteration_number: int
    issues_detected: int
    issues: List[Issue]
    specialist_used: Optional[str] = None
    fix_applied: Optional[str] = None
    answer_before: str = ""
    answer_after: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResponse:
    """Final response from the agent"""
    query: str
    answer: str
    confidence: float
    task_type: TaskType
    iterations: int
    verification_log: List[IterationLog]
    specialists_used: List[str]
    sources: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_environment():
    """Setup environment variables and dependencies"""
    os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
    os.environ["PINECONE_API_KEY"] = Config.PINECONE_API_KEY
    logger.info("✓ Environment configured")


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'langchain': 'langchain',
        'transformers': 'transformers',
        'torch': 'torch',
        'pinecone': 'pinecone-client',
        'sentence_transformers': 'sentence-transformers',
        'FlagEmbedding': 'FlagEmbedding',
        'rank_bm25': 'rank-bm25',
        'nltk': 'nltk',
    }
    
    missing = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        logger.warning(f"⚠️  Missing packages: {', '.join(missing)}")
        logger.warning(f"Install with: pip install {' '.join(missing)}")
        return False
    
    logger.info("✓ All dependencies installed")
    return True


def validate_config():
    """Validate configuration"""
    if Config.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        logger.error("❌ Please set GOOGLE_API_KEY in config")
        return False
    if Config.PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_HERE":
        logger.error("❌ Please set PINECONE_API_KEY in config")
        return False
    logger.info("✓ Configuration validated")
    return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'Config',
    'TaskType',
    'IssueSeverity',
    'IssueType',
    'Issue',
    'VerificationResult',
    'IterationLog',
    'AgentResponse',
    'setup_environment',
    'check_dependencies',
    'validate_config',
    'logger'
]
