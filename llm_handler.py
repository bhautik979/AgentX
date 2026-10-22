# ============================================================================
# LLM HANDLER MODULE
# Handles all LLM interactions with error handling and async support
# ============================================================================

import asyncio
import logging
from typing import Optional
import torch

from config import Config, logger

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Unified LLM handler supporting multiple LLM backends
    Provides async generation capabilities
    """
    
    def __init__(self, llm_type: str = "gemini"):
        """
        Initialize LLM handler
        
        Args:
            llm_type: Type of LLM to use ('gemini', 'ollama', 'huggingface')
        """
        self.llm_type = llm_type.lower()
        self.model = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        logger.info(f"Initializing LLM handler: {self.llm_type}")
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on configured type"""
        
        try:
            if self.llm_type == "gemini":
                self._init_gemini()
            elif self.llm_type == "ollama":
                self._init_ollama()
            elif self.llm_type == "huggingface":
                self._init_huggingface()
            else:
                logger.error(f"Unknown LLM type: {self.llm_type}")
                raise ValueError(f"Unknown LLM type: {self.llm_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_gemini(self):
        """Initialize Google Gemini LLM"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            self.model = ChatGoogleGenerativeAI(
                model=Config.LLM_MODEL,
                temperature=0.2,
                max_output_tokens=2000,
                google_api_key=Config.GOOGLE_API_KEY
            )
            
            logger.info("✓ Gemini LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            raise
    
    def _init_ollama(self):
        """Initialize Ollama LLM (local)"""
        try:
            from langchain.llms import Ollama
            
            self.model = Ollama(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_MODEL,
                temperature=0.2
            )
            
            logger.info("✓ Ollama LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            raise
    
    def _init_huggingface(self):
        """Initialize HuggingFace LLM"""
        try:
            from langchain.llms import HuggingFaceHub
            
            self.model = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                model_kwargs={"temperature": 0.2, "max_length": 2000}
            )
            
            logger.info("✓ HuggingFace LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"HuggingFace initialization failed: {e}")
            raise
    
    async def generate(self, 
                      prompt: str,
                      max_tokens: Optional[int] = None) -> str:
        """
        Generate text using LLM with retry logic
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens in response (optional)
        
        Returns:
            Generated text
        
        Raises:
            Exception: If generation fails after all retries
        """
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM generation attempt {attempt + 1}/{self.max_retries}")
                
                # Try to invoke the model
                if hasattr(self.model, 'apredict'):
                    # Async invoke
                    response = await self.model.apredict(prompt)
                
                elif hasattr(self.model, 'ainvoke'):
                    # Async invoke (newer LangChain)
                    response = await self.model.ainvoke(prompt)
                
                elif hasattr(self.model, 'predict'):
                    # Sync invoke wrapped in async
                    response = await asyncio.to_thread(
                        self.model.predict, 
                        prompt
                    )
                
                else:
                    # Fallback to invoke
                    response = await asyncio.to_thread(
                        self.model.invoke, 
                        prompt
                    )
                
                # Process response
                if isinstance(response, str):
                    result = response.strip()
                else:
                    result = str(response).strip()
                
                logger.debug(f"LLM generation successful (length: {len(result)})")
                return result
                
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    # All retries exhausted
                    logger.error(f"LLM generation failed after {self.max_retries} attempts")
                    raise
        
        raise Exception("Unexpected error in LLM generation")
    
    async def generate_structured(self,
                                 prompt: str,
                                 output_format: str = "json") -> str:
        """
        Generate structured output (JSON, etc.)
        
        Args:
            prompt: Input prompt
            output_format: Expected output format ('json', 'csv', 'list')
        
        Returns:
            Structured output as string
        """
        
        structured_prompt = f"""
{prompt}

Please respond in {output_format} format ONLY.
No additional text or explanations.
"""
        
        return await self.generate(structured_prompt)
    
    def is_available(self) -> bool:
        """Check if LLM is available and responsive"""
        
        try:
            if self.model is None:
                return False
            
            # Simple check - try a minimal call
            test_prompt = "Hello"
            
            # Non-async check
            if hasattr(self.model, 'predict'):
                response = self.model.predict(test_prompt)
                return bool(response)
            elif hasattr(self.model, 'invoke'):
                response = self.model.invoke(test_prompt)
                return bool(response)
            else:
                return True  # Assume available if initialized
                
        except Exception as e:
            logger.error(f"LLM availability check failed: {e}")
            return False


class LLMFallback:
    """
    Fallback system for LLM failures
    Tries multiple LLM types in sequence
    """
    
    def __init__(self):
        """Initialize fallback system"""
        self.primary_llm = None
        self.fallback_llms = []
        self.current_llm = None
        
        logger.info("Initializing LLM fallback system")
    
    def set_primary(self, llm: LLMHandler):
        """Set primary LLM"""
        self.primary_llm = llm
        self.current_llm = llm
        logger.info(f"Primary LLM set: {llm.llm_type}")
    
    def add_fallback(self, llm: LLMHandler):
        """Add fallback LLM"""
        self.fallback_llms.append(llm)
        logger.info(f"Fallback LLM added: {llm.llm_type}")
    
    async def generate(self, prompt: str) -> str:
        """
        Generate using primary LLM, fallback to others if needed
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        
        llms_to_try = [self.primary_llm] + self.fallback_llms
        
        for llm in llms_to_try:
            if llm is None:
                continue
            
            try:
                logger.info(f"Trying LLM: {llm.llm_type}")
                
                result = await llm.generate(prompt)
                
                if result:
                    self.current_llm = llm
                    logger.info(f"✓ Success with {llm.llm_type}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Failed with {llm.llm_type}: {e}")
                continue
        
        # All LLMs failed
        raise Exception("All LLMs failed to generate response")


__all__ = ['LLMHandler', 'LLMFallback']
