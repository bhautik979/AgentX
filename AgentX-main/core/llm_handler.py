import logging
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMBase(ABC):
    """Base class for all LLM implementations"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        pass

class OllamaLLM(LLMBase):
    """Local LLM using Ollama"""
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        logger.info(f"🚀 Initializing Ollama LLM: {model_name}")
        try:
            import requests
            # Test connection
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                logger.info("✅ Connected to Ollama server")
            else:
                raise Exception("Failed to connect to Ollama")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama at {base_url}")
            logger.error("Make sure Ollama is running: ollama run mistral")
            raise
        
        self.model_name = model_name
        self.base_url = base_url
        import requests
        self.requests = requests
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using Ollama"""
        try:
            logger.info(f"🤖 Generating with {self.model_name}...")
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                logger.info("✅ Generation complete")
                return generated_text
            else:
                logger.error(f"❌ Error: {response.status_code}")
                return "Error generating response"
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error: {str(e)}"

class OpenAILLM(LLMBase):
    """OpenAI GPT models"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        logger.info(f"🚀 Initializing OpenAI LLM: {model_name}")
        
        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            logger.info("✅ OpenAI client initialized")
        except ImportError:
            logger.error("❌ openai package not installed")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using OpenAI"""
        try:
            logger.info(f"🤖 Generating with {self.model_name}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            logger.info("✅ Generation complete")
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error: {str(e)}"

class LLMHandler:
    """Unified handler for multiple LLM providers"""
    
    def __init__(self, llm_type: str = "ollama", model_name: str = "mistral", api_key: str = None):
        """Initialize LLM Handler"""
        logger.info(f"🚀 Initializing LLMHandler with {llm_type}")
        
        try:
            if llm_type.lower() == "ollama":
                self.llm = OllamaLLM(model_name)
                self.llm_type = "ollama"
            elif llm_type.lower() == "openai":
                self.llm = OpenAILLM(model_name, api_key)
                self.llm_type = "openai"
            else:
                raise ValueError(f"Unknown LLM type: {llm_type}")
            
            logger.info(f"✅ LLMHandler ready ({llm_type})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLMHandler: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        return self.llm.generate(prompt, max_tokens, temperature)
    
    def generate_with_validation(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate with basic validation"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = self.generate(prompt, max_tokens=max_tokens)
                
                # Validate response
                if len(response.strip()) < 20:
                    logger.warning(f"⚠️ Response too short (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        continue
                
                return response
                
            except Exception as e:
                logger.error(f"❌ Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return "Unable to generate a response. Please try again."
        
        return "Unable to generate a valid response."


if __name__ == "__main__":
    # Test with Ollama
    handler = LLMHandler(llm_type="ollama", model_name="mistral")
    
    test_prompt = """You are a financial expert. Answer this question:
    What is an emergency fund and why is it important?
    
    Answer in 2-3 sentences."""
    
    logger.info("\n🧪 Testing LLM Generation...")
    response = handler.generate(test_prompt, max_tokens=200)
    print(f"\n📝 Response:\n{response}")