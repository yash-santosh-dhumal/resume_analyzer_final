"""
LLM Client Manager
Handles connections to different LLM providers (OpenAI, Hugging Face, etc.)
"""

from typing import Dict, Any, Optional, List
import os
import logging
from abc import ABC, abstractmethod

# LLM provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class OpenAIClient(LLMClient):
    """OpenAI GPT client (supports OpenRouter)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.is_openrouter = False
        
        if self.api_key and OPENAI_AVAILABLE:
            try:
                # Check if this is an OpenRouter API key
                if self.api_key.startswith("sk-or-v1"):
                    self.is_openrouter = True
                    from openai import OpenAI
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                    # Use OpenAI model through OpenRouter
                    self.model = "openai/gpt-3.5-turbo"
                    logger.info(f"OpenRouter client initialized with model: {self.model}")
                else:
                    # Standard OpenAI
                    from openai import OpenAI
                    self.client = OpenAI(api_key=self.api_key)
                    logger.info(f"OpenAI client initialized with model: {self.model}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, 
                     temperature: float = 0.3, **kwargs) -> str:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self.client is not None and OPENAI_AVAILABLE

class HuggingFaceClient(LLMClient):
    """Hugging Face Transformers client"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize Hugging Face client
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=-1  # CPU
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"Hugging Face client initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face client: {str(e)}")
    
    def generate_text(self, prompt: str, max_length: int = 512, 
                     temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text using Hugging Face model
        
        Args:
            prompt: Input prompt
            max_length: Maximum sequence length
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        if not self.is_available():
            raise RuntimeError("Hugging Face client not available")
        
        try:
            # Ensure prompt isn't too long
            if self.tokenizer:
                tokens = self.tokenizer.encode(prompt, truncation=True, max_length=400)
                prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            response = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                **kwargs
            )
            
            generated = response[0]['generated_text']
            # Remove the original prompt from the response
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            return generated
            
        except Exception as e:
            logger.error(f"Hugging Face text generation failed: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Check if Hugging Face client is available"""
        return self.pipeline is not None and TRANSFORMERS_AVAILABLE

class LLMManager:
    """
    Manager for multiple LLM clients with fallback support
    """
    
    def __init__(self, primary_provider: str = "openai", 
                 fallback_provider: str = "huggingface"):
        """
        Initialize LLM manager
        
        Args:
            primary_provider: Primary LLM provider to use
            fallback_provider: Fallback provider if primary fails
        """
        self.clients = {}
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        
        # Initialize available clients
        self._initialize_clients()
        
        logger.info(f"LLM Manager initialized - Primary: {primary_provider}, Fallback: {fallback_provider}")
    
    def _initialize_clients(self):
        """Initialize all available LLM clients"""
        # OpenAI client
        try:
            openai_client = OpenAIClient()
            if openai_client.is_available():
                self.clients["openai"] = openai_client
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {str(e)}")
        
        # Hugging Face client
        try:
            hf_client = HuggingFaceClient("microsoft/DialoGPT-small")  # Smaller model for fallback
            if hf_client.is_available():
                self.clients["huggingface"] = hf_client
        except Exception as e:
            logger.warning(f"Hugging Face client initialization failed: {str(e)}")
    
    def generate_text(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using specified or default provider
        
        Args:
            prompt: Input prompt
            provider: Specific provider to use (optional)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # Determine which provider to use
        target_provider = provider or self.primary_provider
        
        # Try primary provider
        if target_provider in self.clients:
            try:
                return self.clients[target_provider].generate_text(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Primary provider {target_provider} failed: {str(e)}")
        
        # Try fallback provider
        if self.fallback_provider in self.clients and self.fallback_provider != target_provider:
            try:
                logger.info(f"Using fallback provider: {self.fallback_provider}")
                return self.clients[self.fallback_provider].generate_text(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Fallback provider {self.fallback_provider} failed: {str(e)}")
        
        # If all providers fail, return empty string
        logger.error("All LLM providers failed")
        return ""
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return list(self.clients.keys())
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a specific provider is available"""
        return provider in self.clients and self.clients[provider].is_available()