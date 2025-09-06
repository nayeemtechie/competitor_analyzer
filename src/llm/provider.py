# src/llm/provider.py
"""
LLM provider for competitor analysis
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PERPLEXITY = "perplexity"

@dataclass
class LLMResponse:
    """LLM response container"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

class LLMProviderManager:
    """Manages multiple LLM providers with fallback support"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            try:
                import openai
                self.providers[LLMProvider.OPENAI] = OpenAIProvider()
                if not self.default_provider:
                    self.default_provider = LLMProvider.OPENAI
                logger.info("OpenAI provider initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
        
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                self.providers[LLMProvider.ANTHROPIC] = AnthropicProvider()
                if not self.default_provider:
                    self.default_provider = LLMProvider.ANTHROPIC
                logger.info("Anthropic provider initialized")
            except ImportError:
                logger.warning("Anthropic library not available")
        
        # Perplexity
        if os.getenv('PERPLEXITY_API_KEY'):
            self.providers[LLMProvider.PERPLEXITY] = PerplexityProvider()
            logger.info("Perplexity provider initialized")
        
        if not self.providers:
            raise RuntimeError("No LLM providers available. Set API keys for OpenAI, Anthropic, or Perplexity.")
    
    async def chat(self, 
                   system: str,
                   user: str,
                   model: str = "gpt-4o",
                   provider: Optional[LLMProvider] = None,
                   temperature: float = 0.3,
                   max_tokens: int = 4000) -> str:
        """Send chat completion request with fallback support"""
        
        # Determine provider from model name if not specified
        if not provider:
            provider = self._get_provider_for_model(model)
        
        if provider not in self.providers:
            provider = self.default_provider
        
        if not provider:
            raise RuntimeError("No available LLM provider")
        
        try:
            response = await self.providers[provider].chat(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content
            
        except Exception as e:
            logger.warning(f"{provider.value} provider failed: {e}")
            
            # Try fallback providers
            for fallback_provider in self.providers:
                if fallback_provider != provider:
                    try:
                        fallback_model = self._get_fallback_model(fallback_provider)
                        response = await self.providers[fallback_provider].chat(
                            system=system,
                            user=user,
                            model=fallback_model,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        logger.info(f"Fallback to {fallback_provider.value} successful")
                        return response.content
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {fallback_provider.value} also failed: {fallback_error}")
                        continue
            
            raise RuntimeError(f"All LLM providers failed. Last error: {e}")
    
    def _get_provider_for_model(self, model: str) -> LLMProvider:
        """Determine provider based on model name"""
        if model.startswith(('gpt-', 'o1-')):
            return LLMProvider.OPENAI
        elif model.startswith(('claude-', 'sonnet', 'opus', 'haiku')):
            return LLMProvider.ANTHROPIC
        elif model.startswith('sonar'):
            return LLMProvider.PERPLEXITY
        else:
            return self.default_provider
    
    def _get_fallback_model(self, provider: LLMProvider) -> str:
        """Get fallback model for provider"""
        fallback_models = {
            LLMProvider.OPENAI: "gpt-4o-mini",
            LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
            LLMProvider.PERPLEXITY: "sonar-small-online"
        }
        return fallback_models.get(provider, "gpt-4o-mini")

class OpenAIProvider:
    """OpenAI provider implementation"""
    
    def __init__(self):
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    async def chat(self, system: str, user: str, model: str = "gpt-4o", 
                   temperature: float = 0.3, max_tokens: int = 4000) -> LLMResponse:
        """Send chat completion to OpenAI"""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="openai",
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicProvider:
    """Anthropic provider implementation"""
    
    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
    
    async def chat(self, system: str, user: str, model: str = "claude-3-sonnet-20240229",
                   temperature: float = 0.3, max_tokens: int = 4000) -> LLMResponse:
        """Send message to Anthropic"""
        try:
            response = await self.client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=model,
                provider="anthropic",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class PerplexityProvider:
    """Perplexity provider implementation"""
    
    def __init__(self):
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv('PERPLEXITY_API_KEY'),
            base_url="https://api.perplexity.ai"
        )
    
    async def chat(self, system: str, user: str, model: str = "sonar-pro",
                   temperature: float = 0.3, max_tokens: int = 4000) -> LLMResponse:
        """Send chat completion to Perplexity"""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="perplexity",
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            raise

# Main LLMProvider class for backward compatibility
class LLMProvider:
    """Main LLM provider interface"""
    
    def __init__(self):
        self.manager = LLMProviderManager()
    
    def chat(self, system: str, user: str, model: str = "gpt-4o", **kwargs) -> str:
        """Synchronous chat interface"""
        return asyncio.run(self.manager.chat(system, user, model, **kwargs))
    
    async def achat(self, system: str, user: str, model: str = "gpt-4o", **kwargs) -> str:
        """Asynchronous chat interface"""
        return await self.manager.chat(system, user, model, **kwargs)