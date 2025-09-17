# src/llm/provider.py
"""
Multi-provider LLM interface with fallback support for competitor analysis.
Supports OpenAI, Anthropic, and Perplexity APIs with automatic failover.
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

# Third-party imports (install with: pip install openai anthropic aiohttp)
try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:
    anthropic = None
    AsyncAnthropic = None

import aiohttp
from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PERPLEXITY = "perplexity"


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    latency_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class LLMRequest:
    """Standardized LLM request format."""
    system_prompt: str
    user_prompt: str
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4000
    json_mode: bool = False
    retry_attempts: int = 3


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.provider_name = self.__class__.__name__.replace("Client", "").lower()
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        super().__init__(api_key, model_name)
        if not openai:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.cost_per_token = {
            "gpt-4o": {"input": 0.000005, "output": 0.000015},
            "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015}
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt}
            ]
            
            # Prepare request parameters
            params = {
                "model": request.model or self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            # Add JSON mode if requested (only for supported models)
            if request.json_mode and "gpt-4" in params["model"]:
                params["response_format"] = {"type": "json_object"}
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = response.usage.total_tokens if response.usage else None
            cost_estimate = self._calculate_cost(
                params["model"], 
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=params["model"],
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content="",
                provider="openai",
                model=request.model or self.model_name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for the request."""
        if model not in self.cost_per_token:
            return 0.0
        
        costs = self.cost_per_token[model]
        return (input_tokens * costs["input"]) + (output_tokens * costs["output"])
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.api_key and openai)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key, model_name)
        if not anthropic:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.cost_per_token = {
            "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
            "claude-3-5-haiku-20241022": {"input": 0.0000008, "output": 0.000004},
            "claude-3-opus-20240229": {"input": 0.000015, "output": 0.000075}
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            # Prepare the prompt
            full_prompt = f"{request.system_prompt}\n\nHuman: {request.user_prompt}\n\nAssistant:"
            
            # Make API call
            response = await self.client.messages.create(
                model=request.model or self.model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt,
                messages=[
                    {"role": "user", "content": request.user_prompt}
                ]
            )
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost_estimate = self._calculate_cost(
                request.model or self.model_name,
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            
            return LLMResponse(
                content=response.content[0].text,
                provider="anthropic",
                model=request.model or self.model_name,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content="",
                provider="anthropic",
                model=request.model or self.model_name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for the request."""
        if model not in self.cost_per_token:
            return 0.0
        
        costs = self.cost_per_token[model]
        return (input_tokens * costs["input"]) + (output_tokens * costs["output"])
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return bool(self.api_key and anthropic)


class PerplexityClient(BaseLLMClient):
    """Perplexity API client (OpenAI-compatible)."""
    
    def __init__(self, api_key: str, model_name: str = "llama-3.1-sonar-large-128k-online"):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.perplexity.ai"
        self.cost_per_token = {
            "llama-3.1-sonar-large-128k-online": {"input": 0.000001, "output": 0.000001},
            "llama-3.1-sonar-small-128k-online": {"input": 0.0000002, "output": 0.0000002}
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Perplexity API."""
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": request.model or self.model_name,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            timeout = ClientTimeout(total=60)
            async with ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    result = await response.json()
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            cost_estimate = self._calculate_cost(
                payload["model"],
                result.get("usage", {}).get("prompt_tokens", 0),
                result.get("usage", {}).get("completion_tokens", 0)
            )
            
            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                provider="perplexity",
                model=payload["model"],
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return LLMResponse(
                content="",
                provider="perplexity",
                model=request.model or self.model_name,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for the request."""
        if model not in self.cost_per_token:
            return 0.0
        
        costs = self.cost_per_token[model]
        return (input_tokens * costs["input"]) + (output_tokens * costs["output"])
    
    def is_available(self) -> bool:
        """Check if Perplexity is available."""
        return bool(self.api_key)


class LLMProviderManager:
    """Manages multiple LLM providers with automatic fallback."""
    
    def __init__(self, config=None):
        """
        Initialize the LLM provider manager.
        
        Args:
            config: Configuration object with LLM settings
        """
        self.config = config
        self.providers: Dict[str, BaseLLMClient] = {}
        self.primary_provider = None
        self.fallback_providers = []
        
        # Initialize providers
        self._initialize_providers()
        
        # Set up provider hierarchy
        self._setup_provider_hierarchy()
    
    def _initialize_providers(self):
        """Initialize available LLM providers based on API keys."""
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai:
            try:
                openai_model = "gpt-4o"
                if self.config and hasattr(self.config, 'llm'):
                    openai_model = self.config.llm.models.get("analysis", "gpt-4o")
                
                self.providers["openai"] = OpenAIClient(openai_key, openai_model)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic:
            try:
                anthropic_model = "claude-3-5-sonnet-20241022"
                if self.config and hasattr(self.config, 'llm'):
                    anthropic_model = self.config.llm.models.get("analysis", anthropic_model)
                
                self.providers["anthropic"] = AnthropicClient(anthropic_key, anthropic_model)
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")
        
        # Perplexity
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            try:
                perplexity_model = "llama-3.1-sonar-large-128k-online"
                if self.config and hasattr(self.config, 'llm'):
                    perplexity_model = self.config.llm.models.get("analysis", perplexity_model)
                
                self.providers["perplexity"] = PerplexityClient(perplexity_key, perplexity_model)
                logger.info("Perplexity provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity provider: {e}")
        
        if not self.providers:
            raise RuntimeError("No LLM providers available. Please set API keys in environment variables.")
    
    def _setup_provider_hierarchy(self):
        """Set up primary and fallback providers based on configuration."""
        # Determine primary provider from config
        primary_name = "openai"  # default
        if self.config and hasattr(self.config, 'llm'):
            primary_name = self.config.llm.provider
        
        # Set primary provider
        if primary_name in self.providers:
            self.primary_provider = self.providers[primary_name]
            logger.info(f"Primary provider set to: {primary_name}")
        else:
            # Fallback to first available provider
            available_providers = list(self.providers.keys())
            if available_providers:
                primary_name = available_providers[0]
                self.primary_provider = self.providers[primary_name]
                logger.warning(f"Configured provider not available. Using: {primary_name}")
        
        # Set fallback providers (all others)
        self.fallback_providers = [
            provider for name, provider in self.providers.items() 
            if name != primary_name
        ]
        
        logger.info(f"Fallback providers: {[p.provider_name for p in self.fallback_providers]}")
    
    async def generate(self, 
                      system_prompt: str, 
                      user_prompt: str,
                      model: Optional[str] = None,
                      temperature: float = 0.3,
                      max_tokens: int = 4000,
                      json_mode: bool = False,
                      retry_attempts: int = 3) -> LLMResponse:
        """
        Generate a response using the best available provider.
        
        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query/content
            model: Specific model to use (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to request JSON format response
            retry_attempts: Number of retry attempts per provider
            
        Returns:
            LLMResponse object with the generated content
        """
        request = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            retry_attempts=retry_attempts
        )
        
        # Try primary provider first
        if self.primary_provider:
            for attempt in range(retry_attempts):
                try:
                    response = await self.primary_provider.generate(request)
                    if response.success:
                        return response
                    else:
                        logger.warning(f"Primary provider failed (attempt {attempt + 1}): {response.error}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    logger.warning(f"Primary provider error (attempt {attempt + 1}): {e}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)
        
        # Try fallback providers
        for fallback_provider in self.fallback_providers:
            logger.info(f"Trying fallback provider: {fallback_provider.provider_name}")
            
            for attempt in range(retry_attempts):
                try:
                    response = await fallback_provider.generate(request)
                    if response.success:
                        logger.info(f"Fallback provider {fallback_provider.provider_name} succeeded")
                        return response
                    else:
                        logger.warning(f"Fallback provider failed (attempt {attempt + 1}): {response.error}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logger.warning(f"Fallback provider error (attempt {attempt + 1}): {e}")
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)
        
        # All providers failed
        error_msg = "All LLM providers failed to generate a response"
        logger.error(error_msg)
        return LLMResponse(
            content="",
            provider="none",
            model=model or "unknown",
            success=False,
            error=error_msg
        )
    
    async def generate_json(self, 
                           system_prompt: str, 
                           user_prompt: str,
                           schema: Optional[Dict[str, Any]] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            schema: Optional JSON schema for validation
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON dictionary
        """
        # Enhance prompts for JSON output
        enhanced_system = f"{system_prompt}\n\nIMPORTANT: Respond only with valid JSON. Do not include any text outside the JSON structure."
        
        if schema:
            enhanced_user = f"{user_prompt}\n\nPlease format your response as JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        else:
            enhanced_user = f"{user_prompt}\n\nPlease format your response as valid JSON."
        
        response = await self.generate(
            system_prompt=enhanced_system,
            user_prompt=enhanced_user,
            json_mode=True,
            **kwargs
        )
        
        if not response.success:
            raise Exception(f"LLM generation failed: {response.error}")
        
        # Parse JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            content = response.content.strip()
            
            # Remove common markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                raise Exception(f"Invalid JSON response from LLM: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all providers."""
        status = {}
        for name, provider in self.providers.items():
            status[name] = {
                "available": provider.is_available(),
                "model": provider.model_name,
                "is_primary": provider == self.primary_provider
            }
        return status
    
    async def test_connection(self, provider_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Test connection to providers.
        
        Args:
            provider_name: Specific provider to test, or None for all
            
        Returns:
            Dictionary mapping provider names to connection status
        """
        results = {}
        
        providers_to_test = {provider_name: self.providers[provider_name]} if provider_name else self.providers
        
        for name, provider in providers_to_test.items():
            try:
                test_request = LLMRequest(
                    system_prompt="You are a test assistant.",
                    user_prompt="Respond with exactly: 'Connection test successful'",
                    max_tokens=50,
                    temperature=0
                )
                
                response = await provider.generate(test_request)
                results[name] = response.success and "successful" in response.content.lower()
                
            except Exception as e:
                logger.error(f"Connection test failed for {name}: {e}")
                results[name] = False
        
        return results


# Convenience functions for easy usage
async def quick_generate(system_prompt: str, user_prompt: str, **kwargs) -> str:
    """Quick generation without managing providers."""
    manager = LLMProviderManager()
    response = await manager.generate(system_prompt, user_prompt, **kwargs)
    
    if not response.success:
        raise Exception(f"Generation failed: {response.error}")
    
    return response.content


async def quick_generate_json(system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
    """Quick JSON generation without managing providers."""
    manager = LLMProviderManager()
    return await manager.generate_json(system_prompt, user_prompt, **kwargs)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize provider manager
        manager = LLMProviderManager()
        
        # Test connection
        print("Testing connections...")
        connection_status = await manager.test_connection()
        print(f"Connection status: {connection_status}")
        
        # Generate a response
        print("\nGenerating competitor analysis example...")
        response = await manager.generate(
            system_prompt="You are an expert business analyst specializing in competitive intelligence.",
            user_prompt="Analyze the key competitive advantages of a search technology company like Algolia.",
            max_tokens=500
        )
        
        if response.success:
            print(f"Response ({response.provider}, {response.latency_ms:.0f}ms, ${response.cost_estimate:.4f}):")
            print(response.content)
        else:
            print(f"Generation failed: {response.error}")
        
        # Generate JSON response
        print("\nGenerating JSON analysis...")
        try:
            json_result = await manager.generate_json(
                system_prompt="You are a competitive analysis expert.",
                user_prompt="Create a SWOT analysis for a search technology company. Include exactly 3 items for each category.",
                schema={
                    "strengths": ["string"],
                    "weaknesses": ["string"], 
                    "opportunities": ["string"],
                    "threats": ["string"]
                }
            )
            print(f"JSON Result: {json.dumps(json_result, indent=2)}")
        except Exception as e:
            print(f"JSON generation failed: {e}")
    
    # Run the example
    asyncio.run(main())