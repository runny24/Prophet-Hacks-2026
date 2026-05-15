"""Utility functions for OpenRouter API interactions."""

from typing import Any, Dict, List, Optional
import httpx
import json

# API Configuration
BASE_URL = "https://openrouter.ai/api/v1"


def check_credits(api_key: str, base_url: str = BASE_URL) -> Dict[str, Any]:
    """Check remaining OpenRouter credits for this specific API key."""
    url = f"{base_url}/key"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        try:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"HTTP {e.response.status_code}",
                "detail": e.response.text[:200]
            }
        except Exception as e:
            return {"error": str(e)}


def print_remaining_credits(api_key: str, base_url: str = BASE_URL) -> None:
    """Print remaining OpenRouter credits in a formatted way."""
    credits_data = check_credits(api_key, base_url)
    if "error" in credits_data:
        print("⚠️  Error checking credits:", credits_data)
    else:
        data = credits_data.get("data", {})
        
        # Key-specific information
        limit = data.get("limit", 0)  # Credit limit set for this key
        usage = data.get("usage", 0)  # Usage by this key
        remaining = limit - usage if limit else "No limit set"
        
        print(f"💳 API Key Credit Balance:")
        print(f"   Key limit:    ${limit:.2f}" if isinstance(limit, (int, float)) else f"   Key limit:    {limit}")
        print(f"   Key usage:    ${usage:.2f}")
        print(f"   Remaining:    ${remaining:.2f}" if isinstance(remaining, (int, float)) else f"   Remaining:    {remaining}")


def list_models(api_key: str, base_url: str = BASE_URL, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch available models from OpenRouter."""
    url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        try:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])[:limit]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []


def chat_completion(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    base_url: str = BASE_URL,
    temperature: float = 0.7,
    max_tokens: int = 500,
    response_format: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Make a chat completion request to OpenRouter.
    
    Args:
        api_key: OpenRouter API key
        model: Model identifier
        messages: List of message dicts with 'role' and 'content'
        base_url: Base URL for OpenRouter API
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        response_format: Optional dict for structured output, e.g. {"type": "json_object"}
    
    Returns:
        Dict with keys: model, content, error (if any), usage (token counts)
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    
    with httpx.Client(timeout=60) as client:
        try:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            usage = data.get("usage", {})
            
            # If JSON mode was requested, try to parse the content automatically
            parsed_content = None
            if response_format and response_format.get("type") == "json_object":
                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    # If parsing fails, parsed_content stays None
                    # The raw content is still available
                    pass
            
            return {
                "model": model,
                "content": content,  # Always return raw string content
                "parsed_content": parsed_content,  # Parsed JSON if response_format was json_object
                "usage": usage,
                "error": None
            }
            
        except httpx.HTTPStatusError as e:
            return {
                "model": model,
                "content": "",
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "usage": {}
            }
        except Exception as e:
            return {
                "model": model,
                "content": "",
                "error": str(e),
                "usage": {}
            }


def safe_chat(
    api_key: str,
    model: str,
    prompt: str,
    base_url: str = BASE_URL,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Wrapper with retry logic and graceful degradation.
    
    Args:
        api_key: OpenRouter API key
        model: Model identifier
        prompt: User prompt text
        base_url: Base URL for OpenRouter API
        max_retries: Maximum number of retry attempts
    
    Returns:
        Dict with keys: model, content, error (if any), usage (token counts)
    """
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        result = chat_completion(api_key, model, messages, base_url, temperature=0.7)
        
        if not result["error"]:
            return result
        
        print(f"  Attempt {attempt + 1} failed: {result['error']}")
        
        if attempt < max_retries - 1:
            print(f"  Retrying...")
    
    return result  # Return last failed result


def display_comparison(results_df, prompt_name: str) -> None:
    """Display responses from all models for a given prompt."""
    import pandas as pd
    
    subset = results_df[results_df["prompt"] == prompt_name]
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt_name}")
    print(f"{'='*70}\n")
    
    for _, row in subset.iterrows():
        print(f"[{row['model_key'].upper()}] ({row['model_id']})")
        print("-" * 70)
        
        if row["error"]:
            print(f"❌ Error: {row['error']}\n")
        else:
            print(row["content"])
            print()
            usage = row["usage"]
            if isinstance(usage, dict) and usage:
                print(f"Tokens: {usage.get('total_tokens', 'N/A')} total\n")
        
        print()
