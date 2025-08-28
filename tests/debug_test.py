#!/usr/bin/env python3
"""
Debug Test for LLM Latency Anomalies
Investigating why some responses seem too fast
"""

import requests
import time
import json

def load_base_url():
    """Load base URL from baseurl.txt file"""
    # Try multiple possible paths
    possible_paths = [
        'scripts/baseurl.txt',      # When run from root
        '../scripts/baseurl.txt',   # When run from tests/
        'baseurl.txt'               # Legacy fallback
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                base_url = f.read().strip()
                if not base_url.startswith('http'):
                    base_url = f"http://{base_url}"
                return base_url
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    
    print("Error: baseurl.txt not found in any expected location.")
    print("Please create scripts/baseurl.txt with your Hathora service URL")
    print("Example: echo 'your-service.edge.hathora.dev:port' > scripts/baseurl.txt")
    return None

# Load configuration dynamically
BASE_URL = load_base_url()
if not BASE_URL:
    exit(1)

API_ENDPOINT = f"{BASE_URL}/v1/chat/completions"

print(f"Using base URL: {BASE_URL}")

def test_with_details(prompt: str, max_tokens: int):
    """Test with detailed response analysis"""
    print(f"\nTesting: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Max tokens: {max_tokens}")
    
    payload = {
        "model": "distilbert/distilgpt2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        total_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract all the details
            inference_time = data.get("total_inference_latency", 0)
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
            total_tokens = data.get("usage", {}).get("total_tokens", 0)
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
            
            print(f"   Success!")
            print(f"   Total time: {total_time:.2f}ms")
            print(f"   Inference time: {inference_time:.2f}ms")
            print(f"   Response: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
            print(f"   Tokens: {prompt_tokens} + {completion_tokens} = {total_tokens}")
            print(f"   Finish reason: {finish_reason}")
            print(f"   Response length: {len(response_text)} chars")
            
            # Check for anomalies
            if completion_tokens < max_tokens and finish_reason == "stop":
                print(f"   ANOMALY: Generated {completion_tokens}/{max_tokens} tokens but finished early!")
            
            if total_time < 1000 and completion_tokens > 10:
                print(f"   ANOMALY: Very fast generation ({total_time:.2f}ms) for {completion_tokens} tokens!")
                
        else:
            print(f"   Failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   Error: {e}")

def main():
    print("Debug Test for LLM Latency Anomalies")
    print("=" * 50)
    
    # Test the suspicious case
    long_prompt = """Write a comprehensive explanation of deep learning neural networks, including:
1. What are neural networks and how do they work?
2. What is the difference between shallow and deep networks?
3. How does backpropagation work?
4. What are common activation functions and why are they important?
5. What are the challenges of training deep networks?

Please provide detailed examples and practical applications."""
    
    test_with_details(long_prompt, 100)
    
    # Test with a simple prompt to compare
    simple_prompt = "Explain neural networks in detail."
    test_with_details(simple_prompt, 100)
    
    # Test with minimal prompt
    minimal_prompt = "Hi"
    test_with_details(minimal_prompt, 20)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
