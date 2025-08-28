#!/usr/bin/env python3
"""
LLM Latency Test Suite
Comprehensive testing for blog post: "End-to-End LLM Inference Latency Analysis"

This script tests different prompt sizes and provides detailed timing breakdowns
to identify where latency bottlenecks occur in the inference stack.
"""

import requests
import time
import json
import statistics
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import csv

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
HEALTH_ENDPOINT = f"{BASE_URL}/health"

print(f"Using base URL: {BASE_URL}")

@dataclass
class TestResult:
    test_name: str
    prompt_length: int
    prompt_tokens: int
    max_tokens: int
    total_time_ms: float
    network_time_ms: float
    inference_time_ms: float
    response_length: int
    response_tokens: int
    timestamp: str
    success: bool
    error: str = ""

class LatencyTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []
        self.session = requests.Session()
        
    def test_health(self) -> bool:
        try:
            start_time = time.time()
            response = self.session.get(HEALTH_ENDPOINT, timeout=30)
            network_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                print(f"Health check passed - Network latency: {network_time:.2f}ms")
                return True
            else:
                print(f"Health check failed - Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"Health check error: {e}")
            return False
    
    def run_single_test(self, test_name: str, prompt: str, max_tokens: int) -> TestResult:
        print(f"\nRunning test: {test_name}")
        print(f"   Prompt length: {len(prompt)} chars")
        print(f"   Max tokens: {max_tokens}")
        
        payload = {
            "model": "distilbert/distilgpt2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                API_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            total_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                inference_time = data.get("total_inference_latency", 0)
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                network_time = total_time - inference_time
                
                result = TestResult(
                    test_name=test_name,
                    prompt_length=len(prompt),
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    total_time_ms=total_time,
                    network_time_ms=network_time,
                    inference_time_ms=inference_time,
                    response_length=len(response_text),
                    response_tokens=completion_tokens,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
                print(f"   Success!")
                print(f"   Total time: {total_time:.2f}ms")
                print(f"   Network time: {network_time:.2f}ms")
                print(f"   Inference time: {inference_time:.2f}ms")
                print(f"   Response: {len(response_text)} chars, {completion_tokens} tokens")
                
                return result
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"   Failed: {error_msg}")
                return TestResult(
                    test_name=test_name,
                    prompt_length=len(prompt),
                    prompt_tokens=0,
                    max_tokens=max_tokens,
                    total_time_ms=0,
                    network_time_ms=0,
                    inference_time_ms=0,
                    response_length=0,
                    response_tokens=0,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = str(e)
            print(f"   Error: {error_msg}")
            return TestResult(
                test_name=test_name,
                prompt_length=len(prompt),
                prompt_tokens=0,
                max_tokens=max_tokens,
                total_time_ms=0,
                network_time_ms=0,
                inference_time_ms=0,
                response_length=0,
                response_tokens=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=error_msg
            )
    
    def run_comprehensive_tests(self):
        print("Starting Comprehensive LLM Latency Test Suite")
        print("=" * 60)
        
        prompt1 = "Hi"
        self.results.append(self.run_single_test("Minimal Prompt", prompt1, 20))
        
        prompt2 = "What is machine learning?"
        self.results.append(self.run_single_test("Short Prompt", prompt2, 30))
        
        prompt3 = "Explain the difference between supervised and unsupervised learning in machine learning, with examples of each type."
        self.results.append(self.run_single_test("Medium Prompt", prompt3, 50))
        
        prompt4 = """Write a comprehensive explanation of deep learning neural networks, including:
1. What are neural networks and how do they work?
2. What is the difference between shallow and deep networks?
3. How does backpropagation work?
4. What are common activation functions and why are they important?
5. What are the challenges of training deep networks?

Please provide detailed examples and practical applications."""
        self.results.append(self.run_single_test("Long Prompt", prompt4, 100))
        
        prompt5 = """You are an expert machine learning engineer with 15 years of experience. 
Please write a comprehensive guide to building production-ready machine learning systems that covers:

1. Data Pipeline Design
   - Data collection strategies
   - Data validation and quality checks
   - Feature engineering best practices
   - Data versioning and lineage

2. Model Development
   - Model selection criteria
   - Hyperparameter optimization techniques
   - Cross-validation strategies
   - Model interpretability methods

3. Training Infrastructure
   - Distributed training approaches
   - Resource management and scheduling
   - Experiment tracking and reproducibility
   - Cost optimization strategies

4. Model Deployment
   - Model serving architectures
   - A/B testing frameworks
   - Monitoring and alerting systems
   - Model performance tracking

5. Production Operations
   - CI/CD pipelines for ML
   - Model retraining strategies
   - Performance monitoring and optimization
   - Incident response and debugging

6. Ethical Considerations
   - Bias detection and mitigation
   - Privacy preservation techniques
   - Fairness metrics and evaluation
   - Responsible AI practices

Please provide specific examples, code snippets where relevant, and real-world case studies."""
        self.results.append(self.run_single_test("Very Long Prompt", prompt5, 150))
        
        base_prompt = "Explain how transformers work in natural language processing."
        self.results.append(self.run_single_test("Short Output (20 tokens)", base_prompt, 20))
        self.results.append(self.run_single_test("Medium Output (50 tokens)", base_prompt, 50))
        self.results.append(self.run_single_test("Long Output (100 tokens)", base_prompt, 100))
        
        print("\n" + "=" * 60)
        print("Test Suite Complete!")
    
    def generate_summary_report(self):
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("No successful tests to analyze")
            return
        
        print("\nLATENCY ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_times = [r.total_time_ms for r in successful_results]
        network_times = [r.network_time_ms for r in successful_results]
        inference_times = [r.inference_time_ms for r in successful_results]
        
        print(f"Total Tests: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(self.results) - len(successful_results)}")
        
        print(f"\nTIMING BREAKDOWN:")
        print(f"   Total Time:     {statistics.mean(total_times):.2f}ms ± {statistics.stdev(total_times):.2f}ms")
        print(f"   Network Time:   {statistics.mean(network_times):.2f}ms ± {statistics.stdev(network_times):.2f}ms")
        print(f"   Inference Time: {statistics.mean(inference_times):.2f}ms ± {statistics.stdev(inference_times):.2f}ms")
        
        print(f"\nLATENCY BY PROMPT LENGTH:")
        length_groups = {}
        for r in successful_results:
            length_range = f"{(r.prompt_length // 50) * 50}-{(r.prompt_length // 50) * 50 + 49}"
            if length_range not in length_groups:
                length_groups[length_range] = []
            length_groups[length_range].append(r.total_time_ms)
        
        for length_range, times in sorted(length_groups.items()):
            if len(times) > 1:
                print(f"   {length_range} chars: {statistics.mean(times):.2f}ms ± {statistics.stdev(times):.2f}ms")
            else:
                print(f"   {length_range} chars: {statistics.mean(times):.2f}ms (single sample)")
        
        print(f"\nLATENCY BY OUTPUT LENGTH:")
        output_groups = {}
        for r in successful_results:
            output_range = f"{(r.max_tokens // 25) * 25}-{(r.max_tokens // 25) * 25 + 24}"
            if output_range not in output_groups:
                output_groups[output_range] = []
            output_groups[output_range].append(r.total_time_ms)
        
        for output_range, times in sorted(output_groups.items()):
            if len(times) > 1:
                print(f"   {output_range} tokens: {statistics.mean(times):.2f}ms ± {statistics.stdev(times):.2f}ms")
            else:
                print(f"   {output_range} tokens: {statistics.mean(times):.2f}ms (single sample)")
        
        print(f"\nBOTTLENECK ANALYSIS:")
        avg_network_pct = (statistics.mean(network_times) / statistics.mean(total_times)) * 100
        avg_inference_pct = (statistics.mean(inference_times) / statistics.mean(total_times)) * 100
        
        print(f"   Network overhead: {avg_network_pct:.1f}% of total time")
        print(f"   Inference: {avg_inference_pct:.1f}% of total time")
        
        if avg_inference_pct > 80:
            print("   Inference is the primary bottleneck")
        elif avg_network_pct > 30:
            print("   Network latency is significant")
        else:
            print("   Latency is well distributed")
    
    def save_results_to_csv(self, filename: str = "latency_test_results.csv"):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'test_name', 'prompt_length', 'prompt_tokens', 'max_tokens',
                'total_time_ms', 'network_time_ms', 'inference_time_ms',
                'response_length', 'response_tokens', 'timestamp', 'success', 'error'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    'test_name': result.test_name,
                    'prompt_length': result.prompt_length,
                    'prompt_tokens': result.prompt_tokens,
                    'max_tokens': result.max_tokens,
                    'total_time_ms': result.total_time_ms,
                    'network_time_ms': result.network_time_ms,
                    'inference_time_ms': result.inference_time_ms,
                    'response_length': result.response_length,
                    'response_tokens': result.response_tokens,
                    'timestamp': result.timestamp,
                    'success': result.success,
                    'error': result.error
                })
        
        print(f"\nResults saved to {filename}")
    
    def run_performance_analysis(self):
        print("\nPERFORMANCE ANALYSIS")
        print("=" * 60)
        
        print("Testing cold start performance...")
        cold_start_result = self.run_single_test("Cold Start Test", "Hello world", 20)
        
        print("\nTesting warm start performance...")
        warm_results = []
        for i in range(5):
            result = self.run_single_test(f"Warm Start {i+1}", "Hello world", 20)
            warm_results.append(result)
            time.sleep(1)
        
        if cold_start_result.success and all(r.success for r in warm_results):
            cold_time = cold_start_result.total_time_ms
            warm_times = [r.total_time_ms for r in warm_results]
            avg_warm_time = statistics.mean(warm_times)
            
            print(f"\nCOLD vs WARM START:")
            print(f"   Cold start: {cold_time:.2f}ms")
            print(f"   Average warm start: {avg_warm_time:.2f}ms")
            print(f"   Cold start penalty: {((cold_time / avg_warm_time) - 1) * 100:.1f}%")

def main():
    print("LLM Latency Test Suite for Blog Post Research")
    print("=" * 60)
    
    test_suite = LatencyTestSuite()
    
    if not test_suite.test_health():
        print("Service is not healthy. Please check your deployment.")
        return
    
    test_suite.run_comprehensive_tests()
    test_suite.generate_summary_report()
    test_suite.run_performance_analysis()
    test_suite.save_results_to_csv()
    
    print("\nTest suite complete! Check the CSV file for detailed data.")
    print("Use this data for your blog post: 'End-to-End LLM Inference Latency Analysis'")

if __name__ == "__main__":
    main()
