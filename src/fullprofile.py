#!/usr/bin/env python3
import requests
import time
import os
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from typing import List

def load_base_url():
    possible_paths = [
        'scripts/baseurl.txt',
        '../scripts/baseurl.txt',
        'baseurl.txt'
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

BASE_URL = load_base_url()
if not BASE_URL:
    exit(1)

API_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

print(f"Using base URL: {BASE_URL}")

@dataclass
class FullProfileTestResult:
    test_id: str
    test_name: str
    timestamp: str
    iteration: int
    prompt_length: int
    prompt_tokens: int
    prompt_complexity: str
    max_tokens: int
    actual_tokens: int
    response_length: int
    finish_reason: str
    total_time_ms: float
    network_time_ms: float
    inference_time_ms: float
    tokens_per_second: float
    chars_per_second: float
    ms_per_token: float
    success: bool
    time_to_first_token_ms: float | None = None
    prefill_time_ms: float | None = None
    generation_time_per_token_ms: float | None = None
    error: str = ""
    cold_start: bool = False
    batch_size: int = 1

class FullProfileLatencyAnalyzer:
    def __init__(self):
        self.results: List[FullProfileTestResult] = []
        self.session = requests.Session()
        self.test_counter = 0
        self.cold_start_detected = False
        
    def generate_test_id(self) -> str:
        self.test_counter += 1
        return f"test_{self.test_counter:04d}_{int(time.time())}"
    
    def analyze_prompt_complexity(self, prompt: str) -> str:
        if len(prompt) < 50:
            return "simple"
        elif len(prompt) < 200:
            return "medium"
        else:
            return "complex"
    
    def run_full_profile_test(self, prompt: str, max_tokens: int, 
                         iteration: int = 1, cold_start: bool = False) -> FullProfileTestResult:
        test_id = self.generate_test_id()
        test_name = f"{self.analyze_prompt_complexity(prompt)}_{len(prompt)}c_{max_tokens}t"
        
        print(f"\nTest {test_id}: {test_name}")
        print(f"   Prompt: {len(prompt)} chars, {max_tokens} max tokens")
        print(f"   Iteration: {iteration}, Cold start: {cold_start}")
        
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
                timeout=300
            )
            total_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                inference_time = data.get("total_inference_latency", 0)
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
                time_to_first_token = data.get("time_to_first_token")
                if time_to_first_token:
                    time_to_first_token_ms = time_to_first_token * 1000
                else:
                    time_to_first_token_ms = None
                network_time = total_time - inference_time
                tokens_per_second = (completion_tokens / (inference_time / 1000)) if inference_time > 0 else 0
                chars_per_second = (len(response_text) / (inference_time / 1000)) if inference_time > 0 else 0
                ms_per_token = inference_time / completion_tokens if completion_tokens > 0 else 0
                if time_to_first_token_ms is not None:
                    prefill_time_ms = time_to_first_token_ms
                    if completion_tokens > 1:
                        generation_time_per_token_ms = (inference_time - time_to_first_token_ms) / (completion_tokens - 1)
                    else:
                        generation_time_per_token_ms = 0
                else:
                    prefill_time_ms = None
                    generation_time_per_token_ms = None
                result = FullProfileTestResult(
                    test_id=test_id,
                    test_name=test_name,
                    timestamp=datetime.now().isoformat(),
                    iteration=iteration,
                    prompt_length=len(prompt),
                    prompt_tokens=prompt_tokens,
                    prompt_complexity=self.analyze_prompt_complexity(prompt),
                    max_tokens=max_tokens,
                    actual_tokens=completion_tokens,
                    response_length=len(response_text),
                    finish_reason=finish_reason,
                    total_time_ms=total_time,
                    network_time_ms=network_time,
                    inference_time_ms=inference_time,
                    time_to_first_token_ms=time_to_first_token_ms,
                    prefill_time_ms=prefill_time_ms,
                    generation_time_per_token_ms=generation_time_per_token_ms,
                    tokens_per_second=tokens_per_second,
                    chars_per_second=chars_per_second,
                    ms_per_token=ms_per_token,
                    success=True,
                    cold_start=cold_start,
                    batch_size=1
                )
                print(f"   Success!")
                print(f"   Total: {total_time:.2f}ms, Inference: {inference_time:.2f}ms")
                if time_to_first_token_ms:
                    print(f"   TTFT: {time_to_first_token_ms:.2f}ms")
                print(f"   Speed: {tokens_per_second:.2f} tokens/sec, {ms_per_token:.2f}ms/token")
                print(f"   Generated: {completion_tokens}/{max_tokens} tokens ({len(response_text)} chars)")
                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"   Failed: {error_msg}")
                return FullProfileTestResult(
                    test_id=test_id,
                    test_name=test_name,
                    timestamp=datetime.now().isoformat(),
                    iteration=iteration,
                    prompt_length=len(prompt),
                    prompt_tokens=0,
                    prompt_complexity=self.analyze_prompt_complexity(prompt),
                    max_tokens=max_tokens,
                    actual_tokens=0,
                    response_length=0,
                    finish_reason="error",
                    total_time_ms=0,
                    network_time_ms=0,
                    inference_time_ms=0,
                    time_to_first_token_ms=None,
                    prefill_time_ms=None,
                    generation_time_per_token_ms=None,
                    tokens_per_second=0,
                    chars_per_second=0,
                    ms_per_token=0,
                    success=False,
                    error=error_msg,
                    cold_start=cold_start,
                    batch_size=1
                )
        except Exception as e:
            error_msg = str(e)
            print(f"   Error: {error_msg}")
            return FullProfileTestResult(
                test_id=test_id,
                test_name=test_name,
                timestamp=datetime.now().isoformat(),
                iteration=iteration,
                prompt_length=len(prompt),
                prompt_tokens=0,
                prompt_complexity=self.analyze_prompt_complexity(prompt),
                max_tokens=max_tokens,
                actual_tokens=0,
                response_length=0,
                finish_reason="error",
                total_time_ms=0,
                network_time_ms=0,
                inference_time_ms=0,
                time_to_first_token_ms=None,
                prefill_time_ms=None,
                generation_time_per_token_ms=None,
                tokens_per_second=0,
                chars_per_second=0,
                ms_per_token=0,
                success=False,
                error=error_msg,
                cold_start=cold_start,
                batch_size=1
            )
    
    def run_input_length_analysis(self):
        print("\nINPUT LENGTH ANALYSIS")
        print("=" * 50)
        prompts = [
            ("A", 1),
            ("Hi there", 8),
            ("What is machine learning?", 25),
            ("Explain the difference between supervised and unsupervised learning in machine learning, with examples of each type.", 116),
            ("Write a comprehensive explanation of deep learning neural networks, including: 1. What are neural networks and how do they work? 2. What is the difference between shallow and deep networks? 3. How does backpropagation work? 4. What are common activation functions and why are they important? 5. What are the challenges of training deep networks? Please provide detailed examples and practical applications.", 407),
            ("You are an expert machine learning engineer with 15 years of experience. Please write a comprehensive guide to building production-ready machine learning systems that covers: 1. Data Pipeline Design - Data collection strategies, data validation and quality checks, feature engineering best practices, data versioning and lineage. 2. Model Development - Model selection criteria, hyperparameter optimization techniques, cross-validation strategies, model interpretability methods. 3. Training Infrastructure - Distributed training approaches, resource management and scheduling, experiment tracking and reproducibility, cost optimization strategies. 4. Model Deployment - Model serving architectures, A/B testing frameworks, monitoring and alerting systems, model performance tracking. 5. Production Operations - CI/CD pipelines for ML, model retraining strategies, performance monitoring and optimization, incident response and debugging. 6. Ethical Considerations - Bias detection and mitigation, privacy preservation techniques, fairness metrics and evaluation, responsible AI practices. Please provide specific examples, code snippets where relevant, and real-world case studies.", 1274)
        ]
        for prompt, length in prompts:
            for max_tokens in [20, 50, 100]:
                result = self.run_full_profile_test(prompt, max_tokens)
                self.results.append(result)
                time.sleep(2)
    
    def run_output_length_analysis(self):
        print("\nOUTPUT LENGTH ANALYSIS")
        print("=" * 50)
        base_prompt = "Explain how transformers work in natural language processing."
        for max_tokens in [1, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200]:
            result = self.run_full_profile_test(base_prompt, max_tokens)
            self.results.append(result)
            time.sleep(2)
    
    def run_cold_warm_analysis(self):
        print("\nCOLD vs WARM START ANALYSIS")
        print("=" * 50)
        test_prompt = "Hello, how are you today?"
        print("Testing cold start...")
        cold_result = self.run_full_profile_test(test_prompt, 20, cold_start=True)
        self.results.append(cold_result)
        self.cold_start_detected = True
        print("Testing warm starts...")
        for i in range(10):
            warm_result = self.run_full_profile_test(test_prompt, 20, iteration=i+1, cold_start=False)
            self.results.append(warm_result)
            time.sleep(1)
    
    def run_complexity_analysis(self):
        print("\nCOMPLEXITY ANALYSIS")
        print("=" * 50)
        simple_prompts = [
            "What is 2+2?",
            "Name a color.",
            "What is the capital of France?"
        ]
        medium_prompts = [
            "Explain photosynthesis in simple terms.",
            "What are the benefits of exercise?",
            "How do computers work?"
        ]
        complex_prompts = [
            "Compare and contrast the economic systems of capitalism and socialism, considering their historical development, theoretical foundations, practical implementations, and societal impacts.",
            "Analyze the relationship between climate change and economic inequality, discussing how environmental degradation disproportionately affects marginalized communities and proposing policy solutions that address both issues simultaneously.",
            "Evaluate the effectiveness of different educational approaches in preparing students for the 21st century workforce, considering technological changes, globalization, and evolving skill requirements."
        ]
        for prompt in simple_prompts + medium_prompts + complex_prompts:
            result = self.run_full_profile_test(prompt, 50)
            self.results.append(result)
            time.sleep(2)
    
    def run_batch_analysis(self):
        print("\nBATCH PROCESSING ANALYSIS")
        print("=" * 50)
        base_prompt = "Generate a short story about a robot."
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(self.run_full_profile_test, base_prompt, 30)
                futures.append(future)
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
    
    def generate_full_profile_report(self):
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("No successful tests to analyze")
            return
        print("\nFULL PROFILE LATENCY ANALYSIS REPORT")
        print("=" * 60)
        total_times = [r.total_time_ms for r in successful_results]
        network_times = [r.network_time_ms for r in successful_results]
        inference_times = [r.inference_time_ms for r in successful_results]
        tokens_per_second = [r.tokens_per_second for r in successful_results if r.tokens_per_second > 0]
        ms_per_token = [r.ms_per_token for r in successful_results if r.ms_per_token > 0]
        print(f"Total Tests: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(self.results) - len(successful_results)}")
        print(f"\nTIMING BREAKDOWN:")
        print(f"   Total Time:     {np.mean(total_times):.2f}ms ± {np.std(total_times):.2f}ms")
        print(f"   Network Time:   {np.mean(network_times):.2f}ms ± {np.std(network_times):.2f}ms")
        print(f"   Inference Time: {np.mean(inference_times):.2f}ms ± {np.std(inference_times):.2f}ms")
        if tokens_per_second:
            print(f"\nPERFORMANCE METRICS:")
            print(f"   Tokens/Second: {np.mean(tokens_per_second):.2f} ± {np.std(tokens_per_second):.2f}")
            print(f"   ms/Token:      {np.mean(ms_per_token):.2f} ± {np.std(ms_per_token):.2f}")
        ttft_times = [r.time_to_first_token_ms for r in successful_results if r.time_to_first_token_ms is not None]
        if ttft_times:
            print(f"\nTIME TO FIRST TOKEN (Measured Data):")
            print(f"   Average TTFT:  {np.mean(ttft_times):.2f}ms ± {np.std(ttft_times):.2f}ms")
            print(f"   Min TTFT:      {np.min(ttft_times):.2f}ms")
            print(f"   Max TTFT:      {np.max(ttft_times):.2f}ms")
            print(f"   Samples:       {len(ttft_times)}")
        else:
            print(f"\nTIME TO FIRST TOKEN: No measured data available")
        prefill_times = [r.prefill_time_ms for r in successful_results if r.prefill_time_ms is not None]
        if prefill_times:
            print(f"\nPREFILL/KV CACHE TIMING (Measured Data):")
            print(f"   Average Prefill: {np.mean(prefill_times):.2f}ms ± {np.std(prefill_times):.2f}ms")
            print(f"   Samples:         {len(prefill_times)}")
        else:
            print(f"\nPREFILL/KV CACHE TIMING: No measured data available")
        generation_times = [r.generation_time_per_token_ms for r in successful_results if r.generation_time_per_token_ms is not None and r.generation_time_per_token_ms > 0]
        if generation_times:
            print(f"\nPER-TOKEN GENERATION (Measured Data):")
            print(f"   Average per token: {np.mean(generation_times):.2f}ms ± {np.std(generation_times):.2f}ms")
            print(f"   Samples:           {len(generation_times)}")
        else:
            print(f"\nPER-TOKEN GENERATION: No measured data available")
        print(f"\nPERFORMANCE CURVE ANALYSIS:")
        if tokens_per_second:
            optimal_tps = np.max(tokens_per_second)
            worst_tps = np.min(tokens_per_second)
            optimal_scenario = [r for r in successful_results if r.tokens_per_second == optimal_tps][0]
            worst_scenario = [r for r in successful_results if r.tokens_per_second == worst_tps][0]
            print(f"   Peak Performance: {optimal_tps:.2f} tokens/sec")
            print(f"     Scenario: {optimal_scenario.prompt_length} chars, {optimal_scenario.max_tokens} tokens")
            print(f"   Worst Performance: {worst_tps:.2f} tokens/sec")
            print(f"     Scenario: {worst_scenario.prompt_length} chars, {worst_scenario.max_tokens} tokens")
            print(f"   Performance Range: {optimal_tps/worst_tps:.1f}x difference")
        print(f"\nCOMPLEXITY ANALYSIS:")
        for complexity in ["simple", "medium", "complex"]:
            complex_results = [r for r in successful_results if r.prompt_complexity == complexity]
            if complex_results:
                times = [r.total_time_ms for r in complex_results]
                print(f"   {complexity.capitalize()}: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms (n={len(times)})")
        print(f"\nOUTPUT LENGTH ANALYSIS:")
        length_groups = {}
        for r in successful_results:
            length_range = f"{(r.max_tokens // 25) * 25}-{(r.max_tokens // 25) * 25 + 24}"
            if length_range not in length_groups:
                length_groups[length_range] = []
            length_groups[length_range].append(r.total_time_ms)
        for length_range, times in sorted(length_groups.items()):
            if len(times) > 1:
                print(f"   {length_range} tokens: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms (n={len(times)})")
            else:
                print(f"   {length_range} tokens: {np.mean(times):.2f}ms (n=1)")
        cold_results = [r for r in successful_results if r.cold_start]
        if cold_results:
            print(f"\nCOLD vs WARM START:")
            cold_test = cold_results[0]
            warm_same_prompt = [r for r in successful_results if 
                               not r.cold_start and 
                               r.prompt_length == cold_test.prompt_length and 
                               r.max_tokens == cold_test.max_tokens]
            if warm_same_prompt:
                cold_time = cold_test.total_time_ms
                warm_times = [r.total_time_ms for r in warm_same_prompt]
                avg_warm = np.mean(warm_times)
                print(f"   Cold start (same prompt): {cold_time:.2f}ms")
                print(f"   Warm start (same prompt): {avg_warm:.2f}ms ± {np.std(warm_times):.2f}ms")
                print(f"   Warm samples: {len(warm_times)}")
                if cold_time > avg_warm:
                    penalty = ((cold_time / avg_warm) - 1) * 100
                    print(f"   Cold start penalty: +{penalty:.1f}% (cold is slower)")
                else:
                    penalty = ((avg_warm / cold_time) - 1) * 100
                    print(f"   Cold start advantage: +{penalty:.1f}% (cold is faster - unusual)")
            else:
                print(f"   Cold start: {cold_results[0].total_time_ms:.2f}ms")
                print(f"   No matching warm start tests for comparison")
        else:
            print(f"\nCOLD vs WARM START: No cold start data available")
        print(f"\nBOTTLENECK ANALYSIS:")
        avg_network_pct = (np.mean(network_times) / np.mean(total_times)) * 100
        avg_inference_pct = (np.mean(inference_times) / np.mean(total_times)) * 100
        print(f"   Network overhead: {avg_network_pct:.1f}% of total time")
        print(f"   Inference: {avg_inference_pct:.1f}% of total time")
        if avg_inference_pct > 80:
            print("   Inference is the primary bottleneck")
        elif avg_network_pct > 30:
            print("   Network latency is significant")
        else:
            print("   Latency is well distributed")
    
    def save_results_to_csv(self, filename: str = "full_profile_latency_results.csv"):
        data_paths = ['../data/', 'data/']
        data_dir = None
        for path in data_paths:
            if os.path.exists(path):
                data_dir = path
                break
        if not data_dir:
            data_dir = 'data/'
            os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = list(asdict(self.results[0]).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        print(f"\nResults saved to {filepath}")
    
    def create_full_profile_visualizations(self):
        try:
            successful_results = [r for r in self.results if r.success]
            if not successful_results:
                print("No successful results to visualize")
                return
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Full Profile LLM Latency Analysis - Performance Curves & Bottlenecks', fontsize=16)
            ax1 = axes[0, 0]
            max_tokens = [r.max_tokens for r in successful_results]
            tps_values = [r.tokens_per_second for r in successful_results if r.tokens_per_second > 0]
            tps_tokens = [r.max_tokens for r in successful_results if r.tokens_per_second > 0]
            if tps_values:
                ax1.scatter(tps_tokens, tps_values, alpha=0.7, s=50)
                ax1.set_xlabel('Output Length (tokens)')
                ax1.set_ylabel('Tokens per Second')
                ax1.set_title('Performance Curve: Speed vs Output Length')
                ax1.grid(True, alpha=0.3)
                z = np.polyfit(tps_tokens, tps_values, 1)
                p = np.poly1d(z)
                ax1.plot(tps_tokens, p(tps_tokens), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
                ax1.legend()
            ax2 = axes[0, 1]
            ax2.scatter(max_tokens, [r.total_time_ms for r in successful_results], alpha=0.7, s=50)
            ax2.set_xlabel('Output Length (tokens)')
            ax2.set_ylabel('Total Time (ms)')
            ax2.set_title('Latency vs Output Length\n(Performance Zones)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1000, color='g', linestyle='--', alpha=0.7, label='Fast (<1s)')
            ax2.axhline(y=5000, color='y', linestyle='--', alpha=0.7, label='Medium (1-5s)')
            ax2.axhline(y=10000, color='r', linestyle='--', alpha=0.7, label='Slow (>10s)')
            ax2.legend()
            ax3 = axes[0, 2]
            ttft_data = [r.time_to_first_token_ms for r in successful_results if r.time_to_first_token_ms is not None]
            if ttft_data:
                ax3.hist(ttft_data, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
                ax3.set_xlabel('Time to First Token (ms)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Time to First Token Distribution')
                ax3.grid(True, alpha=0.3)
                ax3.axvline(np.mean(ttft_data), color='red', linestyle='--', label=f'Mean: {np.mean(ttft_data):.0f}ms')
                ax3.legend()
            ax4 = axes[1, 0]
            gen_times = [r.generation_time_per_token_ms for r in successful_results if r.generation_time_per_token_ms is not None and r.generation_time_per_token_ms > 0]
            if gen_times:
                ax4.hist(gen_times, bins=15, alpha=0.7, edgecolor='black', color='lightgreen')
                ax4.set_xlabel('Generation Time per Token (ms)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Per-Token Generation Time')
                ax4.grid(True, alpha=0.3)
                ax4.axvline(np.mean(gen_times), color='red', linestyle='--', label=f'Mean: {np.mean(gen_times):.1f}ms')
                ax4.legend()
            ax5 = axes[1, 1]
            cold_times = [r.total_time_ms for r in successful_results if r.cold_start]
            warm_times = [r.total_time_ms for r in successful_results if not r.cold_start]
            if cold_times and warm_times:
                ax5.boxplot([cold_times, warm_times], labels=['Cold Start', 'Warm Start'])
                ax5.set_ylabel('Total Time (ms)')
                ax5.set_title('Cold vs Warm Start Performance')
                ax5.grid(True, alpha=0.3)
                cold_avg = np.mean(cold_times)
                warm_avg = np.mean(warm_times)
                if cold_avg > warm_avg:
                    penalty = ((cold_avg / warm_avg) - 1) * 100
                    label_text = f'Cold Start Penalty: +{penalty:.1f}%'
                    color = 'yellow'
                else:
                    penalty = ((warm_avg / cold_avg) - 1) * 100
                    label_text = f'Cold Start Advantage: +{penalty:.1f}%'
                    color = 'lightgreen'
                ax5.text(0.5, 0.95, label_text, 
                         transform=ax5.transAxes, ha='center', va='top',
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
            ax6 = axes[1, 2]
            heatmap_data = []
            input_ranges = ['0-50', '51-200', '201+']
            output_ranges = ['1-25', '26-50', '51-100', '101+']
            for input_range in input_ranges:
                row = []
                for output_range in output_ranges:
                    filtered = [r for r in successful_results 
                               if self._get_input_range(r.prompt_length) == input_range and
                                  self._get_output_range(r.max_tokens) == output_range]
                    if filtered:
                        avg_time = np.mean([r.total_time_ms for r in filtered])
                        row.append(avg_time)
                    else:
                        row.append(0)
                heatmap_data.append(row)
            if any(any(row) for row in heatmap_data):
                im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax6.set_xticks(range(len(output_ranges)))
                ax6.set_yticks(range(len(input_ranges)))
                ax6.set_xticklabels(output_ranges)
                ax6.set_yticklabels(input_ranges)
                ax6.set_xlabel('Output Length (tokens)')
                ax6.set_ylabel('Input Length (chars)')
                ax6.set_title('Performance Heatmap\n(Total Time in ms)')
                cbar = plt.colorbar(im, ax=ax6)
                cbar.set_label('Time (ms)')
                for i in range(len(input_ranges)):
                    for j in range(len(output_ranges)):
                        if heatmap_data[i][j] > 0:
                            ax6.text(j, i, f'{heatmap_data[i][j]:.0f}', 
                                    ha='center', va='center', color='black', fontweight='bold')
            plt.tight_layout()
            data_paths = ['../data/', 'data/']
            data_dir = None
            for path in data_paths:
                if os.path.exists(path):
                    data_dir = path
                    break
            if not data_dir:
                data_dir = 'data/'
                os.makedirs(data_dir, exist_ok=True)
            chart_path = os.path.join(data_dir, 'full_profile_latency_analysis.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"Full profile charts saved to '{chart_path}'")
        except ImportError:
            print("matplotlib not available - skipping visualizations")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def _get_input_range(self, length):
        if length <= 50:
            return '0-50'
        elif length <= 200:
            return '51-200'
        else:
            return '201+'
    
    def _get_output_range(self, tokens):
        if tokens <= 25:
            return '1-25'
        elif tokens <= 50:
            return '26-50'
        elif tokens <= 100:
            return '51-100'
        else:
            return '101+'

def main():
    analyzer = FullProfileLatencyAnalyzer()
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=30)
        if response.status_code == 200:
            print("Service is healthy")
        else:
            print("Service health check failed")
            return
    except Exception as e:
        print(f"Cannot connect to service: {e}")
        return
    print("\nStarting full profile comprehensive analysis...")
    analyzer.run_input_length_analysis()
    analyzer.run_output_length_analysis()
    analyzer.run_cold_warm_analysis()
    analyzer.run_complexity_analysis()
    analyzer.run_batch_analysis()
    analyzer.generate_full_profile_report()
    analyzer.save_results_to_csv()
    analyzer.create_full_profile_visualizations()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
