#!/usr/bin/env python3
"""
Script to continue benchmark runs from where they stopped for existing models,
and start qwen model fresh on all datasets - PARALLEL VERSION.
"""

import subprocess
import os
import json
import sys
from pathlib import Path

def get_existing_files_and_instances():
    """Get existing result files and their max instances for models to continue."""
    results_dir = Path('results')
    existing_runs = {}
    
    # Based on your terminal output, these are the progress points for models to CONTINUE
    # Using correct model names with prefixes from run_bench.sh
    progress_data = [
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324", "bias_discrimination", 13),
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324", "competition", 11), 
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324", "consumer_protection", 13),
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324", "copyright", 21),
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324", "data_protection", 16),
        ("deepseek/deepseek-chat-v3-0324", "deepseek-chat-v3-0324", "scientific_misconduct", 11),
        ("google/gemini-2.5-flash", "gemini-2.5-flash", "bias_discrimination", 30),
        ("google/gemini-2.5-flash", "gemini-2.5-flash", "competition", 60),
        ("google/gemini-2.5-flash", "gemini-2.5-flash", "consumer_protection", 86),
        ("google/gemini-2.5-flash", "gemini-2.5-flash", "copyright", 63),
        ("google/gemini-2.5-flash", "gemini-2.5-flash", "data_protection", 70),
        ("google/gemini-2.5-flash", "gemini-2.5-flash", "scientific_misconduct", 48),
        ("openai/gpt-4.1", "gpt-4.1", "bias_discrimination", 49),
        ("openai/gpt-4.1", "gpt-4.1", "competition", 48),
        ("openai/gpt-4.1", "gpt-4.1", "consumer_protection", 56),
        ("openai/gpt-4.1", "gpt-4.1", "copyright", 67),
        ("openai/gpt-4.1", "gpt-4.1", "data_protection", 61),
        ("openai/gpt-4.1", "gpt-4.1", "scientific_misconduct", 55),
        ("moonshotai/kimi-k2", "kimi-k2", "bias_discrimination", 9),
        ("moonshotai/kimi-k2", "kimi-k2", "competition", 15),
        ("moonshotai/kimi-k2", "kimi-k2", "consumer_protection", 20),
        ("moonshotai/kimi-k2", "kimi-k2", "copyright", 20),
        ("moonshotai/kimi-k2", "kimi-k2", "data_protection", 10),
        ("moonshotai/kimi-k2", "kimi-k2", "scientific_misconduct", 17),
    ]
    
    for full_model_name, file_model_name, dataset, max_instance in progress_data:
        # Find existing file
        pattern = f"clean_{file_model_name}_{dataset}_*.json"
        matching_files = list(results_dir.glob(pattern))
        
        if matching_files:
            # Use the most recent file
            existing_file = max(matching_files, key=lambda x: x.stat().st_mtime)
            # Calculate safe start instance
            start_instance = max(0, max_instance - 1)  # Start from max_instance, not max_instance + 1
            
            key = f"{full_model_name}_{dataset}"
            existing_runs[key] = {
                'model': full_model_name,  # Use full model name for API calls
                'dataset': dataset,
                'existing_file': str(existing_file),
                'start_instance': start_instance,
                'is_continuation': True
            }
            print(f"Found {full_model_name} on {dataset}: {existing_file.name}, will continue from instance {start_instance}")
    
    return existing_runs

def get_qwen_fresh_runs():
    """Get qwen runs to start fresh."""
    datasets = ["bias_discrimination", "competition", "consumer_protection", "copyright", "data_protection", "scientific_misconduct"]
    qwen_model = "qwen/qwen3-30b-a3b-instruct-2507"
    
    qwen_runs = {}
    for dataset in datasets:
        key = f"{qwen_model.split('/')[-1]}_{dataset}"
        qwen_runs[key] = {
            'model': qwen_model,
            'dataset': dataset,
            'existing_file': None,
            'start_instance': 0,
            'is_continuation': False
        }
        print(f"Will start qwen fresh on {dataset}")
    
    return qwen_runs

def run_benchmark_jobs_parallel():
    """Run all benchmark jobs in parallel."""
    # Get continuation jobs for existing models
    existing_runs = get_existing_files_and_instances()
    
    # Get fresh qwen jobs
    qwen_runs = get_qwen_fresh_runs()
    
    # Combine all jobs
    all_jobs = {**existing_runs, **qwen_runs}
    
    print(f"\nTotal jobs to run: {len(all_jobs)}")
    print(f"- Continuation jobs: {len(existing_runs)}")
    print(f"- Fresh qwen jobs: {len(qwen_runs)}")
    
    processes = []
    
    for job_key, job_info in all_jobs.items():
        cmd = [
            sys.executable, "run_benchmark.py",
            "--model", job_info['model'],
            "--trials", "10",
            "--data_path", f"benchmark_data/{job_info['dataset']}.json",
            "--backend", "openrouter",
            "--temperature", "0.7",
            "--clean_run",
            "--start_instance", str(job_info['start_instance'])
        ]
        
        # Add existing output path if this is a continuation
        if job_info['is_continuation']:
            cmd.extend(["--existing_output_path", job_info['existing_file']])
        
        if job_info['is_continuation']:
            print(f"Launching CONTINUATION: {job_info['model']} on {job_info['dataset']} from instance {job_info['start_instance']}")
        else:
            print(f"Launching FRESH: {job_info['model']} on {job_info['dataset']}")
        
        # Run in background and capture PID
        process = subprocess.Popen(cmd)
        processes.append((process, job_info['model'], job_info['dataset']))
    
    print(f"\nAll {len(processes)} jobs launched in parallel. Waiting for completion...")
    print("PIDs:", [p[0].pid for p in processes])
    
    # Wait for all background processes to complete
    for process, model, dataset in processes:
        process.wait()
        exit_code = process.returncode
        if exit_code == 0:
            print(f"✓ {model} on {dataset} completed successfully")
        else:
            print(f"✗ {model} on {dataset} failed with exit code {exit_code}")

if __name__ == "__main__":
    run_benchmark_jobs_parallel() 