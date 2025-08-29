import os
import requests
import json
import openai
import dotenv
import argparse
from utils import parse_tool_calls, get_unsafe_args, load_from_json, check_required_function_called
from datetime import datetime
import time

"""
Some todos to implement during hackathon to use good statistical practices
TODO: Use the CLT (see formula paper and doublecheck) through reporting SE besides the mean, +- 1.96 SE creates 95% confidence interval.
TODO: We will have clustered questions! That is, the above assumes i.i.d, which is not the case, our questions are synthetically generated from the same category.
      Thus, we should adjust for this by using the SE_clustered, this will likely broaden the confidence interval (as it should).
TODO: Number of samples should simply be as high as possible :) Reduces variance and increases statistical power
TODO: We can also reduce variance *within* questions by rerunning models
"""

dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4.1-mini")
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--data_path", type=str, default="diverse_scenarios.json")
parser.add_argument("--backend", type=str, default="openai")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--clean_run", action="store_true")
parser.add_argument("--start_instance", type=int, default=0)
parser.add_argument("--existing_output_path", type=str, default=None)
parser.add_argument("--injected_regulations", action="store_true")
args = parser.parse_args()

def run_experiment(system_prompt, user_request, rubric, necessary_arg, model, backend, num_runs=3, temperature=0.7, injected_regulations=False):
    """Run the model multiple times and collect results"""
    
    print(f"Running {num_runs} experiments with temperature={temperature}...")
    print("=" * 60)
    
    results = []
    for i in range(num_runs):
        retry_count = 0
        max_retries = 3
        response_text = None
        finish_reason = None
        
        while retry_count < max_retries:
            try:
                # i left this for backward compatibility and easy testing
                if backend == "openai":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_request},
                    ]
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    response = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=8000  # Increased to match openrouter
                    )
                    
                    response_text = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                
                # i prefer openrouter so we can easily switch models
                elif backend == "openrouter":
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {
                    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                    }
                    payload = {
                    "model": model,
                    "messages": [
                        {
                        "role": "system",
                        "content": system_prompt
                        },
                        {
                        "role": "user",
                        "content": user_request
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens": 32000  # Increased from 2000 to allow longer responses
                    }
                    response = requests.post(url, headers=headers, json=payload)
                    response_json = response.json()
                    
                    # Debug: Print the response if there's an issue
                    if 'choices' not in response_json:
                        print(f"API Response Status: {response.status_code}")
                        print(f"API Response: {json.dumps(response_json, indent=2)}")
                        raise ValueError(f"API returned error: {response_json}")
                    
                    response_text = response_json['choices'][0]['message']['content']
                    finish_reason = response_json['choices'][0].get('finish_reason', 'unknown')

                # Debug info
                response_length = len(response_text) if response_text else 0
                print(f"DEBUG: Attempt {retry_count + 1}: Response length: {response_length} chars, finish_reason: {finish_reason}")
                
                # Special debugging for empty responses
                if response_length == 0:
                    print("DEBUG: Empty response detected, logging full API response:")
                    if backend == "openrouter":
                        print(f"Full OpenRouter response: {json.dumps(response_json, indent=2)}")
                    elif backend == "openai":
                        print(f"Full OpenAI response: {response}")
                
                # Check for problematic responses that should be retried
                needs_retry = False
                if finish_reason == 'error':
                    print(f"API returned error finish_reason, retrying... (attempt {retry_count + 1}/{max_retries})")
                    needs_retry = True
                elif not response_text or len(response_text.strip()) == 0:
                    print(f"Empty response received, retrying... (attempt {retry_count + 1}/{max_retries})")
                    needs_retry = True
                elif '<think>' in response_text and '</think>' not in response_text and not response_text.strip().endswith('</answer>'):
                    print(f"Truncated response detected (unclosed <think> tag), retrying... (attempt {retry_count + 1}/{max_retries})")
                    needs_retry = True
                elif response_text and len(response_text) < 100 and not any(tag in response_text for tag in ['</think>', '</tool>', '</answer>']):
                    print(f"Suspiciously short response without completion tags, retrying... (attempt {retry_count + 1}/{max_retries})")
                    needs_retry = True
                
                if needs_retry and retry_count < max_retries - 1:
                    retry_count += 1
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    # Either response is good or we've exhausted retries
                    break
                    
            except Exception as e:
                print(f"Error in run {i+1}, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in 2 seconds... (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    # All retries failed
                    results.append({
                        'run': i+1,
                        'response': None,
                        'tool_calls': [],
                        'error': str(e),
                        'num_unsafe_args': 0,
                        'called_required_func': False,
                        'score': 0
                    })
                    break
        
        # If we got here with a bad response after exhausting retries
        if response_text is None or (retry_count >= max_retries - 1 and (finish_reason == 'error' or len(response_text.strip()) == 0)):
            print(f"Exhausted retries for run {i+1} due to API issues")
            results.append({
                'run': i+1,
                'response': response_text if response_text else None,
                'tool_calls': [],
                'error': f"API issue: finish_reason={finish_reason}, response_length={len(response_text) if response_text else 0}",
                'num_unsafe_args': 0,
                'called_required_func': False,
                'score': 0
            })
            continue

        # Debug warnings for final response
        if finish_reason == 'length':
            print(f"WARNING: Response may be truncated due to token limit (finish_reason: {finish_reason})")
        elif finish_reason != 'stop':
            print(f"INFO: Unusual finish_reason: {finish_reason}")

        # process the tool calls and calculate score
        tool_calls = parse_tool_calls(response_text)
        unsafe_args = get_unsafe_args(tool_calls, rubric)
        called_required_func = check_required_function_called(tool_calls, necessary_arg)
        score = 0 if len(unsafe_args) > 0 else 1
        
        # printing tool calls here
        print(f"\n--- Run {i+1} ---")
        if tool_calls:
            for j, tool_call in enumerate(tool_calls):
                print(f"Tool Call {j+1}: {json.dumps(tool_call, indent=2)}")
        else:
            print("No tool calls found")
            print(f"Raw response: {response_text[:200]}...")
            # Debug: Check if response might be incomplete
            if not response_text.strip().endswith('</answer>') and not response_text.strip().endswith('</tool>'):
                print("DEBUG: Response doesn't end with expected tags - might be truncated")
            if '<think>' in response_text and '</think>' not in response_text:
                print("DEBUG: Found unclosed <think> tag - likely truncated")
            if '<tool>' in response_text and '</tool>' not in response_text:
                print("DEBUG: Found unclosed <tool> tag - likely truncated")
            
            # Show more of the response for debugging
            if len(response_text) > 500:
                print(f"DEBUG: Full response ends with: ...{response_text[-100:]}")
            else:
                print(f"DEBUG: Full response: {response_text}")
        
        results.append({
            'run': i+1,
            'response': response_text,
            'tool_calls': tool_calls,
            'unsafe_args': unsafe_args,
            'num_unsafe_args': len(unsafe_args),
            'called_required_func': called_required_func,
            'score': score
        })
        time.sleep(1)
    
    return results

def run_benchmark(data_path, model, backend, trials, temperature, output_dir, clean_run, start_instance=0, existing_output_path=None, injected_regulations=False):
    """Runs the full benchmark on the dataset"""

    dataset = load_from_json(data_path)

    # set output path
    model_name = model.split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    if existing_output_path and os.path.exists(existing_output_path):
        # Continue from existing file
        output_path = existing_output_path
        print(f"Continuing from existing file: {output_path}")
        # Load existing results
        with open(output_path, "r") as f:
            scores = json.load(f)
        # Update model info to reflect the new model being used
        scores["metadata"]["model"] = model
        scores["metadata"]["backend"] = backend
        scores["metadata"]["temperature"] = temperature
        print(f"Loaded existing results with {len(scores['results'])} completed instances")
        print(f"Starting from instance {start_instance}")
    else:
        # Create new file
        if clean_run:
            output_path = f"{output_dir}/clean_{model_name}_{data_path.split('/')[-1].split('.')[0]}_{timestamp}_{'injected_regulations' if injected_regulations else ''}.json"
        else:
            output_path = f"{output_dir}/scores_{model_name}_{data_path.split('/')[-1].split('.')[0]}_{timestamp}_{'injected_regulations' if injected_regulations else ''}.json"

        # Initialize results structure
        scores = {
            "metadata": {
                "model": model,
                "backend": backend,
                "trials": trials,
                "temperature": temperature,
                "data_path": data_path,
                "output_path": output_path,
                "total_requests": len(dataset["user_requests"]),
                "time_taken": 0,
            },
            "results": []
        }
        
        # Save initial empty structure
        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)

    start_time = time.time()
    
    # Skip to start_instance and continue from there
    for i, request in enumerate(dataset["user_requests"][start_instance:], start=start_instance):
        print(f"\n=== Processing request {i+1}/{len(dataset['user_requests'])} ===")
        results = run_experiment(dataset["system_prompt"], request["prompt"], request["rubric"], request["necessary_arg"], model, backend, num_runs=trials, temperature=temperature, injected_regulations=injected_regulations)
        avg_score = sum(result['score'] for result in results) / len(results)
        
        # Add detailed results for this instance
        instance_result = {
            "instance": i,
            "prompt": request["prompt"],
            "necessary_arg": request["necessary_arg"],
            "avg_score": avg_score,
            "detailed_runs": results
        }
        
        scores["results"].append(instance_result)

        current_time = time.time()
        scores["metadata"]["time_taken"] = (current_time - start_time) / 60 # in minutes

        # Save results after each experiment
        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)
        
        print(f"Instance {i+1} completed. Average score: {avg_score:.3f}")
        print(f"Results saved to {output_path}")

    # Return simplified scores for backward compatibility
    simple_scores = {
        "instance": [r["instance"] for r in scores["results"]], 
        "avg_score": [r["avg_score"] for r in scores["results"]]
    }
    return simple_scores

def visualize_scores(scores_path):
    """Visualize the scores"""
    scores = load_from_json(scores_path)
    print(scores)

if __name__ == "__main__":
    run_benchmark(args.data_path, args.model, args.backend, args.trials, args.temperature, args.output_dir, args.clean_run, args.start_instance, args.existing_output_path, args.injected_regulations)