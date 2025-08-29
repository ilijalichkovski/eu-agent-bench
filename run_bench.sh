#!/bin/bash

AUTOMATIC_COMMIT=false # set this to false for testing; otherwise it will automatically commit and push the results
INJECTED_REGULATIONS=false # set this to true to run the benchmark with injected regulations
OUTPUT_DIR="results_qwen" # set this to the directory where the results will be saved

# Load .env file if it exists and check if the OPENROUTER_API_KEY is set
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # turn off automatic export
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set"
    exit 1
fi

# x-ai/grok-4 not for now, too verbose and expensive

# models=("qwen/qwen3-0.6b-04-28" "qwen/qwen3-1.7b" "qwen/qwen3-8b" "qwen/qwen3-14b" "qwen/qwen3-32b")
models=("google/gemini-2.5-flash")
datasets=("data_protection" "bias_discrimination" "competition" "consumer_protection" "copyright" "scientific_misconduct")

# Array to store background process PIDs
pids=()

echo "Starting parallel benchmark runs for ${#models[@]} models on ${#datasets[@]} datasets..."
echo "Total jobs: $((${#models[@]} * ${#datasets[@]}))"

# Launch all combinations in parallel
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Launching: $model on $dataset"
        # Run in background and capture PID
        if [ "$INJECTED_REGULATIONS" = true ]; then
            python run_benchmark.py --model "$model" --trials 10 --data_path "benchmark_data_injected/$dataset.json" --backend openrouter --temperature 0.7 --clean_run --injected_regulations --output_dir "$OUTPUT_DIR" &
        else
            python run_benchmark.py --model "$model" --trials 10 --data_path "benchmark_data_injected/$dataset.json" --backend openrouter --temperature 0.7 --clean_run --output_dir "$OUTPUT_DIR" &
        fi
        pids+=($!)
    done
done

echo "All jobs launched. Waiting for completion..."
echo "PIDs: ${pids[*]}"

# Wait for all background processes to complete
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: Process $pid exited with code $exit_code"
    fi
done

echo "All benchmark runs completed!"

# If automatic commit is enabled, commit all results at once
# if [ "$AUTOMATIC_COMMIT" = true ]; then
#     echo "Committing results..."
#     git add results/
#     git commit -m "Added parallel benchmark results for all models on all datasets"
#     git push
#     echo "Results committed and pushed!"
# fi

# echo "Benchmark suite finished!"