# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
import json
import numpy as np
import os
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %%
def load_all_results(results_dir="results"):
    """Load all result files excluding Claude models"""
    results = {}
    
    # Define the expected categories
    categories = [
        "bias_discrimination",
        "competition", 
        "consumer_protection",
        "copyright",
        "data_protection", 
        "scientific_misconduct"
    ]
    
    # Define the expected models
    models = [
        "deepseek-chat-v3-0324",
        "gemini-2.5-flash", 
        "gpt-4.1",
        "kimi-k2",
        "qwen3-0.6b-04-28",
        "qwen3-1.7b",
        "qwen3-8b",
        "qwen3-14b",
        #"qwen3-30b-a3b-instruct-2507",
        "qwen3-32b"
    ]
    
    for filename in os.listdir(results_dir):
        if filename.startswith("clean_") and filename.endswith(".json"):
            # Skip Claude results as requested
            if "claude" in filename:
                continue
            # Skip injected results in this function (handled separately)
            if "injected" in filename:
                continue
                
            filepath = os.path.join(results_dir, filename)
            
            # Parse model and category from filename
            # Remove prefix and suffix
            base_name = filename.replace("clean_", "").replace(".json", "")
            
            # Find model and category
            model = None
            category = None
            
            for m in models:
                if base_name.startswith(m + "_"):
                    model = m
                    # Extract category part
                    remaining = base_name[len(m) + 1:]  # +1 for the underscore
                    # Find category by checking if remaining starts with any category
                    for c in categories:
                        if remaining.startswith(c + "_"):  # category followed by timestamp
                            category = c
                            break
                    break
            
            if model is None or category is None:
                print(f"Could not parse filename: {filename}")
                print(f"  Base name: {base_name}")
                print(f"  Model found: {model}")
                print(f"  Category found: {category}")
                continue
            
            print(f"Loading: {model} - {category}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if model not in results:
                results[model] = {}
            results[model][category] = data
            
    return results

# %%
def load_injected_vs_normal_results(results_dir="results"):
    """Load both normal and injected regulation results for comparison"""
    results = {
        'normal': {},
        'injected': {}
    }
    
    # Define the expected categories
    categories = [
        "bias_discrimination",
        "competition", 
        "consumer_protection",
        "copyright",
        "data_protection", 
        "scientific_misconduct"
    ]
    
    # Focus on gemini model for injection comparison
    target_model = "gemini-2.5-flash"
    
    for filename in os.listdir(results_dir):
        if not filename.startswith("clean_") or not filename.endswith(".json"):
            continue
        if "claude" in filename:
            continue
        if target_model not in filename:
            continue
            
        filepath = os.path.join(results_dir, filename)
        
        # Determine if this is normal or injected
        is_injected = "injected" in filename
        condition = 'injected' if is_injected else 'normal'
        
        # Parse model and category from filename
        base_name = filename.replace("clean_", "").replace(".json", "")
        
        # For injected files, remove the injected suffix
        if is_injected:
            # Remove various possible injected suffixes
            base_name = base_name.replace("_injected_regulations", "").replace("_injected", "")
        
        # Find category
        category = None
        if base_name.startswith(target_model + "_"):
            remaining = base_name[len(target_model) + 1:]  # +1 for the underscore
            for c in categories:
                if remaining.startswith(c + "_"):  # category followed by timestamp
                    category = c
                    break
        
        if category is None:
            print(f"Could not parse category from filename: {filename}")
            print(f"  Base name: {base_name}")
            continue
        
        print(f"Loading {condition}: {target_model} - {category}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if target_model not in results[condition]:
            results[condition][target_model] = {}
        results[condition][target_model][category] = data
    
    return results

# %%
def analyze_injection_comparison(clip_trials=True):
    """Analyze comparison between normal and injected regulation conditions"""
    
    print("=== LOADING NORMAL VS INJECTED RESULTS ===")
    
    # Load injected vs normal results
    comparison_data = load_injected_vs_normal_results()
    
    # Check what we actually loaded
    print(f"Normal results loaded: {list(comparison_data['normal'].keys())}")
    print(f"Injected results loaded: {list(comparison_data['injected'].keys())}")
    
    # Initialize analysis structure
    analysis = {
        'metadata': {
            'description': 'Comparison analysis of normal vs injected regulation conditions',
            'target_model': 'gemini-2.5-flash',
            'clip_trials': clip_trials,
            'conditions': ['normal', 'injected']
        },
        'by_condition_category': {},
        'comparison_summary': {},
        'statistical_tests': {}
    }
    
    target_model = "gemini-2.5-flash"
    
    # Analyze each condition
    for condition in ['normal', 'injected']:
        if target_model in comparison_data[condition]:
            print(f"\nAnalyzing {condition} condition...")
            analysis['by_condition_category'][condition] = {}
            
            model_data = comparison_data[condition][target_model]
            for category, data in model_data.items():
                print(f"  - {category}")
                category_analysis = analyze_model_category(data, clip_trials=clip_trials)
                analysis['by_condition_category'][condition][category] = category_analysis
    
    # Create comparison summary
    if 'normal' in analysis['by_condition_category'] and 'injected' in analysis['by_condition_category']:
        normal_cats = analysis['by_condition_category']['normal']
        injected_cats = analysis['by_condition_category']['injected']
        
        # Find common categories
        common_categories = set(normal_cats.keys()) & set(injected_cats.keys())
        
        analysis['comparison_summary'] = {}
        for category in common_categories:
            normal_mean = normal_cats[category]['overall_mean']
            injected_mean = injected_cats[category]['overall_mean']
            
            normal_se = normal_cats[category]['se_clustered']
            injected_se = injected_cats[category]['se_clustered']
            
            # Calculate difference and combined SE
            mean_diff = injected_mean - normal_mean
            se_diff = np.sqrt(normal_se**2 + injected_se**2)  # Assuming independence
            
            # 95% CI for difference
            ci_diff_lower = mean_diff - 1.96 * se_diff
            ci_diff_upper = mean_diff + 1.96 * se_diff
            
            analysis['comparison_summary'][category] = {
                'normal_mean': float(normal_mean),
                'injected_mean': float(injected_mean),
                'difference': float(mean_diff),
                'difference_se': float(se_diff),
                'difference_ci_lower': float(ci_diff_lower),
                'difference_ci_upper': float(ci_diff_upper),
                'significant_at_5pct': bool(abs(mean_diff) > 1.96 * se_diff),
                'improvement_direction': 'injected_better' if mean_diff > 0 else 'normal_better' if mean_diff < 0 else 'no_difference'
            }
    
    return analysis

# %%
def calculate_clustered_se(scores, cluster_size=10):
    """
    Calculate clustered standard error using the correct statistical formula.
    
    Args:
        scores: Array of request-level scores (should be 100 scores per category)
        cluster_size: Number of requests per cluster (10 augmentations per original)
    
    Returns:
        se_clt: Standard CLT standard error
        se_clustered: Clustered standard error accounting for within-cluster correlation
        intracluster_corr: Estimated intracluster correlation coefficient
    """
    n = len(scores)
    
    # Handle edge cases
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, 0.0, 0.0
    
    overall_mean = np.mean(scores)
    
    # Standard CLT standard error: SE_CLT = sqrt(var(s) / n)
    variance = np.var(scores, ddof=1)
    if variance <= 0 or not np.isfinite(variance):
        return 0.0, 0.0, 0.0
    
    se_clt = np.sqrt(variance / n)
    
    # Calculate clustered standard error using proper formula
    num_clusters = n // cluster_size
    
    # Handle case where we don't have enough data for clustering analysis
    if num_clusters < 2:
        # Fall back to CLT standard error
        return se_clt, se_clt, 0.0
    
    if num_clusters * cluster_size != n:
        print(f"Warning: {n} scores cannot be evenly divided into clusters of size {cluster_size}")
        # Truncate to complete clusters only
        n_complete = num_clusters * cluster_size
        scores = scores[:n_complete]
        n = n_complete
        num_clusters = n // cluster_size
    
    # Calculate cluster means
    cluster_means = []
    for c in range(num_clusters):
        cluster_start = c * cluster_size
        cluster_end = (c + 1) * cluster_size
        cluster_scores = scores[cluster_start:cluster_end]
        if len(cluster_scores) > 0:
            cluster_means.append(np.mean(cluster_scores))
    
    cluster_means = np.array(cluster_means)
    
    # Handle edge cases in cluster means
    if len(cluster_means) < 2:
        return se_clt, se_clt, 0.0
    
    # Clustered standard error formula: SE_clustered = sqrt(var_between_clusters / num_clusters)
    # This accounts for the design effect of clustering
    var_between_clusters = np.var(cluster_means, ddof=1)
    
    # Handle numerical issues
    if not np.isfinite(var_between_clusters) or var_between_clusters <= 0:
        return se_clt, se_clt, 0.0
    
    se_clustered = np.sqrt(var_between_clusters / cluster_size)  # Divide by cluster size to get SE of individual obs
    
    # Calculate intracluster correlation coefficient (ICC) for diagnostic purposes
    # ICC = (MSB - MSW) / (MSB + (m-1)*MSW) where m = cluster_size
    mse_between = var_between_clusters * cluster_size  # Convert cluster mean variance to between-cluster MSE
    
    # Calculate within-cluster variance
    within_cluster_var = 0
    valid_clusters = 0
    for c in range(num_clusters):
        cluster_start = c * cluster_size
        cluster_end = (c + 1) * cluster_size
        cluster_scores = scores[cluster_start:cluster_end]
        if len(cluster_scores) > 1:  # Need at least 2 observations for variance
            cluster_var = np.var(cluster_scores, ddof=1)
            if np.isfinite(cluster_var):
                within_cluster_var += cluster_var
                valid_clusters += 1
    
    mse_within = within_cluster_var / valid_clusters if valid_clusters > 0 else 0
    
    # ICC calculation with numerical safety
    denominator = mse_between + (cluster_size - 1) * mse_within
    if denominator > 0 and np.isfinite(denominator) and np.isfinite(mse_between) and np.isfinite(mse_within):
        icc = (mse_between - mse_within) / denominator
        icc = max(0, min(1, icc))  # ICC should be between 0 and 1
    else:
        icc = 0
    
    # Design effect: DEFF = 1 + (m-1)*ICC
    design_effect = 1 + (cluster_size - 1) * icc
    
    # Alternative clustered SE formula using design effect
    se_clustered_alt = se_clt * np.sqrt(design_effect)
    
    # Handle numerical issues in final calculation
    if not np.isfinite(se_clustered):
        se_clustered = se_clt
    if not np.isfinite(se_clustered_alt):
        se_clustered_alt = se_clt
    
    # Use the larger of the two estimates for conservative inference
    se_clustered_final = max(se_clustered, se_clustered_alt)
    
    # Final safety check
    if not np.isfinite(se_clustered_final):
        se_clustered_final = se_clt
    
    return se_clt, se_clustered_final, icc

# %%
def calculate_aggregate_ci(category_means, category_ses, use_clustered=True):
    """
    Calculate confidence intervals for model aggregates (across categories).
    
    Args:
        category_means: List of category-level means for this model
        category_ses: List of category-level standard errors for this model
        use_clustered: Whether to use clustered or CLT standard errors
        
    Returns:
        overall_mean: Mean across categories
        se_aggregate: Standard error for the aggregate
        ci_lower, ci_upper: 95% confidence interval bounds
    """
    category_means = np.array(category_means)
    category_ses = np.array(category_ses)
    
    # Overall mean across categories
    overall_mean = np.mean(category_means)
    
    # Aggregate standard error (assuming independence across categories)
    # Correct formula: SE_aggregate = sqrt(sum(SE_i^2) / K^2) where K is number of categories
    K = len(category_means)
    se_aggregate = np.sqrt(np.sum(category_ses**2) / K**2)
    
    # 95% confidence interval
    ci_lower = overall_mean - 1.96 * se_aggregate
    ci_upper = overall_mean + 1.96 * se_aggregate
    
    return overall_mean, se_aggregate, ci_lower, ci_upper

# %%
def analyze_model_category(data, clip_trials=False):
    """Analyze a single model-category combination
    
    Args:
        data: The JSON data for this model-category
        clip_trials: If True, clip to exactly 100 requests and 10 trials each (1000 total)
                    If False, count all trials found (default)
    """
    results = data['results']
    
    # Validate structure
    expected_requests = 100
    expected_trials_per_request = 10
    expected_total_trials = expected_requests * expected_trials_per_request
    
    if clip_trials:
        # Limit to exactly 100 requests
        results = results[:expected_requests]
        print(f"  Clipping to {len(results)} requests (max {expected_requests})")
    
    if len(results) != expected_requests and not clip_trials:
        print(f"Warning: Expected {expected_requests} requests, found {len(results)}")
    
    # Extract all individual scores and necessary arg completion
    all_scores = []
    necessary_arg_counts = []
    request_level_scores = []
    
    for i, result in enumerate(results):
        # Request-level average score (already computed)
        request_avg = result['avg_score']
        request_level_scores.append(request_avg)
        
        # Validate trials per request
        detailed_runs = result['detailed_runs']
        
        if clip_trials:
            # Limit to exactly 10 trials per request
            detailed_runs = detailed_runs[:expected_trials_per_request]
        
        if len(detailed_runs) != expected_trials_per_request and not clip_trials:
            print(f"Warning: Request {i} has {len(detailed_runs)} trials, expected {expected_trials_per_request}")
        
        # Individual trial scores and necessary arg completion
        for run in detailed_runs:
            all_scores.append(run['score'])
            # Check if necessary arg was called
            necessary_arg_used = run.get('called_required_func', False)
            necessary_arg_counts.append(1 if necessary_arg_used else 0)
    
    # Validate total trials
    actual_total_trials = len(all_scores)
    if clip_trials:
        if actual_total_trials > expected_total_trials:
            print(f"  Clipped to {expected_total_trials} trials (was {actual_total_trials})")
        actual_total_trials = min(actual_total_trials, expected_total_trials)
    elif actual_total_trials != expected_total_trials:
        print(f"Warning: Expected {expected_total_trials} total trials, found {actual_total_trials}")
    
    # Calculate statistics
    request_level_scores = np.array(request_level_scores)
    all_scores = np.array(all_scores)
    necessary_arg_counts = np.array(necessary_arg_counts)
    
    # Overall mean (using request-level averages)
    overall_mean = np.mean(request_level_scores)
    
    # Calculate standard errors and confidence intervals
    se_clt, se_clustered, icc = calculate_clustered_se(request_level_scores, cluster_size=10)
    
    # 95% confidence intervals
    ci_clt = (overall_mean - 1.96 * se_clt, overall_mean + 1.96 * se_clt)
    ci_clustered = (overall_mean - 1.96 * se_clustered, overall_mean + 1.96 * se_clustered)
    
    # Necessary argument statistics (across all counted trials)
    necessary_arg_total = int(np.sum(necessary_arg_counts))
    necessary_arg_rate = float(np.mean(necessary_arg_counts))
    
    return {
        'overall_mean': float(overall_mean),
        'se_clt': float(se_clt),
        'se_clustered': float(se_clustered),
        'intracluster_correlation': float(icc),
        'ci_clt_lower': float(ci_clt[0]),
        'ci_clt_upper': float(ci_clt[1]),
        'ci_clustered_lower': float(ci_clustered[0]),
        'ci_clustered_upper': float(ci_clustered[1]),
        'necessary_arg_total': necessary_arg_total,  # Count out of actual trials
        'necessary_arg_rate': necessary_arg_rate,   # Rate: necessary_arg_total / actual_trials
        'total_trials': len(all_scores),            # Actual trials counted
        'total_requests': len(request_level_scores), # Actual requests counted
        'clipped': clip_trials,                      # Whether clipping was applied
        'clustering_diagnostics': {
            'icc': float(icc),
            'design_effect': float(1 + (10 - 1) * icc),
            'se_ratio': float(se_clustered / se_clt) if se_clt > 0 else 1.0
        }
    }

# %%
def analyze_model_category_filtered(data, clip_trials=False):
    """Analyze a single model-category combination, filtering for trials where necessary function was called
    
    Args:
        data: The JSON data for this model-category
        clip_trials: If True, clip to exactly 100 requests and 10 trials each (1000 total)
                    If False, count all trials found (default)
    """
    results = data['results']
    
    # Validate structure
    expected_requests = 100
    expected_trials_per_request = 10
    expected_total_trials = expected_requests * expected_trials_per_request
    
    if clip_trials:
        # Limit to exactly 100 requests
        results = results[:expected_requests]
        print(f"  Clipping to {len(results)} requests (max {expected_requests})")
    
    if len(results) != expected_requests and not clip_trials:
        print(f"Warning: Expected {expected_requests} requests, found {len(results)}")
    
    # Extract all individual scores and necessary arg completion - FILTERED VERSION
    all_scores_filtered = []
    necessary_arg_counts_filtered = []
    request_level_scores_filtered = []
    total_trials_before_filter = 0
    total_trials_after_filter = 0
    
    for i, result in enumerate(results):
        # Validate trials per request
        detailed_runs = result['detailed_runs']
        
        if clip_trials:
            # Limit to exactly 10 trials per request
            detailed_runs = detailed_runs[:expected_trials_per_request]
        
        if len(detailed_runs) != expected_trials_per_request and not clip_trials:
            print(f"Warning: Request {i} has {len(detailed_runs)} trials, expected {expected_trials_per_request}")
        
        # Filter trials where necessary function was called
        filtered_runs = []
        for run in detailed_runs:
            total_trials_before_filter += 1
            necessary_arg_used = run.get('called_required_func', False)
            if necessary_arg_used:  # Only include trials where necessary function was called
                filtered_runs.append(run)
                total_trials_after_filter += 1
                all_scores_filtered.append(run['score'])
                necessary_arg_counts_filtered.append(1)  # By definition, all filtered trials have this = 1
        
        # Calculate request-level average only from filtered trials
        if filtered_runs:
            request_avg_filtered = sum(run['score'] for run in filtered_runs) / len(filtered_runs)
            request_level_scores_filtered.append(request_avg_filtered)
        # If no trials in this request had the necessary function called, we skip this request
    
    # Validate we have enough data
    if len(request_level_scores_filtered) == 0:
        print(f"Warning: No trials found where necessary function was called!")
        return {
            'overall_mean': 0.0,
            'se_clt': 0.0,
            'se_clustered': 0.0,
            'intracluster_correlation': 0.0,
            'ci_clt_lower': 0.0,
            'ci_clt_upper': 0.0,
            'ci_clustered_lower': 0.0,
            'ci_clustered_upper': 0.0,
            'necessary_arg_total': 0,
            'necessary_arg_rate': 0.0,
            'total_trials_before_filter': total_trials_before_filter,
            'total_trials_after_filter': total_trials_after_filter,
            'total_requests_with_necessary_calls': 0,
            'filter_rate': 0.0,
            'clipped': clip_trials,
            'clustering_diagnostics': {
                'icc': 0.0,
                'design_effect': 1.0,
                'se_ratio': 1.0
            }
        }
    
    print(f"  Filtered trials: {total_trials_after_filter}/{total_trials_before_filter} ({100*total_trials_after_filter/total_trials_before_filter:.1f}%)")
    print(f"  Requests with necessary calls: {len(request_level_scores_filtered)}/{len(results)}")
    
    # Calculate statistics on filtered data
    request_level_scores_filtered = np.array(request_level_scores_filtered)
    all_scores_filtered = np.array(all_scores_filtered)
    necessary_arg_counts_filtered = np.array(necessary_arg_counts_filtered)
    
    # Overall mean (using request-level averages from filtered data)
    overall_mean = np.mean(request_level_scores_filtered)
    
    # Calculate standard errors and confidence intervals
    # Note: For filtered data, we may not have exact cluster sizes, so we adapt
    avg_cluster_size = len(all_scores_filtered) / len(request_level_scores_filtered) if len(request_level_scores_filtered) > 0 else 1
    se_clt, se_clustered, icc = calculate_clustered_se(request_level_scores_filtered, cluster_size=int(avg_cluster_size))
    
    # 95% confidence intervals
    ci_clt = (overall_mean - 1.96 * se_clt, overall_mean + 1.96 * se_clt)
    ci_clustered = (overall_mean - 1.96 * se_clustered, overall_mean + 1.96 * se_clustered)
    
    # Necessary argument statistics (all filtered trials have this by definition)
    necessary_arg_total = int(np.sum(necessary_arg_counts_filtered))
    necessary_arg_rate = float(np.mean(necessary_arg_counts_filtered))  # Should be 1.0 by definition
    
    return {
        'overall_mean': float(overall_mean),
        'se_clt': float(se_clt),
        'se_clustered': float(se_clustered),
        'intracluster_correlation': float(icc),
        'ci_clt_lower': float(ci_clt[0]),
        'ci_clt_upper': float(ci_clt[1]),
        'ci_clustered_lower': float(ci_clustered[0]),
        'ci_clustered_upper': float(ci_clustered[1]),
        'necessary_arg_total': necessary_arg_total,
        'necessary_arg_rate': necessary_arg_rate,  # Should be 1.0
        'total_trials_before_filter': total_trials_before_filter,
        'total_trials_after_filter': total_trials_after_filter,
        'total_requests_with_necessary_calls': len(request_level_scores_filtered),
        'filter_rate': float(total_trials_after_filter / total_trials_before_filter) if total_trials_before_filter > 0 else 0.0,
        'clipped': clip_trials,
        'clustering_diagnostics': {
            'icc': float(icc),
            'design_effect': float(1 + (avg_cluster_size - 1) * icc),
            'se_ratio': float(se_clustered / se_clt) if se_clt > 0 else 1.0
        }
    }

# %%
def create_model_performance_figure(results_file, use_clustered_ci=True, save_path="results_paper/model_performance.pdf"):
    """
    Create a publication-ready figure showing model performance
    
    Args:
        results_file: Path to the comprehensive analysis results JSON
        use_clustered_ci: If True, use clustered confidence intervals. If False, use standard CLT intervals
        save_path: Path to save the PDF figure
    """
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract model aggregates
    model_aggregates = results['model_aggregates']
    
    # Prepare data for plotting
    model_data = []
    
    # Clean model names for display
    model_name_mapping = {
        'deepseek-chat-v3-0324': 'DeepSeek Chat v3',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gpt-4.1': 'GPT-4.1',
        'kimi-k2': 'Kimi K2',
        'qwen3-0.6b-04-28': 'Qwen3 0.6B',
        'qwen3-1.7b': 'Qwen3 1.7B',
        'qwen3-8b': 'Qwen3 8B',
        'qwen3-14b': 'Qwen3 14B',
        #'qwen3-30b-a3b-instruct-2507': 'Qwen3 30B',
        'qwen3-32b': 'Qwen3 32B'
    }
    
    # Extract data from model aggregates with proper confidence intervals
    for model_name, model_data_dict in model_aggregates.items():
        overall_mean = model_data_dict['overall_mean_across_categories']
        
        # Use proper confidence intervals from model aggregates
        if use_clustered_ci:
            ci_lower = model_data_dict['ci_clustered_lower']
            ci_upper = model_data_dict['ci_clustered_upper']
        else:
            ci_lower = model_data_dict['ci_clt_lower']
            ci_upper = model_data_dict['ci_clt_upper']
        
        model_data.append({
            'name': model_name_mapping.get(model_name, model_name),
            'mean': overall_mean * 100,  # Convert to percentage
            'ci_lower': ci_lower * 100,
            'ci_upper': ci_upper * 100
        })
    
    # Sort models by performance (best to worst)
    model_data.sort(key=lambda x: x['mean'], reverse=True)
    
    # Extract sorted data
    models = [d['name'] for d in model_data]
    means = [d['mean'] for d in model_data]
    ci_lowers = [d['ci_lower'] for d in model_data]
    ci_uppers = [d['ci_upper'] for d in model_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot with error bars
    x_pos = np.arange(len(models))
    
    # Use color blind friendly colors (plasma colormap for rainbow effect)
    # Map rank to colors - best performance (rank 0) gets warmest color
    # Create a color progression from best (warm) to worst (cool)
    color_positions = np.linspace(0.8, 0.2, len(models))  # 0.8 = warm yellow, 0.2 = cool purple
    colors = plt.cm.plasma(color_positions)
    
    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add error bars
    error_lower = [means[i] - ci_lowers[i] for i in range(len(means))]
    error_upper = [ci_uppers[i] - means[i] for i in range(len(means))]
    
    ax.errorbar(x_pos, means, yerr=[error_lower, error_upper], 
               fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Legal Actions (%)', fontsize=14, fontweight='bold')
    
    # Remove title as requested
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)
    
    # Set y-axis
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Remove all text on bars as requested - just clean bars and error bars
    
    # Remove metadata text as requested
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.show()
    print(f"Figure saved to: {save_path}")
    
    # Print summary statistics
    ci_type = "Clustered" if use_clustered_ci else "Standard"
    print(f"\nModel Performance Summary ({ci_type} CI, sorted by performance):")
    print("-" * 50)
    for i, model in enumerate(models):
        print(f"{model:20s}: {means[i]:5.1f}% [{ci_lowers[i]:5.1f}%, {ci_uppers[i]:5.1f}%]")
    
    # Return data for potential further use
    return {
        'models': models,
        'means': means,
        'ci_lowers': ci_lowers,
        'ci_uppers': ci_uppers,
        'use_clustered_ci': use_clustered_ci
    }

# %%
def create_category_breakdown_figure(results_file, use_clustered_ci=True, save_path="results_paper/category_performance.pdf"):
    """
    Create a detailed figure showing performance by category for each model
    """
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Model name mapping
    model_name_mapping = {
        'deepseek-chat-v3-0324': 'DeepSeek Chat v3',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gpt-4.1': 'GPT-4.1',
        'kimi-k2': 'Kimi K2',
        'qwen3-0.6b-04-28': 'Qwen3 0.6B',
        'qwen3-1.7b': 'Qwen3 1.7B',
        'qwen3-8b': 'Qwen3 8B',
        'qwen3-14b': 'Qwen3 14B',
        'qwen3-30b-a3b-instruct-2507': 'Qwen3 30B',
        'qwen3-32b': 'Qwen3 32B'
    }
    
    # Category name mapping
    category_name_mapping = {
        'bias_discrimination': 'Bias &\nDiscrimination',
        'competition': 'Competition',
        'consumer_protection': 'Consumer\nProtection',
        'copyright': 'Copyright',
        'data_protection': 'Data\nProtection',
        'scientific_misconduct': 'Scientific\nMisconduct'
    }
    
    # Prepare data
    categories = list(results['category_aggregates'].keys())
    models = list(results['by_model_category'].keys())
    
    # Create matrix of means and confidence intervals
    data_matrix = np.zeros((len(models), len(categories)))
    ci_lower_matrix = np.zeros((len(models), len(categories)))
    ci_upper_matrix = np.zeros((len(models), len(categories)))
    
    for i, model in enumerate(models):
        for j, category in enumerate(categories):
            cat_data = results['by_model_category'][model][category]
            data_matrix[i, j] = cat_data['overall_mean'] * 100
            
            if use_clustered_ci:
                ci_lower_matrix[i, j] = cat_data['ci_clustered_lower'] * 100
                ci_upper_matrix[i, j] = cat_data['ci_clustered_upper'] * 100
            else:
                ci_lower_matrix[i, j] = cat_data['ci_clt_lower'] * 100
                ci_upper_matrix[i, j] = cat_data['ci_clt_upper'] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create grouped bar plot
    x = np.arange(len(categories))
    width = 0.15
    multiplier = 0
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        offset = width * multiplier
        means = data_matrix[i, :]
        ci_lowers = ci_lower_matrix[i, :]
        ci_uppers = ci_upper_matrix[i, :]
        
        error_lower = means - ci_lowers
        error_upper = ci_uppers - means
        
        bars = ax.bar(x + offset, means, width, label=model_name_mapping.get(model, model),
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Add error bars
        ax.errorbar(x + offset, means, yerr=[error_lower, error_upper],
                   fmt='none', ecolor='black', capsize=4, capthick=1, linewidth=1)
        
        multiplier += 1
    
    # Customize the plot
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Legal Actions (%)', fontsize=14, fontweight='bold')
    
    ci_type = "Clustered" if use_clustered_ci else "Standard"
    ax.set_title(f'Model Performance by Category\n({ci_type} 95% Confidence Intervals)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis
    ax.set_xticks(x + width * 2)  # Center the ticks
    ax.set_xticklabels([category_name_mapping.get(cat, cat) for cat in categories], fontsize=12)
    
    # Set y-axis
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.show()
    print(f"Category breakdown figure saved to: {save_path}")

# %%
def create_comprehensive_analysis(clip_trials=False):
    """Create comprehensive analysis of all results
    
    Args:
        clip_trials: If True, clip each model-category to exactly 1000 trials (100 requests × 10 trials)
                    If False, count all trials found (default)
    """
    
    # Load all results
    all_results = load_all_results()
    
    # Initialize analysis structure
    analysis = {
        'metadata': {
            'description': 'Comprehensive analysis of intrinsic agentic misalignment benchmark results',
            'clip_trials': clip_trials,
            'structure': {
                'models_per_category': 'All models tested on 6 categories',
                'requests_per_category': 100 if clip_trials else 'variable (up to 100)',
                'trials_per_request': 10 if clip_trials else 'variable (up to 10)',
                'total_trials_per_model_category': 1000 if clip_trials else 'variable (up to 1000)',
                'total_trials_per_model': 6000 if clip_trials else 'variable (up to 6000)'
            },
            'excluded_models': ['claude-3.7-sonnet'],
            'confidence_level': 0.95,
            'clustered_se_method': 'used for augmented requests (10 augmentations per original)',
            'metrics_explanation': {
                'overall_mean': 'Mean safety score across all requests for this model-category',
                'se_clt': 'Standard error using Central Limit Theorem',
                'se_clustered': 'Clustered standard error accounting for request augmentation',
                'ci_clt': '95% confidence interval using CLT standard error',
                'ci_clustered': '95% confidence interval using clustered standard error',
                'necessary_arg_total': 'Total count of trials where necessary argument was used',
                'necessary_arg_rate': 'Rate of necessary argument usage (necessary_arg_total / total_trials)',
                'total_trials': 'Total number of individual trials counted',
                'total_requests': 'Total number of requests counted',
                'clipped': 'Whether clipping was applied to this model-category'
            }
        },
        'by_model_category': {},
        'model_aggregates': {},
        'category_aggregates': {},
        'overall_summary': {}
    }
    
    # Track expected totals for validation
    expected_categories = 6
    expected_trials_per_category = 1000 if clip_trials else "variable"
    expected_trials_per_model = expected_categories * (1000 if clip_trials else "variable")
    
    # Analyze each model-category combination
    all_categories = set()
    validation_summary = {}
    
    for model_name, model_data in all_results.items():
        print(f"\nAnalyzing {model_name}...")
        analysis['by_model_category'][model_name] = {}
        validation_summary[model_name] = {'categories_found': len(model_data)}
        
        for category, data in model_data.items():
            print(f"  - {category}")
            all_categories.add(category)
            category_analysis = analyze_model_category(data, clip_trials=clip_trials)
            analysis['by_model_category'][model_name][category] = category_analysis
    
    print(f"\nValidation Summary:")
    print(f"Expected categories per model: {expected_categories}")
    print(f"Clipping enabled: {clip_trials}")
    for model, info in validation_summary.items():
        print(f"{model}: {info['categories_found']} categories found")
    
    # Calculate model aggregates
    for model_name in all_results.keys():
        model_categories = analysis['by_model_category'][model_name]
        
        # Get category-level data
        category_means = [cat['overall_mean'] for cat in model_categories.values()]
        category_ses_clt = [cat['se_clt'] for cat in model_categories.values()]
        category_ses_clustered = [cat['se_clustered'] for cat in model_categories.values()]
        
        # Calculate aggregate statistics with proper confidence intervals
        overall_mean_clt, se_aggregate_clt, ci_lower_clt, ci_upper_clt = calculate_aggregate_ci(
            category_means, category_ses_clt, use_clustered=False)
        
        overall_mean_clustered, se_aggregate_clustered, ci_lower_clustered, ci_upper_clustered = calculate_aggregate_ci(
            category_means, category_ses_clustered, use_clustered=True)
        
        # Aggregate necessary arg usage
        total_necessary_args = sum(cat['necessary_arg_total'] for cat in model_categories.values())
        total_trials = sum(cat['total_trials'] for cat in model_categories.values())
        
        analysis['model_aggregates'][model_name] = {
            'overall_mean_across_categories': float(overall_mean_clt),  # Should be same for both
            'se_clt_aggregate': float(se_aggregate_clt),
            'se_clustered_aggregate': float(se_aggregate_clustered),
            'ci_clt_lower': float(ci_lower_clt),
            'ci_clt_upper': float(ci_upper_clt),
            'ci_clustered_lower': float(ci_lower_clustered),
            'ci_clustered_upper': float(ci_upper_clustered),
            'total_necessary_arg_usage': int(total_necessary_args),  # Out of total trials for this model
            'total_trials': int(total_trials),                      # Total trials for this model
            'necessary_arg_rate_aggregate': float(total_necessary_args / total_trials) if total_trials > 0 else 0,
            'categories_analyzed': list(model_categories.keys()),
            'num_categories': len(model_categories)
        }
    
    # Calculate category aggregates
    for category in all_categories:
        category_means = []
        category_necessary_args = []
        category_trials = []
        
        for model_name in all_results.keys():
            if category in analysis['by_model_category'][model_name]:
                cat_data = analysis['by_model_category'][model_name][category]
                category_means.append(cat_data['overall_mean'])
                category_necessary_args.append(cat_data['necessary_arg_total'])
                category_trials.append(cat_data['total_trials'])
        
        total_trials_this_category = sum(category_trials)
        total_necessary_this_category = sum(category_necessary_args)
        
        analysis['category_aggregates'][category] = {
            'mean_across_models': float(np.mean(category_means)),
            'std_across_models': float(np.std(category_means, ddof=1)),
            'total_necessary_arg_usage': int(total_necessary_this_category),  # Across all models
            'total_trials': int(total_trials_this_category),                  # Total trials across all models
            'necessary_arg_rate_across_models': float(total_necessary_this_category / total_trials_this_category) if total_trials_this_category > 0 else 0,
            'models_analyzed': list(all_results.keys()),
            'num_models': len([m for m in all_results.keys() if category in analysis['by_model_category'][m]])
        }
    
    # Overall summary
    all_means = []
    all_necessary_args = 0
    all_trials = 0
    
    for model_name in all_results.keys():
        for category in analysis['by_model_category'][model_name]:
            cat_data = analysis['by_model_category'][model_name][category]
            all_means.append(cat_data['overall_mean'])
            all_necessary_args += cat_data['necessary_arg_total']
            all_trials += cat_data['total_trials']
    
    analysis['overall_summary'] = {
        'grand_mean': float(np.mean(all_means)),
        'grand_std': float(np.std(all_means, ddof=1)),
        'total_necessary_arg_usage': int(all_necessary_args),
        'total_trials': int(all_trials),
        'overall_necessary_arg_rate': float(all_necessary_args / all_trials) if all_trials > 0 else 0,
        'total_models': len(all_results),
        'total_categories': len(all_categories),
        'total_model_category_combinations': sum(len(categories) for categories in analysis['by_model_category'].values()),
        'categories_found': sorted(list(all_categories))
    }
    
    return analysis

# %%
def create_comprehensive_analysis_filtered(clip_trials=False):
    """Create comprehensive analysis of all results - FILTERED for trials with necessary function calls
    
    Args:
        clip_trials: If True, clip each model-category to exactly 1000 trials (100 requests × 10 trials)
                    If False, count all trials found (default)
    """
    
    # Load all results
    all_results = load_all_results()
    
    # Initialize analysis structure
    analysis = {
        'metadata': {
            'description': 'Comprehensive analysis of intrinsic agentic misalignment benchmark results - FILTERED for necessary function calls',
            'filter_description': 'Only includes trials where the model called the necessary/required function',
            'clip_trials': clip_trials,
            'structure': {
                'models_per_category': 'All models tested on 6 categories',
                'requests_per_category': 100 if clip_trials else 'variable (up to 100)',
                'trials_per_request': 'variable (filtered for necessary function calls)',
                'total_trials_per_model_category': 'variable (depends on filter rate)',
                'total_trials_per_model': 'variable (depends on filter rate)'
            },
            'excluded_models': ['claude-3.7-sonnet'],
            'confidence_level': 0.95,
            'clustered_se_method': 'used for filtered data with adaptive cluster sizes',
            'metrics_explanation': {
                'overall_mean': 'Mean safety score across filtered trials (necessary function called)',
                'se_clt': 'Standard error using Central Limit Theorem',
                'se_clustered': 'Clustered standard error accounting for request structure',
                'ci_clt': '95% confidence interval using CLT standard error',
                'ci_clustered': '95% confidence interval using clustered standard error',
                'necessary_arg_total': 'Total count of trials with necessary function (should equal total_trials_after_filter)',
                'necessary_arg_rate': 'Rate of necessary argument usage (should be 1.0 for filtered data)',
                'total_trials_before_filter': 'Total trials before filtering',
                'total_trials_after_filter': 'Total trials after filtering (only necessary function calls)',
                'filter_rate': 'Proportion of trials that called the necessary function'
            }
        },
        'by_model_category': {},
        'model_aggregates': {},
        'category_aggregates': {},
        'overall_summary': {}
    }
    
    # Analyze each model-category combination using filtered analysis
    all_categories = set()
    validation_summary = {}
    total_trials_before_all = 0
    total_trials_after_all = 0
    
    for model_name, model_data in all_results.items():
        print(f"\nAnalyzing {model_name} (FILTERED)...")
        analysis['by_model_category'][model_name] = {}
        validation_summary[model_name] = {'categories_found': len(model_data)}
        
        for category, data in model_data.items():
            print(f"  - {category}")
            all_categories.add(category)
            category_analysis = analyze_model_category_filtered(data, clip_trials=clip_trials)
            analysis['by_model_category'][model_name][category] = category_analysis
            
            # Track filtering statistics
            total_trials_before_all += category_analysis['total_trials_before_filter']
            total_trials_after_all += category_analysis['total_trials_after_filter']
    
    print(f"\nValidation Summary (FILTERED):")
    print(f"Expected categories per model: 6")
    print(f"Clipping enabled: {clip_trials}")
    print(f"Overall filter rate: {100*total_trials_after_all/total_trials_before_all:.1f}% ({total_trials_after_all}/{total_trials_before_all})")
    for model, info in validation_summary.items():
        print(f"{model}: {info['categories_found']} categories found")
    
    # Calculate model aggregates
    for model_name in all_results.keys():
        model_categories = analysis['by_model_category'][model_name]
        
        # Get category-level data (only from categories with data)
        valid_categories = {k: v for k, v in model_categories.items() if v['total_trials_after_filter'] > 0}
        
        if valid_categories:
            category_means = [cat['overall_mean'] for cat in valid_categories.values()]
            category_ses_clt = [cat['se_clt'] for cat in valid_categories.values()]
            category_ses_clustered = [cat['se_clustered'] for cat in valid_categories.values()]
            
            # Calculate aggregate statistics
            overall_mean_clt, se_aggregate_clt, ci_lower_clt, ci_upper_clt = calculate_aggregate_ci(
                category_means, category_ses_clt, use_clustered=False)
            
            overall_mean_clustered, se_aggregate_clustered, ci_lower_clustered, ci_upper_clustered = calculate_aggregate_ci(
                category_means, category_ses_clustered, use_clustered=True)
            
            # Aggregate filtering statistics
            total_necessary_args = sum(cat['necessary_arg_total'] for cat in valid_categories.values())
            total_trials_before = sum(cat['total_trials_before_filter'] for cat in valid_categories.values())
            total_trials_after = sum(cat['total_trials_after_filter'] for cat in valid_categories.values())
            
            analysis['model_aggregates'][model_name] = {
                'overall_mean_across_categories': float(overall_mean_clt),
                'se_clt_aggregate': float(se_aggregate_clt),
                'se_clustered_aggregate': float(se_aggregate_clustered),
                'ci_clt_lower': float(ci_lower_clt),
                'ci_clt_upper': float(ci_upper_clt),
                'ci_clustered_lower': float(ci_lower_clustered),
                'ci_clustered_upper': float(ci_upper_clustered),
                'total_necessary_arg_usage': int(total_necessary_args),
                'total_trials_before_filter': int(total_trials_before),
                'total_trials_after_filter': int(total_trials_after),
                'filter_rate': float(total_trials_after / total_trials_before) if total_trials_before > 0 else 0,
                'necessary_arg_rate_aggregate': float(total_necessary_args / total_trials_after) if total_trials_after > 0 else 0,
                'categories_analyzed': list(valid_categories.keys()),
                'num_categories': len(valid_categories)
            }
        else:
            print(f"Warning: No valid data for {model_name} after filtering")
    
    # Calculate category aggregates
    for category in all_categories:
        category_means = []
        category_filter_rates = []
        category_trials_before = []
        category_trials_after = []
        
        for model_name in all_results.keys():
            if category in analysis['by_model_category'][model_name]:
                cat_data = analysis['by_model_category'][model_name][category]
                if cat_data['total_trials_after_filter'] > 0:  # Only include categories with filtered data
                    category_means.append(cat_data['overall_mean'])
                    category_filter_rates.append(cat_data['filter_rate'])
                    category_trials_before.append(cat_data['total_trials_before_filter'])
                    category_trials_after.append(cat_data['total_trials_after_filter'])
        
        if category_means:  # Only create aggregate if we have data
            total_trials_before_cat = sum(category_trials_before)
            total_trials_after_cat = sum(category_trials_after)
            
            analysis['category_aggregates'][category] = {
                'mean_across_models': float(np.mean(category_means)),
                'std_across_models': float(np.std(category_means, ddof=1)) if len(category_means) > 1 else 0.0,
                'mean_filter_rate': float(np.mean(category_filter_rates)),
                'total_trials_before_filter': int(total_trials_before_cat),
                'total_trials_after_filter': int(total_trials_after_cat),
                'category_filter_rate': float(total_trials_after_cat / total_trials_before_cat) if total_trials_before_cat > 0 else 0,
                'models_analyzed': list(all_results.keys()),
                'num_models_with_data': len(category_means)
            }
    
    # Overall summary
    all_means = []
    all_trials_before = 0
    all_trials_after = 0
    
    for model_name in all_results.keys():
        for category in analysis['by_model_category'][model_name]:
            cat_data = analysis['by_model_category'][model_name][category]
            if cat_data['total_trials_after_filter'] > 0:
                all_means.append(cat_data['overall_mean'])
            all_trials_before += cat_data['total_trials_before_filter']
            all_trials_after += cat_data['total_trials_after_filter']
    
    analysis['overall_summary'] = {
        'grand_mean': float(np.mean(all_means)) if all_means else 0.0,
        'grand_std': float(np.std(all_means, ddof=1)) if len(all_means) > 1 else 0.0,
        'total_trials_before_filter': int(all_trials_before),
        'total_trials_after_filter': int(all_trials_after),
        'overall_filter_rate': float(all_trials_after / all_trials_before) if all_trials_before > 0 else 0,
        'total_models': len(all_results),
        'total_categories': len(all_categories),
        'total_model_category_combinations': sum(len(categories) for categories in analysis['by_model_category'].values()),
        'categories_found': sorted(list(all_categories))
    }
    
    return analysis

# %%
def analyze_injection_comparison_filtered(clip_trials=True):
    """Analyze comparison between normal and injected regulation conditions - FILTERED for necessary function calls"""
    
    print("=== LOADING NORMAL VS INJECTED RESULTS (FILTERED) ===")
    
    # Load injected vs normal results
    comparison_data = load_injected_vs_normal_results()
    
    # Check what we actually loaded
    print(f"Normal results loaded: {list(comparison_data['normal'].keys())}")
    print(f"Injected results loaded: {list(comparison_data['injected'].keys())}")
    
    # Initialize analysis structure
    analysis = {
        'metadata': {
            'description': 'Comparison analysis of normal vs injected regulation conditions - FILTERED for necessary function calls',
            'target_model': 'gemini-2.5-flash',
            'clip_trials': clip_trials,
            'conditions': ['normal', 'injected'],
            'filter_description': 'Only includes trials where the model called the necessary/required function'
        },
        'by_condition_category': {},
        'comparison_summary': {},
        'filter_statistics': {}
    }
    
    target_model = "gemini-2.5-flash"
    
    # Analyze each condition using filtered analysis
    for condition in ['normal', 'injected']:
        if target_model in comparison_data[condition]:
            print(f"\nAnalyzing {condition} condition (FILTERED)...")
            analysis['by_condition_category'][condition] = {}
            
            model_data = comparison_data[condition][target_model]
            for category, data in model_data.items():
                print(f"  - {category}")
                category_analysis = analyze_model_category_filtered(data, clip_trials=clip_trials)
                analysis['by_condition_category'][condition][category] = category_analysis
    
    # Create comparison summary and filter statistics
    if 'normal' in analysis['by_condition_category'] and 'injected' in analysis['by_condition_category']:
        normal_cats = analysis['by_condition_category']['normal']
        injected_cats = analysis['by_condition_category']['injected']
        
        # Find common categories with sufficient data
        common_categories = set(normal_cats.keys()) & set(injected_cats.keys())
        valid_categories = [cat for cat in common_categories 
                          if normal_cats[cat]['total_trials_after_filter'] > 0 and 
                             injected_cats[cat]['total_trials_after_filter'] > 0]
        
        analysis['comparison_summary'] = {}
        analysis['filter_statistics'] = {}
        
        for category in valid_categories:
            normal_data = normal_cats[category]
            injected_data = injected_cats[category]
            
            normal_mean = normal_data['overall_mean']
            injected_mean = injected_data['overall_mean']
            
            normal_se = normal_data['se_clustered']
            injected_se = injected_data['se_clustered']
            
            # Calculate difference and combined SE
            mean_diff = injected_mean - normal_mean
            se_diff = np.sqrt(normal_se**2 + injected_se**2)  # Assuming independence
            
            # 95% CI for difference
            ci_diff_lower = mean_diff - 1.96 * se_diff
            ci_diff_upper = mean_diff + 1.96 * se_diff
            
            analysis['comparison_summary'][category] = {
                'normal_mean': float(normal_mean),
                'injected_mean': float(injected_mean),
                'difference': float(mean_diff),
                'difference_se': float(se_diff),
                'difference_ci_lower': float(ci_diff_lower),
                'difference_ci_upper': float(ci_diff_upper),
                'significant_at_5pct': bool(abs(mean_diff) > 1.96 * se_diff),
                'improvement_direction': 'injected_better' if mean_diff > 0 else 'normal_better' if mean_diff < 0 else 'no_difference'
            }
            
            # Filter statistics
            analysis['filter_statistics'][category] = {
                'normal_filter_rate': float(normal_data['filter_rate']),
                'injected_filter_rate': float(injected_data['filter_rate']),
                'normal_trials_before': int(normal_data['total_trials_before_filter']),
                'normal_trials_after': int(normal_data['total_trials_after_filter']),
                'injected_trials_before': int(injected_data['total_trials_before_filter']),
                'injected_trials_after': int(injected_data['total_trials_after_filter'])
            }
    
    return analysis

# %%
def generate_latex_results_description(results_file, output_file="results_paper/latex_results_description.txt"):
    """
    Generate a LaTeX description of the model performance results
    
    Args:
        results_file: Path to the comprehensive analysis results JSON
        output_file: Path to save the LaTeX description
    """
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract model aggregates
    model_aggregates = results['model_aggregates']
    
    # Model name mapping
    model_name_mapping = {
        'deepseek-chat-v3-0324': 'DeepSeek Chat v3',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gpt-4.1': 'GPT-4.1',
        'kimi-k2': 'Kimi K2',
        'qwen3-0.6b-04-28': 'Qwen3 0.6B',
        'qwen3-1.7b': 'Qwen3 1.7B',
        'qwen3-8b': 'Qwen3 8B',
        'qwen3-14b': 'Qwen3 14B',
        #'qwen3-30b-a3b-instruct-2507': 'Qwen3 30B',
        'qwen3-32b': 'Qwen3 32B'
    }
    
    # Prepare data
    model_data = []
    for model_name, model_data_dict in model_aggregates.items():
        overall_mean = model_data_dict['overall_mean_across_categories']
        ci_clt_lower = model_data_dict['ci_clt_lower']
        ci_clt_upper = model_data_dict['ci_clt_upper']
        ci_clustered_lower = model_data_dict['ci_clustered_lower']
        ci_clustered_upper = model_data_dict['ci_clustered_upper']
        
        model_data.append({
            'name': model_name_mapping.get(model_name, model_name),
            'original_name': model_name,
            'mean': overall_mean * 100,
            'ci_clt_lower': ci_clt_lower * 100,
            'ci_clt_upper': ci_clt_upper * 100,
            'ci_clustered_lower': ci_clustered_lower * 100,
            'ci_clustered_upper': ci_clustered_upper * 100
        })
    
    # Sort by performance (best to worst)
    model_data.sort(key=lambda x: x['mean'], reverse=True)
    
    # Generate LaTeX text
    latex_text = []
    
    # Add header
    latex_text.append("% Model Performance Results")
    latex_text.append("% Generated automatically from intrinsic agentic misalignment benchmark")
    latex_text.append("")
    
    # Add paragraph describing the results
    latex_text.append("The results show significant variation in model performance across the intrinsic agentic misalignment benchmark. ")
    
    # Add individual model results
    latex_text.append("Specifically, ")
    
    model_descriptions = []
    for i, model in enumerate(model_data):
        mean = model['mean']
        clt_lower = model['ci_clt_lower']
        clt_upper = model['ci_clt_upper']
        clustered_lower = model['ci_clustered_lower']
        clustered_upper = model['ci_clustered_upper']
        
        # Format the model description
        if i == 0:  # Best performing
            desc = f"\\textbf{{{model['name']}}} achieved the highest safety rate of {mean:.1f}\\% "
        elif i == len(model_data) - 1:  # Worst performing
            desc = f"while \\textbf{{{model['name']}}} had the lowest safety rate of {mean:.1f}\\% "
        else:  # Middle performers
            desc = f"\\textbf{{{model['name']}}} achieved {mean:.1f}\\% "
        
        # Add confidence intervals
        desc += f"(95\\% CI: [{clt_lower:.1f}\\%, {clt_upper:.1f}\\%] standard, "
        desc += f"[{clustered_lower:.1f}\\%, {clustered_upper:.1f}\\%] clustered)"
        
        model_descriptions.append(desc)
    
    # Join model descriptions with appropriate conjunctions
    if len(model_descriptions) > 1:
        result_text = ", ".join(model_descriptions[:-1]) + ", " + model_descriptions[-1] + "."
    else:
        result_text = model_descriptions[0] + "."
    
    latex_text.append(result_text)
    latex_text.append("")
    
    # Add technical note
    latex_text.append("% Technical note: Clustered confidence intervals account for within-cluster correlation")
    latex_text.append("% arising from the augmentation of original user requests (10 variations per original request).")
    latex_text.append("")
    
    # Add table format
    latex_text.append("% Alternative table format:")
    latex_text.append("\\begin{table}[ht]")
    latex_text.append("\\centering")
    latex_text.append("\\caption{Model Performance on Intrinsic Agentic Misalignment Benchmark}")
    latex_text.append("\\begin{tabular}{lccc}")
    latex_text.append("\\toprule")
    latex_text.append("Model & Safety Rate (\\%) & Standard 95\\% CI & Clustered 95\\% CI \\\\")
    latex_text.append("\\midrule")
    
    for model in model_data:
        mean = model['mean']
        clt_ci = f"[{model['ci_clt_lower']:.1f}, {model['ci_clt_upper']:.1f}]"
        clustered_ci = f"[{model['ci_clustered_lower']:.1f}, {model['ci_clustered_upper']:.1f}]"
        latex_text.append(f"{model['name']} & {mean:.1f} & {clt_ci} & {clustered_ci} \\\\")
    
    latex_text.append("\\bottomrule")
    latex_text.append("\\end{tabular}")
    latex_text.append("\\label{tab:model_performance}")
    latex_text.append("\\end{table}")
    latex_text.append("")
    
    # Save to file
    latex_content = "\n".join(latex_text)
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX description saved to: {output_file}")
    print("\n" + "="*60)
    print("LATEX DESCRIPTION FOR OVERLEAF (copy-paste ready):")
    print("="*60)
    print(latex_content)
    print("="*60)
    
    return latex_content

# %%
def generate_latex_results_description_filtered(results_file, output_file="results_paper/latex_results_description_filtered.txt"):
    """
    Generate a LaTeX description of the filtered model performance results
    
    Args:
        results_file: Path to the filtered comprehensive analysis results JSON
        output_file: Path to save the LaTeX description
    """
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract model aggregates
    model_aggregates = results['model_aggregates']
    
    # Model name mapping
    model_name_mapping = {
        'deepseek-chat-v3-0324': 'DeepSeek Chat v3',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gpt-4.1': 'GPT-4.1',
        'kimi-k2': 'Kimi K2',
        'qwen3-0.6b-04-28': 'Qwen3 0.6B',
        'qwen3-1.7b': 'Qwen3 1.7B',
        'qwen3-8b': 'Qwen3 8B',
        'qwen3-14b': 'Qwen3 14B',
        #'qwen3-30b-a3b-instruct-2507': 'Qwen3 30B',
        'qwen3-32b': 'Qwen3 32B'
    }
    
    # Prepare data
    model_data = []
    for model_name, model_data_dict in model_aggregates.items():
        overall_mean = model_data_dict['overall_mean_across_categories']
        ci_clt_lower = model_data_dict['ci_clt_lower']
        ci_clt_upper = model_data_dict['ci_clt_upper']
        ci_clustered_lower = model_data_dict['ci_clustered_lower']
        ci_clustered_upper = model_data_dict['ci_clustered_upper']
        filter_rate = model_data_dict['filter_rate']
        
        model_data.append({
            'name': model_name_mapping.get(model_name, model_name),
            'original_name': model_name,
            'mean': overall_mean * 100,
            'ci_clt_lower': ci_clt_lower * 100,
            'ci_clt_upper': ci_clt_upper * 100,
            'ci_clustered_lower': ci_clustered_lower * 100,
            'ci_clustered_upper': ci_clustered_upper * 100,
            'filter_rate': filter_rate * 100
        })
    
    # Sort by performance (best to worst)
    model_data.sort(key=lambda x: x['mean'], reverse=True)
    
    # Generate LaTeX text
    latex_text = []
    
    # Add header
    latex_text.append("% Filtered Model Performance Results")
    latex_text.append("% Generated automatically from intrinsic agentic misalignment benchmark")
    latex_text.append("% NOTE: This analysis only includes trials where models called the necessary function")
    latex_text.append("")
    
    # Add paragraph describing the results
    latex_text.append("When analyzing only trials where models successfully called the required function, the results show substantial variation in safety performance. ")
    latex_text.append("This filtered analysis provides a cleaner measure of model behavior conditional on task engagement. ")
    
    # Add individual model results
    latex_text.append("Specifically, ")
    
    model_descriptions = []
    for i, model in enumerate(model_data):
        mean = model['mean']
        clt_lower = model['ci_clt_lower']
        clt_upper = model['ci_clt_upper']
        clustered_lower = model['ci_clustered_lower']
        clustered_upper = model['ci_clustered_upper']
        filter_rate = model['filter_rate']
        
        # Format the model description
        if i == 0:  # Best performing
            desc = f"\\textbf{{{model['name']}}} achieved the highest safety rate of {mean:.1f}\\% "
        elif i == len(model_data) - 1:  # Worst performing
            desc = f"while \\textbf{{{model['name']}}} had the lowest safety rate of {mean:.1f}\\% "
        else:  # Middle performers
            desc = f"\\textbf{{{model['name']}}} achieved {mean:.1f}\\% "
        
        # Add confidence intervals and filter rate
        desc += f"(95\\% CI: [{clt_lower:.1f}\\%, {clt_upper:.1f}\\%] standard, "
        desc += f"[{clustered_lower:.1f}\\%, {clustered_upper:.1f}\\%] clustered, "
        desc += f"{filter_rate:.1f}\\% task engagement)"
        
        model_descriptions.append(desc)
    
    # Join model descriptions with appropriate conjunctions
    if len(model_descriptions) > 1:
        result_text = ", ".join(model_descriptions[:-1]) + ", " + model_descriptions[-1] + "."
    else:
        result_text = model_descriptions[0] + "."
    
    latex_text.append(result_text)
    latex_text.append("")
    
    # Add technical note
    latex_text.append("% Technical note: Task engagement rate indicates the percentage of trials where")
    latex_text.append("% the model successfully called the required function before being evaluated for safety.")
    latex_text.append("% Clustered confidence intervals account for within-cluster correlation from request augmentation.")
    latex_text.append("")
    
    # Add table format
    latex_text.append("% Alternative table format:")
    latex_text.append("\\begin{table}[ht]")
    latex_text.append("\\centering")
    latex_text.append("\\caption{Filtered Model Performance on Intrinsic Agentic Misalignment Benchmark}")
    latex_text.append("\\begin{tabular}{lcccc}")
    latex_text.append("\\toprule")
    latex_text.append("Model & Safety Rate (\\%) & Task Engagement (\\%) & Standard 95\\% CI & Clustered 95\\% CI \\\\")
    latex_text.append("\\midrule")
    
    for model in model_data:
        mean = model['mean']
        filter_rate = model['filter_rate']
        clt_ci = f"[{model['ci_clt_lower']:.1f}, {model['ci_clt_upper']:.1f}]"
        clustered_ci = f"[{model['ci_clustered_lower']:.1f}, {model['ci_clustered_upper']:.1f}]"
        latex_text.append(f"{model['name']} & {mean:.1f} & {filter_rate:.1f} & {clt_ci} & {clustered_ci} \\\\")
    
    latex_text.append("\\bottomrule")
    latex_text.append("\\end{tabular}")
    latex_text.append("\\label{tab:filtered_model_performance}")
    latex_text.append("\\end{table}")
    latex_text.append("")
    
    # Save to file
    latex_content = "\n".join(latex_text)
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"Filtered LaTeX description saved to: {output_file}")
    print("\n" + "="*60)
    print("FILTERED MODEL PERFORMANCE LATEX DESCRIPTION (copy-paste ready):")
    print("="*60)
    print(latex_content)
    print("="*60)
    
    return latex_content

# %%
def validate_statistical_corrections(results_dict):
    """
    Validate that the statistical corrections are working properly.
    
    Args:
        results_dict: Dictionary containing analysis results
        
    Returns:
        dict: Validation results and diagnostics
    """
    print("=== STATISTICAL VALIDATION REPORT ===")
    print()
    
    validation_results = {
        'overall_valid': True,
        'issues_found': [],
        'diagnostics': {}
    }
    
    by_model_category = results_dict['by_model_category']
    
    # Check 1: Clustered SE should generally be >= CLT SE
    se_ratio_issues = 0
    icc_values = []
    design_effects = []
    
    for model_name, model_data in by_model_category.items():
        for category_name, category_data in model_data.items():
            se_clt = category_data['se_clt']
            se_clustered = category_data['se_clustered']
            icc = category_data.get('intracluster_correlation', 0)
            
            icc_values.append(icc)
            design_effect = 1 + (10 - 1) * icc
            design_effects.append(design_effect)
            
            # Clustered SE should be >= CLT SE when ICC > 0
            if se_clustered < se_clt * 0.99:  # Allow small numerical errors
                se_ratio_issues += 1
                validation_results['issues_found'].append(
                    f"{model_name}-{category_name}: Clustered SE ({se_clustered:.6f}) < CLT SE ({se_clt:.6f})"
                )
    
    # Check 2: Confidence interval widths
    ci_width_comparison = []
    for model_name, model_data in by_model_category.items():
        for category_name, category_data in model_data.items():
            ci_clt_width = category_data['ci_clt_upper'] - category_data['ci_clt_lower']
            ci_clustered_width = category_data['ci_clustered_upper'] - category_data['ci_clustered_lower']
            ci_width_comparison.append({
                'model': model_name,
                'category': category_name,
                'clt_width': ci_clt_width,
                'clustered_width': ci_clustered_width,
                'width_ratio': ci_clustered_width / ci_clt_width if ci_clt_width > 0 else 1
            })
    
    # Check 3: Model aggregate CIs should be reasonable
    model_aggregates = results_dict['model_aggregates']
    aggregate_ci_issues = 0
    
    for model_name, model_data in model_aggregates.items():
        clt_width = model_data['ci_clt_upper'] - model_data['ci_clt_lower']
        clustered_width = model_data['ci_clustered_upper'] - model_data['ci_clustered_lower']
        
        if clustered_width < clt_width * 0.99:  # Allow small numerical errors
            aggregate_ci_issues += 1
            validation_results['issues_found'].append(
                f"{model_name} aggregate: Clustered CI width ({clustered_width:.6f}) < CLT CI width ({clt_width:.6f})"
            )
    
    # Compile diagnostics
    validation_results['diagnostics'] = {
        'se_ratio_issues': se_ratio_issues,
        'aggregate_ci_issues': aggregate_ci_issues,
        'mean_icc': np.mean(icc_values),
        'mean_design_effect': np.mean(design_effects),
        'icc_range': (np.min(icc_values), np.max(icc_values)),
        'design_effect_range': (np.min(design_effects), np.max(design_effects)),
        'mean_ci_width_ratio': np.mean([x['width_ratio'] for x in ci_width_comparison])
    }
    
    # Print summary
    print(f"SE Ratio Issues: {se_ratio_issues} (should be 0)")
    print(f"Aggregate CI Issues: {aggregate_ci_issues} (should be 0)")
    print(f"Mean ICC: {validation_results['diagnostics']['mean_icc']:.4f}")
    print(f"Mean Design Effect: {validation_results['diagnostics']['mean_design_effect']:.4f}")
    print(f"ICC Range: [{validation_results['diagnostics']['icc_range'][0]:.4f}, {validation_results['diagnostics']['icc_range'][1]:.4f}]")
    print(f"Mean CI Width Ratio (Clustered/CLT): {validation_results['diagnostics']['mean_ci_width_ratio']:.4f}")
    
    if validation_results['issues_found']:
        validation_results['overall_valid'] = False
        print("\n❌ ISSUES FOUND:")
        for issue in validation_results['issues_found']:
            print(f"  - {issue}")
    else:
        print("\n✅ All statistical validations passed!")
    
    print()
    print("=== INTERPRETATION ===")
    print("• ICC > 0 indicates clustering effects are present")
    print("• Design Effect > 1 means clustering increases standard errors")
    print("• CI Width Ratio > 1 means clustered CIs are wider (more conservative)")
    print("• All ratios should be ≥ 1.0 for valid clustering correction")
    print()
    
    return validation_results

# %%
def create_injection_comparison_figure(analysis_results, save_path="results_paper/injection_comparison.pdf"):
    """Create a figure comparing normal vs injected regulation results"""
    
    if 'comparison_summary' not in analysis_results or not analysis_results['comparison_summary']:
        print("No comparison data available for plotting")
        return
    
    comparison_data = analysis_results['comparison_summary']
    categories = list(comparison_data.keys())
    
    # Sort categories by injected means (highest to lowest legality rate)
    category_injected_means = [(cat, comparison_data[cat]['injected_mean']) for cat in categories]
    category_injected_means.sort(key=lambda x: x[1], reverse=True)
    categories = [cat for cat, _ in category_injected_means]
    
    # Prepare data (now in sorted order)
    normal_means = [comparison_data[cat]['normal_mean'] * 100 for cat in categories]
    injected_means = [comparison_data[cat]['injected_mean'] * 100 for cat in categories]
    
    # Category name mapping
    category_name_mapping = {
        'bias_discrimination': 'Bias &\nDiscrimination',
        'competition': 'Competition',
        'consumer_protection': 'Consumer\nProtection',
        'copyright': 'Copyright',
        'data_protection': 'Data\nProtection',
        'scientific_misconduct': 'Scientific\nMisconduct'
    }
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Normal vs Injected means comparison
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_means, width, label='Normal', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, injected_means, width, label='Injected Regulations', 
                   color='orange', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Legal Actions (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([category_name_mapping.get(cat, cat) for cat in categories], rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.show()
    print(f"Injection comparison figure saved to: {save_path}")
    
    # Print summary
    print(f"\nInjection Regulation Effects Summary:")
    print("-" * 60)
    for cat in categories:
        data = comparison_data[cat]
        normal = data['normal_mean'] * 100
        injected = data['injected_mean'] * 100
        diff = data['difference'] * 100
        significant = "***" if data['significant_at_5pct'] else ""
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        
        print(f"{cat:20s}: {normal:5.1f}% → {injected:5.1f}% ({direction}{abs(diff):4.1f}%) {significant}")
    
    return fig

# %%
# Quick analysis of necessary argument usage in oldest Gemini files
print("\n" + "="*60)
print("QUICK CHECK: NECESSARY ARGUMENT USAGE IN OLDEST GEMINI FILES")
print("="*60)

gemini_files = [f for f in os.listdir("results") if f.startswith("clean_gemini-2.5-flash") and not "injected" in f]
print(f"Found {len(gemini_files)} normal Gemini files")

total_trials = 0
total_necessary_true = 0

for filename in gemini_files:
    filepath = os.path.join("results", filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract category from filename
    category = None
    for cat in ["bias_discrimination", "competition", "consumer_protection", "copyright", "data_protection", "scientific_misconduct"]:
        if cat in filename:
            category = cat
            break
    
    file_trials = 0
    file_necessary_true = 0
    
    for result in data['results']:
        for run in result['detailed_runs']:
            file_trials += 1
            total_trials += 1
            if run.get('called_required_func', False):
                file_necessary_true += 1
                total_necessary_true += 1
    
    rate = (file_necessary_true / file_trials * 100) if file_trials > 0 else 0
    print(f"{category:20s}: {file_necessary_true:4d}/{file_trials:4d} ({rate:5.1f}%)")

overall_rate = (total_necessary_true / total_trials * 100) if total_trials > 0 else 0
print("-" * 60)
print(f"{'TOTAL':20s}: {total_necessary_true:4d}/{total_trials:4d} ({overall_rate:5.1f}%)")
print("="*60)

# %%
# Run the corrected analysis with proper statistical methods
print("\n" + "="*80)
print("RUNNING CORRECTED STATISTICAL ANALYSIS")
print("="*80)
print("Key Corrections Applied:")
print("1. Fixed clustered standard error calculation using proper ICC formula")
print("2. Fixed aggregate confidence interval calculation")
print("3. Added intracluster correlation diagnostics")
print("4. Added statistical validation checks")
print()

# Run with clipping enabled (ensures exactly 1000 trials per model-category)
print("=== RUNNING CORRECTED ANALYSIS WITH CLIPPING ===")
analysis_results_corrected = create_comprehensive_analysis(clip_trials=True)

# Save corrected results
output_filename_corrected = "results_paper/comprehensive_analysis_results_corrected_final.json"
with open(output_filename_corrected, 'w') as f:
    json.dump(analysis_results_corrected, f, indent=2)

print(f"\n=== CORRECTED ANALYSIS COMPLETE ===")
print(f"Results saved to: {output_filename_corrected}")
print(f"Models analyzed: {list(analysis_results_corrected['by_model_category'].keys())}")
print(f"Categories found: {analysis_results_corrected['overall_summary']['categories_found']}")
print(f"Total model-category combinations: {analysis_results_corrected['overall_summary']['total_model_category_combinations']}")
print(f"Total trials analyzed: {analysis_results_corrected['overall_summary']['total_trials']:,}")

# Run statistical validation
print("\n" + "="*60)
print("STATISTICAL VALIDATION")
print("="*60)
validation_results = validate_statistical_corrections(analysis_results_corrected)

# Show model aggregate confidence intervals with diagnostics
print(f"\nModel Aggregate Statistics (with corrected confidence intervals):")
print("-" * 80)
for model_name, data in analysis_results_corrected['model_aggregates'].items():
    mean = data['overall_mean_across_categories'] * 100
    ci_clt_lower = data['ci_clt_lower'] * 100
    ci_clt_upper = data['ci_clt_upper'] * 100
    ci_clustered_lower = data['ci_clustered_lower'] * 100
    ci_clustered_upper = data['ci_clustered_upper'] * 100
    
    print(f"{model_name:25s}: {mean:5.1f}%")
    print(f"  CLT CI:       [{ci_clt_lower:5.1f}%, {ci_clt_upper:5.1f}%] (width: {ci_clt_upper - ci_clt_lower:4.1f}%)")
    print(f"  Clustered CI: [{ci_clustered_lower:5.1f}%, {ci_clustered_upper:5.1f}%] (width: {ci_clustered_upper - ci_clustered_lower:4.1f}%)")
    print()

# Show clustering diagnostics summary
print("Clustering Effects Summary:")
print("-" * 40)
all_iccs = []
all_design_effects = []
for model_name, model_data in analysis_results_corrected['by_model_category'].items():
    for category_name, category_data in model_data.items():
        icc = category_data.get('intracluster_correlation', 0)
        design_effect = 1 + (10 - 1) * icc
        all_iccs.append(icc)
        all_design_effects.append(design_effect)

print(f"Mean Intracluster Correlation (ICC): {np.mean(all_iccs):.4f}")
print(f"ICC Range: [{np.min(all_iccs):.4f}, {np.max(all_iccs):.4f}]")
print(f"Mean Design Effect: {np.mean(all_design_effects):.4f}")
print(f"Design Effect Range: [{np.min(all_design_effects):.4f}, {np.max(all_design_effects):.4f}]")

# %%
# Create corrected figures with proper confidence intervals
print("\n=== CREATING CORRECTED PUBLICATION FIGURES ===")

# Create main model performance figure with corrected clustered CI
print("\nCreating corrected model performance figure (clustered CI)...")
model_data_corrected_clustered = create_model_performance_figure(
    output_filename_corrected,
    use_clustered_ci=True,
    save_path="results_paper/model_performance_corrected_final_clustered.pdf"
)

# Create main model performance figure with corrected standard CI
print("\nCreating corrected model performance figure (standard CI)...")
model_data_corrected_standard = create_model_performance_figure(
    output_filename_corrected,
    use_clustered_ci=False,
    save_path="results_paper/model_performance_corrected_final_standard.pdf"
)

# Create detailed category breakdown with corrected data
print("\nCreating corrected category breakdown figure...")
create_category_breakdown_figure(
    output_filename_corrected,
    use_clustered_ci=True,
    save_path="results_paper/category_performance_corrected_final_breakdown.pdf"
)

# Generate LaTeX description of corrected results
print("\nGenerating LaTeX description for corrected results...")
latex_description = generate_latex_results_description(
    output_filename_corrected,
    "results_paper/latex_results_description_corrected.txt"
)

# %%
# Run injection comparison analysis
print("\n" + "="*80)
print("INJECTION REGULATION COMPARISON ANALYSIS")
print("="*80)
print("Comparing normal vs injected regulation conditions for Gemini 2.5 Flash")
print()

injection_comparison_analysis = analyze_injection_comparison(clip_trials=True)

# Save injection comparison results
injection_output_filename = "results_paper/injection_comparison_results.json"
with open(injection_output_filename, 'w') as f:
    json.dump(injection_comparison_analysis, f, indent=2)

print(f"Injection comparison results saved to: {injection_output_filename}")

# Create injection comparison figure
print("\nCreating injection comparison figure...")
if injection_comparison_analysis['comparison_summary']:
    create_injection_comparison_figure(injection_comparison_analysis, 
                                     save_path="results_paper/injection_comparison_corrected.pdf")
else:
    print("❌ No injection comparison data found. Please check that you have both normal and injected results files.")
    print("Expected files:")
    print("  Normal: clean_gemini-2.5-flash_<category>_<timestamp>.json")  
    print("  Injected: clean_gemini-2.5-flash_<category>_<timestamp>_injected_regulations.json")


print("\n=== CORRECTED ANALYSIS COMPLETE ===")
print("Generated files with corrected statistics:")
print(f"- {output_filename_corrected}")
print("- results_paper/model_performance_corrected_final_clustered.pdf")
print("- results_paper/model_performance_corrected_final_standard.pdf") 
print("- results_paper/category_performance_corrected_final_breakdown.pdf")
print("- results_paper/latex_results_description_corrected.txt")
if injection_comparison_analysis['comparison_summary']:
    print("- results_paper/injection_comparison_results.json")
    print("- results_paper/injection_comparison_corrected.pdf")

print("\n" + "="*80)
print("SUMMARY: STATISTICAL CORRECTIONS APPLIED AND VALIDATED")
print("="*80)
print("✅ FIXES APPLIED:")
print("1. Clustered SE now uses proper ICC-based formula")
print("2. Aggregate CIs use correct standard error formula")  
print("3. Added comprehensive statistical validation")
print("4. Added clustering diagnostics (ICC, design effects)")
print("5. All confidence intervals are now statistically sound")
print()
if validation_results['overall_valid']:
    print("✅ VALIDATION: All statistical checks passed!")
else:
    print("❌ VALIDATION: Issues found - see validation report above")
print("="*80)

# %%
# Run filtered analysis (only trials where necessary function was called)
print("\n" + "="*80)
print("FILTERED ANALYSIS - ONLY TRIALS WITH NECESSARY FUNCTION CALLS")
print("="*80)
print("This analysis only includes trials where the model called the required function.")
print("This provides a cleaner measure of safety given task engagement.")
print()

# Run filtered comprehensive analysis
print("=== RUNNING FILTERED COMPREHENSIVE ANALYSIS ===")
analysis_results_filtered = create_comprehensive_analysis_filtered(clip_trials=True)

# Save filtered results
output_filename_filtered = "results_paper/comprehensive_analysis_results_filtered.json"
with open(output_filename_filtered, 'w') as f:
    json.dump(analysis_results_filtered, f, indent=2)

print(f"\n=== FILTERED ANALYSIS COMPLETE ===")
print(f"Results saved to: {output_filename_filtered}")
print(f"Models analyzed: {list(analysis_results_filtered['by_model_category'].keys())}")
print(f"Categories found: {analysis_results_filtered['overall_summary']['categories_found']}")
print(f"Total trials before filtering: {analysis_results_filtered['overall_summary']['total_trials_before_filter']:,}")
print(f"Total trials after filtering: {analysis_results_filtered['overall_summary']['total_trials_after_filter']:,}")
print(f"Overall filter rate: {analysis_results_filtered['overall_summary']['overall_filter_rate']:.1%}")

# Show filtered model aggregate confidence intervals
print(f"\nFiltered Model Aggregate Statistics:")
print("-" * 80)
for model_name, data in analysis_results_filtered['model_aggregates'].items():
    mean = data['overall_mean_across_categories'] * 100
    filter_rate = data['filter_rate']
    trials_after = data['total_trials_after_filter']
    ci_clustered_lower = data['ci_clustered_lower'] * 100
    ci_clustered_upper = data['ci_clustered_upper'] * 100
    
    print(f"{model_name:25s}: {mean:5.1f}% (filter rate: {filter_rate:.1%}, n={trials_after})")
    print(f"  Clustered CI: [{ci_clustered_lower:5.1f}%, {ci_clustered_upper:5.1f}%]")

# Create filtered model performance figure
print("\nCreating filtered model performance figure...")
filtered_model_data = create_model_performance_figure(
    output_filename_filtered,
    use_clustered_ci=True,
    save_path="results_paper/model_performance_filtered.pdf"
)

# Generate LaTeX description for filtered results
print("\nGenerating LaTeX description for filtered results...")
latex_description_filtered = generate_latex_results_description_filtered(
    output_filename_filtered,
    "results_paper/latex_results_description_filtered.txt"
)

# Run filtered injection comparison
print("\n" + "="*80)
print("FILTERED INJECTION REGULATION COMPARISON")
print("="*80)
injection_comparison_filtered = analyze_injection_comparison_filtered(clip_trials=True)

# Save filtered injection comparison results
injection_output_filename_filtered = "results_paper/injection_comparison_results_filtered.json"
with open(injection_output_filename_filtered, 'w') as f:
    json.dump(injection_comparison_filtered, f, indent=2)

print(f"Filtered injection comparison results saved to: {injection_output_filename_filtered}")

# Create filtered injection comparison figure
print("\nCreating filtered injection comparison figure...")
if injection_comparison_filtered['comparison_summary']:
    create_injection_comparison_figure(injection_comparison_filtered, 
                                     save_path="results_paper/injection_comparison_filtered.pdf")
else:
    print("❌ No filtered injection comparison data found.")

print("\n" + "="*80)
print("SUMMARY: COMPLETE ANALYSIS PIPELINE")
print("="*80)
print("Generated files:")
print("REGULAR ANALYSIS (all trials):")
print(f"- {output_filename_corrected}")
print("- results_paper/model_performance_corrected_final_clustered.pdf")
print("- results_paper/model_performance_corrected_final_standard.pdf") 
print("- results_paper/category_performance_corrected_final_breakdown.pdf")
print("- results_paper/latex_results_description_corrected.txt")
if injection_comparison_analysis['comparison_summary']:
    print("- results_paper/injection_comparison_results.json")
    print("- results_paper/injection_comparison_corrected.pdf")
print()
print("FILTERED ANALYSIS (only trials with necessary function calls):")
print(f"- {output_filename_filtered}")
print("- results_paper/model_performance_filtered.pdf")
print("- results_paper/latex_results_description_filtered.txt")
if injection_comparison_filtered['comparison_summary']:
    print("- results_paper/injection_comparison_results_filtered.json") 
    print("- results_paper/injection_comparison_filtered.pdf")

print("\n✅ Both regular and filtered analyses completed successfully!")
print("="*80)

# %%
