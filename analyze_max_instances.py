#!/usr/bin/env python3
"""
Script to analyze the maximum instance number reached in each results JSON file.
"""

import json
import os
import re
from pathlib import Path

def extract_info_from_filename(filename):
    """Extract model and category from filename."""
    # Pattern: clean_{model}_{category}_{timestamp}.json
    pattern = r'clean_(.+?)_(.+?)_\d{8}_\d{6}\_.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def get_max_instance(filepath):
    """Get the maximum instance number from a JSON results file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        max_instance = -1
        for result in data.get('results', []):
            instance = result.get('instance', -1)
            max_instance = max(max_instance, instance)
        
        return max_instance
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return -1

def main():
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("Results directory not found!")
        return
    
    results = []
    
    # Process all JSON files in results directory
    for json_file in results_dir.glob('clean_*.json'):
        model, category = extract_info_from_filename(json_file.name)
        if model and category:
            max_instance = get_max_instance(json_file)
            results.append({
                'model': model,
                'category': category,
                'max_instance': max_instance,
                'filename': json_file.name
            })
    
    # Sort results by model, then by category
    results.sort(key=lambda x: (x['model'], x['category']))
    
    # Print results
    print(f"{'Model':<25} {'Category':<20} {'Max Instance':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model']:<25} {result['category']:<20} {result['max_instance']:<12}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"Total files analyzed: {len(results)}")
    
    if results:
        max_overall = max(r['max_instance'] for r in results if r['max_instance'] >= 0)
        min_overall = min(r['max_instance'] for r in results if r['max_instance'] >= 0)
        print(f"Highest instance reached: {max_overall}")
        print(f"Lowest instance reached: {min_overall}")
        
        # Find the file with highest instance
        best_result = max(results, key=lambda x: x['max_instance'])
        print(f"Best performer: {best_result['model']} on {best_result['category']} (instance {best_result['max_instance']})")

if __name__ == "__main__":
    main() 