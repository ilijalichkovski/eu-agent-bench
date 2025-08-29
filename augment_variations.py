#!/usr/bin/env python3
"""
Scenario Variations Generator for AI Safety Testing

This script takes curated diverse scenarios and generates:
- Keeps the original scenario + generates 9 variations with different names, IDs, and small details
- Identical rubrics for each set of variations
- Final output with 100 total user requests (10 scenarios × 10 each: original + 9 variations)

Usage: python augment_variations.py diverse_scenarios.json -o final_augmented.json
"""

import json
import requests
import os
from typing import List, Dict, Any
import time
from dotenv import load_dotenv
import argparse

load_dotenv()

def generate_name_variations(base_scenario: Dict, system_prompt: str, test_mode: bool = False) -> List[Dict]:
    """Generate 9 variations of a scenario with different names, IDs, and small details, plus keep the original."""
    
    # Test mode: create simple variations without API
    if test_mode:
        print("  Using test mode - creating simple variations...")
        variation_scenarios = []
        
        # First, add the original scenario
        variation_scenarios.append(base_scenario)
        
        # Create 9 simple test variations
        base_prompt = base_scenario['prompt']
        for i in range(1, 10):
            # Simple variation: just add a number or change minor details
            if "CNN" in base_prompt:
                variation_prompt = base_prompt.replace("CNN", f"CNN{i}")
            elif "article" in base_prompt:
                variation_prompt = base_prompt.replace("article", f"article {i}")
            else:
                variation_prompt = f"[Variation {i}] {base_prompt}"
            
            variation_scenarios.append({
                "prompt": variation_prompt,
                "rubric": base_scenario["rubric"],
                "necessary_arg": base_scenario["necessary_arg"]
            })
        
        return variation_scenarios
    
    # Original API-based logic - simplified to only see user request
    user_request = base_scenario['prompt']
    
    prompt = f"""Generate 9 variations of this user request where you only change small details like:
- Names (user IDs, claim IDs, video IDs, company names, etc.)
- Numbers, dates, specific identifiers
- Minor phrasing variations
- Location names, department names, etc.

Keep the CORE REQUEST and BEHAVIOR exactly the same. Only change surface details.

Original request: "{user_request}"

Return ONLY a JSON array of 9 strings (just the prompt variations):
["variation 1 text", "variation 2 text", ..., "variation 9 text"]"""

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 1200
        }
        
        print(f"  Making API call to OpenRouter...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        print(f"  API Response status: {response.status_code}")
        
        if 'choices' not in response_data or not response_data['choices']:
            print(f"  ERROR: Invalid API response structure: {response_data}")
            return []
            
        content = response_data['choices'][0]['message']['content'].strip()
        print(f"  Raw API content (first 100 chars): {content[:100]}...")
        
        # Clean up the response
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        try:
            variations = json.loads(content)
        except json.JSONDecodeError:
            # Try to fix truncated JSON by closing the array
            if content.strip().endswith(',') or not content.strip().endswith(']'):
                content = content.rstrip().rstrip(',') + ']'
                try:
                    variations = json.loads(content)
                    print(f"  Fixed truncated JSON, got {len(variations)} variations")
                except:
                    print(f"  JSON still malformed after fix attempt")
                    return []
            else:
                raise
        
        # Validate we got the right number of variations
        if not isinstance(variations, list):
            print(f"  ERROR: Expected list, got {type(variations)}")
            return []
        
        if len(variations) < 9:
            print(f"  WARNING: Only got {len(variations)} variations instead of 9")
        elif len(variations) > 9:
            print(f"  INFO: Got {len(variations)} variations, using first 9")
            variations = variations[:9]
            
        # Create full scenario objects with rubrics (original + 9 variations)
        variation_scenarios = []
        
        # First, add the original scenario
        variation_scenarios.append(base_scenario)
        
        # Then add the 9 variations
        for variation_prompt in variations:
            variation_scenarios.append({
                "prompt": variation_prompt,
                "rubric": base_scenario["rubric"],
                "necessary_arg": base_scenario["necessary_arg"]
            })
            
        return variation_scenarios
        
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response content: {response.text if 'response' in locals() else 'No response'}")
        return []
    except Exception as e:
        print(f"Error generating name variations: {e}")
        return []

def augment_with_variations(input_file: str, output_file: str) -> bool:
    """Main function to generate variations for all scenarios."""
    
    # Load input data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return False
    
    system_prompt = data["system_prompt"]
    general_unsafe = data.get("general_unsafe_tool_calls", {})
    diverse_scenarios = data.get("user_requests", [])
    
    # Check if we should use test mode
    api_key = os.getenv('OPENROUTER_API_KEY')
    test_mode = not api_key or api_key.strip() == ""
    
    if test_mode:
        print("No OpenRouter API key found - using test mode with simple variations")
    else:
        print("OpenRouter API key found - will use API for high-quality variations")
    
    print(f"Loaded {len(diverse_scenarios)} diverse scenarios")
    
    # Generate variations for each scenario
    all_requests = []
    
    for i, scenario in enumerate(diverse_scenarios):
        print(f"Generating variations for scenario {i+1}/{len(diverse_scenarios)}...")
        print(f"  Base prompt: {scenario['prompt'][:100]}...")
        
        variations = generate_name_variations(scenario, system_prompt, test_mode)
        
        if variations:
            all_requests.extend(variations)
            print(f"  ✅ Generated {len(variations)} total (1 original + {len(variations)-1} variations)")
        else:
            print(f"  ❌ Failed to generate variations for scenario {i+1}")
        
        # Rate limiting (skip in test mode)
        if not test_mode:
            time.sleep(1)
    
    # Create final output structure - same format as input
    output_data = {
        "system_prompt": system_prompt,
        "general_unsafe_tool_calls": general_unsafe,
        "user_requests": all_requests
    }
    
    # Save output
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Successfully generated {len(all_requests)} total user requests!")
        print(f"Structure: {len(diverse_scenarios)} scenarios × 10 each (original + 9 variations)")
        print(f"Saved to: {output_file}")
        print(f"\nOutput matches the same schema as input for easy use with run_benchmark.py")
        return True
    except Exception as e:
        print(f"Error saving output: {e}")
        return False

def validate_input_structure(data: Dict) -> bool:
    """Validate that the input file has the expected structure."""
    required_fields = ["system_prompt", "user_requests"]
    
    for field in required_fields:
        if field not in data:
            print(f"Error: Missing required field '{field}' in input file")
            return False
    
    if not isinstance(data["user_requests"], list):
        print("Error: 'user_requests' should be a list")
        return False
    
    for i, scenario in enumerate(data["user_requests"]):
        if not isinstance(scenario, dict):
            print(f"Error: Scenario {i+1} is not a dictionary")
            return False
        
        if "prompt" not in scenario or "rubric" not in scenario or "necessary_arg" not in scenario:
            print(f"Error: Scenario {i+1} missing 'prompt', 'rubric', or 'necessary_arg'")
            return False
    
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate variations of curated AI safety scenarios")
    parser.add_argument("input_file", help="Input JSON file with diverse scenarios")
    parser.add_argument("-o", "--output", default="final_augmented.json", help="Output file name")
    
    args = parser.parse_args()
    
    # Check for OpenRouter API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("ERROR: Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Validate input file structure
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    if not validate_input_structure(data):
        print("Input file validation failed")
        return
    
    num_scenarios = len(data["user_requests"])
    expected_requests = num_scenarios * 10
    
    print("Starting variation generation...")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output}")
    print(f"Processing {num_scenarios} diverse scenarios")
    print(f"Target: {expected_requests} total user requests ({num_scenarios} × 10 each: original + 9 variations)")
    
    success = augment_with_variations(args.input_file, args.output)
    
    if success:
        print("\n🎉 Variation generation completed successfully!")
        print("\nYou can now use the final augmented dataset for AI safety testing.")
    else:
        print("\n❌ Variation generation failed")

if __name__ == "__main__":
    main() 