"""
Test program to output NBA API endpoint information and compare with current implementation.
"""
import json
import sys
from api.endpoint_utils import build_endpoint_library, inspect_endpoint_class, load_endpoint_class
from nba_api.stats import endpoints

def main():
    print("Building endpoint library...")
    endpoint_library = build_endpoint_library()
    print(f"Found {len(endpoint_library)} endpoints")
    
    # Output detailed information for each endpoint
    print("\nGenerating detailed endpoint information...")
    
    # Choose output format
    output_format = input("Output format (1=console, 2=json file): ")
    
    if output_format == "2":
        output_file = input("Output file path (default: endpoint_details.json): ") or "endpoint_details.json"
        
        # Collect all endpoint details
        endpoint_details = {}
        for name, info in endpoint_library.items():
            endpoint_details[name] = {
                "description": info.get("description", ""),
                "url": info.get("url", ""),
                "required_params": info.get("required", []),
                "optional_params": [p for p in info.get("parameters", {}).keys() if p not in info.get("required", [])],
                "expected_output": info.get("expected_output", {})
            }
            
            # Load the actual endpoint class to check for expected_data
            endpoint_class = load_endpoint_class(name)
            if endpoint_class and hasattr(endpoint_class, 'expected_data'):
                endpoint_details[name]["expected_data_from_class"] = endpoint_class.expected_data
            
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(endpoint_details, f, indent=2)
        print(f"Endpoint details written to {output_file}")
        
    else:
        # Print to console
        for name, info in endpoint_library.items():
            print(f"\n{'='*50}")
            print(f"ENDPOINT: {name}")
            print(f"Description: {info.get('description', '')}")
            print(f"URL: {info.get('url', '')}")
            print(f"Required parameters: {info.get('required', [])}")
            
            optional_params = [p for p in info.get("parameters", {}).keys() if p not in info.get("required", [])]
            print(f"Optional parameters: {optional_params}")
            
            print("Expected output:")
            for output_name, output_fields in info.get("expected_output", {}).items():
                print(f"  {output_name}: {output_fields[:5]}{'...' if len(output_fields) > 5 else ''}")
            
            # Check if the endpoint class has expected_data
            endpoint_class = load_endpoint_class(name)
            if endpoint_class and hasattr(endpoint_class, 'expected_data'):
                print("Expected data from class:")
                for data_name, data_fields in endpoint_class.expected_data.items():
                    print(f"  {data_name}: {data_fields[:5]}{'...' if len(data_fields) > 5 else ''}")
    
    # Compare with what's available in the nba_api package
    print("\nComparing with available endpoints in nba_api package...")
    available_endpoints = []
    for module_info in endpoints.__all__:
        if not module_info.startswith('_'):
            available_endpoints.append(module_info)
    
    missing_endpoints = [ep for ep in available_endpoints if ep not in endpoint_library]
    if missing_endpoints:
        print(f"\nMissing endpoints in our implementation ({len(missing_endpoints)}):")
        for ep in missing_endpoints:
            print(f"  - {ep}")
    else:
        print("\nAll endpoints from nba_api are included in our implementation.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)