"""
Utility functions for data processing in the NBA NLP Analyzer.
"""
import json
import pandas as pd

def dataframe_to_json(df):
    """Convert a pandas DataFrame to a JSON-serializable format"""
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records"))

def process_api_dataframes(dataframes):
    """Process a list of dataframes from an NBA API response"""
    result = {}
    for i, df in enumerate(dataframes):
        print(f"  Processing dataframe {i} with {len(df)} rows and {len(df.columns)} columns")
        if len(df) > 0:
            print(f"  Column names: {', '.join(df.columns.tolist())}")
        result[f"table_{i}"] = dataframe_to_json(df)
    
    return result

def clean_json_response(response_text):
    """Clean a JSON response text from Gemini to handle potential markdown formatting"""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "", 1)
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "", 1)
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text.rsplit("```", 1)[0]
    return cleaned_text.strip()

def print_sample_data(data, max_length=200):
    """Print a sample of the data for debugging"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {len(value)} records")
                print(f"  Sample data: {json.dumps(value[0], indent=2)[:max_length]}...")
                if max_length < len(json.dumps(value[0], indent=2)):
                    print("  ...")
            else:
                print(f"  {key}: {value}")
    elif isinstance(data, list) and len(data) > 0:
        print(f"  {len(data)} records")
        print(f"  Sample data: {json.dumps(data[0], indent=2)[:max_length]}...")
        if max_length < len(json.dumps(data[0], indent=2)):
            print("  ...")
    else:
        print(f"  {data}")