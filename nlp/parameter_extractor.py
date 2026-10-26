"""
Module for extracting parameters from user queries using Gemini.
"""
import json
from utils.data_utils import clean_json_response  # Changed from relative import
from api.parameter_utils import get_player_id_from_name  # Changed from relative import

class ParameterExtractor:
    def __init__(self, model, parameter_library):
        self.model = model
        self.parameter_library = parameter_library
    
    def extract_parameters(self, query):
        """Extract parameters from the user query using Gemini"""
        param_summary = []
        for class_name, values in self.parameter_library.items():
            param_summary.append(f"class {class_name}:")
            example_values = list(values.items())[:5]
            for attr_name, attr_value in example_values:
                param_summary.append(f"    {attr_name} = \"{attr_value}\"")
            if len(values) > 5:
                param_summary.append(f"    # ... and {len(values) - 5} more values")
            param_summary.append("")
        
        param_prompt = f"""
        You are an NBA API expert analyzing this query: "{query}"
        
        Think like a basketball analyst:
        - What context would you naturally assume?
        - What timeframes make sense for this type of question?
        - What additional parameters would help provide a complete answer?
        
        Available NBA API parameters:
        {"\n".join(param_summary)}
        
        Return a JSON object with parameters that would give the most relevant and complete answer.
        """
        
        response = self.model.generate_content(
            param_prompt,
            generation_config={'temperature': 0.1}
        )
        
        try:
            response_text = clean_json_response(response.text)
            extracted_params = json.loads(response_text)
            
            if "Player" in extracted_params:
                player_id = get_player_id_from_name(extracted_params["Player"])
                if player_id:
                    extracted_params["player_id"] = player_id
            
            return {k: v for k, v in extracted_params.items() if v is not None}
            
        except Exception as e:
            print(f"Error extracting parameters: {str(e)}")
            return {}