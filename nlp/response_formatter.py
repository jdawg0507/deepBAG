"""
Module for formatting API results into user-friendly responses using Gemini.
"""
import json

class ResponseFormatter:
    def __init__(self, model):
        self.model = model
    
    def format_response(self, query, api_results, extracted_params):
        """Format the API results into a user-friendly response"""
        # Check if we have any successful API calls
        has_successful_call = False
        for endpoint, result in api_results.items():
            if "error" not in result:
                has_successful_call = True
                break
        
        # Extract player name for reference
        player_name = extracted_params.get("player_name", "the player")
        
        format_prompt = f"""
        The user asked: "{query}"
        
        Based on the query, we extracted these parameters:
        {json.dumps(extracted_params, indent=2)}
        
        Here are the results from the NBA API:
        {json.dumps(api_results, indent=2)}
        
        I need you to analyze this data thoroughly and answer the user's question.
        
        Important instructions:
        1. Be creative and resourceful with the data. If the answer isn't directly available, find ways to derive it.
        
        2. You may need to:
           - Perform calculations on the data (sums, averages, percentages, etc.)
           - Combine data from multiple endpoints
           - Filter data based on criteria in the query
           - Look for patterns or trends in the data
        
        3. If you need to make calculations:
           - Clearly explain your methodology
           - Show the key numbers that led to your conclusion
        
        4. If the data is incomplete or doesn't fully answer the query:
           - Provide the best answer possible with the available data
           - Explain what additional data would be needed for a complete answer
        
        5. Format your response in a user-friendly way with relevant statistics.
        
        6. Don't just say there's no data - try to extract whatever insights you can from what's available.
        """
        
        response = self.model.generate_content(
            format_prompt,
            generation_config={'temperature': 0.7}
        )
        
        return response.text