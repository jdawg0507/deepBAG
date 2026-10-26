import json

def get_meta_tools_declarations():
    return [
        {
            "name": "search_api_docs",
            "description": "Search the library of 100+ NBA API endpoints for documentation on how to fetch specific stats (e.g. 'tracking data', 'player logs', 'team stats'). Returns endpoint names and their required parameter schemas.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "keywords": {
                        "type": "STRING",
                        "description": "Keywords describing the data you need (e.g. 'defensive tracking' or 'playoff box scores')."
                    }
                },
                "required": ["keywords"]
            }
        },
        {
            "name": "call_nba_api",
            "description": "Execute a specific NBA API endpoint after you have read its documentation via search_api_docs. Provide the exact endpoint name and a JSON dictionary of parameters.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "endpoint_name": {
                        "type": "STRING",
                        "description": "The exact name of the endpoint to call (e.g. 'LeagueDashPlayerStats')."
                    },
                    "parameters": {
                        "type": "STRING",
                        "description": "A JSON string containing the required parameters (e.g. '{\"Season\": \"2023-24\", \"MeasureType\": \"Advanced\"}')."
                    }
                },
                "required": ["endpoint_name", "parameters"]
            }
        }
    ]

class MetaToolExecutor:
    def __init__(self, endpoint_library):
        self.endpoint_library = endpoint_library
        from nlp.tool_retriever import EndpointToolRetriever
        self.retriever = EndpointToolRetriever(endpoint_library)
        from nlp.tool_builder import EndpointToolBuilder
        self.builder = EndpointToolBuilder(endpoint_library)
        
    def search_api_docs(self, keywords: str):
        try:
            # Get top 5 matching endpoints
            top_endpoints = self.retriever.retrieve_tools(keywords, top_k=5)
            if not top_endpoints:
                return {"error": "No matching endpoints found."}
                
            docs = []
            for ep in top_endpoints:
                # Format a clean documentation string
                name = ep.get('name', 'Unknown')
                desc = ep.get('description', '')
                params = ep.get('parameters', [])
                
                param_strs = []
                for p in params:
                    required = "REQUIRED" if p.get("required") else "OPTIONAL"
                    param_strs.append(f"- {p.get('name')} ({p.get('type')}) [{required}]")
                
                docs.append({
                    "endpoint_name": name,
                    "description": desc,
                    "parameters": "\n".join(param_strs) if param_strs else "None"
                })
                
            return {"matching_endpoints": docs}
        except Exception as e:
            return {"error": str(e)}
