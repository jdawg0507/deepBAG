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
            # Get top 5 matching endpoint names
            top_endpoints = self.retriever.retrieve_tools(keywords, top_k=5)
            if not top_endpoints:
                return {"error": "No matching endpoints found."}
                
            docs = []
            for ep_item in top_endpoints:
                # Handle both string endpoint names and dict representations
                if isinstance(ep_item, str):
                    ep_name = ep_item
                    info = self.endpoint_library.get(ep_name, {})
                elif isinstance(ep_item, dict):
                    ep_name = ep_item.get('name', 'Unknown')
                    info = ep_item
                else:
                    continue

                desc = info.get('description', f'Endpoint to fetch {ep_name} data.')
                params = info.get('parameters', {})
                required_list = info.get('required', [])
                
                param_strs = []
                if isinstance(params, dict):
                    for p_name, p_info in params.items():
                        req_label = "REQUIRED" if p_name in required_list else "OPTIONAL"
                        p_type = p_info.get("type", "str") if isinstance(p_info, dict) else "str"
                        param_strs.append(f"- {p_name} ({p_type}) [{req_label}]")
                elif isinstance(params, list):
                    for p in params:
                        if isinstance(p, dict):
                            p_name = p.get('name', 'param')
                            req_label = "REQUIRED" if p.get("required") else "OPTIONAL"
                            param_strs.append(f"- {p_name} ({p.get('type', 'str')}) [{req_label}]")
                        else:
                            param_strs.append(f"- {p} [OPTIONAL]")
                
                docs.append({
                    "endpoint_name": ep_name,
                    "description": desc,
                    "parameters": "\n".join(param_strs[:10]) if param_strs else "None"
                })
                
            return {"matching_endpoints": docs}
        except Exception as e:
            return {"error": str(e)}
