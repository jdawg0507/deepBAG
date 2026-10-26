from google.ai.generativelanguage import FunctionDeclaration, Type, Tool

class EndpointToolBuilder:
    def __init__(self, endpoint_library):
        self.endpoint_library = endpoint_library

    def build_tool(self, endpoint_name):
        """Builds a Gemini FunctionDeclaration for a specific NBA API endpoint"""
        info = self.endpoint_library.get(endpoint_name)
        if not info:
            return None
            
        properties = {}
        for param_name, param_info in info.get("parameters", {}).items():
            properties[param_name] = {
                "type": Type.STRING,
                "description": param_info.get("description", f"{param_name} parameter")
            }
            # Enums might be too large, but if you want to include them:
            # if param_info.get("enum"):
            #    properties[param_name]["enum"] = [str(x) for x in param_info["enum"]][:20]

        return FunctionDeclaration(
            name=f"call_{endpoint_name}",
            description=info.get("description", f"Fetches data from the {endpoint_name} endpoint."),
            parameters={
                "type": Type.OBJECT,
                "properties": properties,
                "required": info.get("required", [])
            }
        )

    def build_tools_list(self, endpoint_names):
        """Builds a list of FunctionDeclarations from a list of endpoint names"""
        declarations = []
        for name in endpoint_names:
            decl = self.build_tool(name)
            if decl:
                declarations.append(decl)
        return declarations
