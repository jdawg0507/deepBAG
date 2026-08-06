analyzer backup
def process_query(self, query, context=None):
        """Process a user query and return a response"""
        try:
            print("\n=== Query ===")
            print(query)
            
            # Step 1: Extract parameters from the query
            extracted_params = self.parameter_extractor.extract_parameters(query)
            print("\n=== Extracted Parameters ===")
            print(json.dumps(extracted_params, indent=2))
            
            # Step 2: Identify relevant endpoints
            relevant_endpoints = self.endpoint_selector.identify_endpoints(query, extracted_params)
            print("\n=== Relevant Endpoints ===")
            print(json.dumps(relevant_endpoints, indent=2))
            
            # Step 3: Execute initial API calls
            api_results = {}
            for endpoint in relevant_endpoints:
                print(f"\n=== Preparing API Call for {endpoint} ===")
                call_params = self.endpoint_selector.prepare_api_call(endpoint, extracted_params, query)
                print(f"Parameters: {json.dumps(call_params, indent=2)}")
                
                print(f"\n=== Executing API Call for {endpoint} ===")
                result = execute_api_call(endpoint, call_params)
                api_results[endpoint] = result
            
            # Step 4: Check if we need additional data
            supplemental_prompt = f"""
            Based on the query: "{query}"
            And the current data we have:
            {json.dumps(api_results, indent=2)}
            
            Do we need additional data to fully answer this query?
            If yes, specify what additional data we need and which NBA API endpoint would provide it.
            Return a JSON object with this structure:
            {{
                "needs_more_data": true/false,
                "reason": "explanation of why more data is needed",
                "additional_endpoints": ["endpoint1", "endpoint2"],
                "additional_parameters": {{"param1": "value1", "param2": "value2"}}
            }}
            """
            
            supplemental_response = self.model.generate_content(
                supplemental_prompt,
                generation_config={'temperature': 0.3}
            )
            
            try:
                supplemental_info = json.loads(supplemental_response.text)
                if supplemental_info.get("needs_more_data", False):
                    print("\n=== Making Supplemental API Calls ===")
                    for endpoint in supplemental_info.get("additional_endpoints", []):
                        if endpoint not in api_results:
                            print(f"\nFetching additional data from {endpoint}")
                            params = {**extracted_params, **supplemental_info.get("additional_parameters", {})}
                            result = execute_api_call(endpoint, params)
                            api_results[endpoint] = result
            except Exception as e:
                print(f"Error processing supplemental data request: {str(e)}")
            
            # Step 5: Format the final response
            print("\n=== Formatting Response ===")
            response = self.response_formatter.format_response(query, api_results, extracted_params)
            
            print("\n=== Final Response ===")
            print(response)
            return response
                
        except Exception as e:
            print(f"\n=== Error Details ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def process_query_with_planning(self, query, context=None):
        """Process a user query with a planning step before making API calls"""
        try:
            print("\n=== Query ===")
            print(query)
            
            # Step 1: Extract parameters from the query
            extracted_params = self.parameter_extractor.extract_parameters(query)
            print("\n=== Extracted Parameters ===")
            print(json.dumps(extracted_params, indent=2))
            
            # Step 2: Plan data retrieval strategy
            print("\n=== Planning Data Retrieval Strategy ===")
            
            # Create a detailed summary of available endpoints for the planning prompt
            endpoint_summary = {}
            for name, info in self.endpoint_library.items():
                endpoint_summary[name] = {
                    "description": info["description"],
                    "required_params": info["required"],
                    "valid_params": list(info.get("parameters", {}).keys()),
                    "expected_output": info.get("expected_output", {})  # Include expected output fields
                }
            
            planning_prompt = f"""
            I need to answer this NBA stats query: "{query}"
            
            Based on the query, I've extracted these parameters:
            {json.dumps(extracted_params, indent=2)}
            
            I need to plan what data to retrieve from the NBA API to fully answer this query.
            
            Here are the available NBA API endpoints with their valid parameters and expected outputs:
            {json.dumps(endpoint_summary, indent=2)}
            
            Think step by step:
            1. What specific information do I need to answer this query?
            2. Which endpoints would provide this information?
            3. What parameters do I need for each endpoint? ONLY USE PARAMETERS LISTED IN "valid_params" FOR EACH ENDPOINT.
            4. Do I need to make multiple API calls to get all the necessary data?
            5. What additional filtering or analysis will I need to perform on the data?
            
            Return a JSON object with this structure:
            {{
                "reasoning": "Your step-by-step reasoning about what data is needed",
                "data_plan": [
                    {{
                        "endpoint": "EndpointName",
                        "purpose": "Why you need this data",
                        "parameters": {{"param1": "value1", "param2": "value2"}} // ONLY USE VALID PARAMETERS FOR THIS ENDPOINT
                    }},
                    // Additional endpoints as needed
                ]
            }}
            """
            
            planning_response = self.model.generate_content(
                planning_prompt,
                generation_config={'temperature': 0.3}
            )
            
            try:
                # Clean the response text to handle potential markdown formatting
                from utils.data_utils import clean_json_response
                response_text = clean_json_response(planning_response.text)
                
                data_plan = json.loads(response_text)
                print("Data Retrieval Plan:")
                print(json.dumps(data_plan.get("reasoning", "No reasoning provided"), indent=2))
                
                # Step 3: Execute API calls based on the plan
                api_results = {}
                for step in data_plan.get("data_plan", []):
                    endpoint = step.get("endpoint")
                    purpose = step.get("purpose")
                    params = step.get("parameters", {})
                    
                    # Validate parameters against endpoint requirements
                    if endpoint in self.endpoint_library:
                        endpoint_info = self.endpoint_library[endpoint]
                        valid_params = list(endpoint_info.get("parameters", {}).keys())
                        
                        # Filter out invalid parameters
                        filtered_params = {k: v for k, v in params.items() if k in valid_params}
                        
                        # Check for missing required parameters
                        required_params = endpoint_info.get("required", [])
                        missing_params = [p for p in required_params if p not in filtered_params]
                        
                        # Handle missing required parameters
                        if missing_params:
                            print(f"Missing required parameters for {endpoint}: {missing_params}")
                            # Use the endpoint selector to handle missing parameters
                            filtered_params = self.endpoint_selector._handle_missing_parameters(
                                endpoint, missing_params, filtered_params, 
                                extracted_params, query, endpoint_info
                            )
                        
                        print(f"\n=== Executing API Call for {endpoint} ===")
                        print(f"Purpose: {purpose}")
                        print(f"Parameters: {json.dumps(filtered_params, indent=2)}")
                        
                        result = execute_api_call(endpoint, filtered_params)
                    else:
                        print(f"\n=== ERROR: Unknown endpoint {endpoint} ===")
                        result = {"error": f"Unknown endpoint: {endpoint}"}
                    
                    # Check if there was an error
                    if "error" in result:
                        print(f"  API call failed: {result['error']}")
                    else:
                        print(f"  API call successful")
                        # Print a sample of the data
                        for table_name, table_data in result.items():
                            print(f"  {table_name}: {len(table_data)} records")
                            if len(table_data) > 0:
                                print(f"  Sample data: {json.dumps(table_data[0], indent=2)[:200]}...")
                    
                    api_results[endpoint] = result
                
                # Step 4: Format the response
                print("\n=== Formatting Response ===")
                response = self.response_formatter.format_response(query, api_results, extracted_params)
                
                print("\n=== Final Response ===")
                print(response)
                return response
                
            except json.JSONDecodeError as e:
                print("Error parsing JSON response from planning step:")
                print(planning_response.text)
                print(f"JSON error: {str(e)}")
                # Fall back to regular processing
                print("Falling back to standard query processing...")
                return self.process_query(query, context)
                
        except Exception as e:
            print(f"\n=== Error Details ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}