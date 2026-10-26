"""
Main entry point for the NBA NLP Analyzer application.
This file handles user interaction and calls the analyzer.
"""
import sys
import json
from nba_nlp_analyzer_modular import NBAQueryAnalyzer

def main():
    import os
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please export it in your shell: export GEMINI_API_KEY='your-key'")
        sys.exit(1)
    
    # Initialize the analyzer
    print("Initializing NBA Query Analyzer...")
    analyzer = NBAQueryAnalyzer(api_key=api_key)
    print("Initialization complete!")

    
    # Main interaction loop
    while True:
        # Get user query
        print("\n" + "="*50)
        print("Enter your NBA stats question (or 'quit' to exit):")
        query = input("> ")
        
        # Check if user wants to quit
        if query.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the NBA Query Analyzer!")
            break
        
        # Process the query with planning step
        print("\nAnalyzing your question...")
        print("Planning data retrieval strategy...")
        response = analyzer.process_query_with_planning(query)
        
        # Display the response
        print("\n" + "="*50)
        print("ANSWER:")
        print(response)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)