import os
import sys

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nba_nlp_analyzer_modular import NBAQueryAnalyzer

def test_agentic_pipeline():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Notice: GEMINI_API_KEY is not set. Please export GEMINI_API_KEY='your-key' to run live queries.")
        return
    
    analyzer = NBAQueryAnalyzer(api_key=api_key)

    
    # 1. Simple lookup
    query1 = "What are LeBron James' stats from his last game?"
    print("\n" + "="*50)
    print(f"Testing Query 1: {query1}")
    print("="*50)
    result1 = analyzer.process_query_agentic(query1)
    print(f"\nFinal Result:\n{result1}")
    
    # 2. Complex Multi-step Comparison
    query2 = "Who scored more points in their last 5 games, LeBron James or Stephen Curry?"
    print("\n" + "="*50)
    print(f"Testing Query 2 (Complex): {query2}")
    print("="*50)
    result2 = analyzer.process_query_agentic(query2)
    print(f"\nFinal Result:\n{result2}")

if __name__ == "__main__":
    test_agentic_pipeline()
