from tavily import TavilyClient
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class SearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform web search using Tavily"""
        try:
            result = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            return {
                "success": True,
                "query": query,
                "results": result.get("results", []),
                "answer": result.get("answer", "")
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": []
            }
    
    def multi_search(self, queries: List[str], max_results: int = 3) -> List[Dict[str, Any]]:
        """Perform multiple searches"""
        results = []
        for query in queries:
            result = self.search(query, max_results)
            results.append(result)
        return results