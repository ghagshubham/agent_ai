from typing import Dict, Any
from tavily import TavilyClient
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class ResearchAgent:
    def __init__(self):
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute research for the given task"""
        
        try:
            # Extract research queries from task
            search_queries = self._extract_search_queries(task)
            
            # Perform searches
            search_results = []
            for query in search_queries[:3]:  # Limit to 3 queries
                try:
                    result = self.tavily.search(
                        query=query,
                        search_depth="advanced",
                        max_results=5
                    )
                    search_results.append({
                        "query": query,
                        "results": result.get("results", [])
                    })
                except Exception as e:
                    print(f"Search error for query '{query}': {e}")
            
            # Synthesize findings
            synthesis = self._synthesize_findings(task, search_results)
            
            return {
                "search_queries": search_queries,
                "raw_results": search_results,
                "synthesis": synthesis,
                "key_findings": self._extract_key_findings(synthesis),
                "status": "completed"
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
            }
    
    def _extract_search_queries(self, task: str) -> list:
        """Extract relevant search queries from the task"""
        
        prompt = f"""
        Given the following research task, identify 2–3 research queries that explore different dimensions (e.g., impact, trends, solutions) and include at least one focused on recent developments or standards.

        Task: {task}
        
        Return only the search queries, one per line.
        Make them specific and research-focused.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return queries[:3]
        
        except Exception:
            # Fallback queries based on task keywords
            if "quantum computing" in task.lower() and "cybersecurity" in task.lower():
                return [
                    "quantum computing cybersecurity threats",
                    "quantum resistant cryptography algorithms",
                    "post-quantum cryptography vulnerable sectors"
                ]
            else:
                return [task[:100]]  # Use first 100 chars as query
    
    def _synthesize_findings(self, task: str, search_results: list) -> str:
        """Synthesize research findings into a coherent summary"""
        
        # Compile all search content
        all_content = ""
        for result_set in search_results:
            for result in result_set.get("results", []):
                all_content += f"{result.get('title', '')}: {result.get('content', '')}\n"
        
        prompt = f"""
        You are a research analyst. Based on the task and findings below, write a structured, expert-level summary.

        Task: {task}

        Content extracted from search results:
        {all_content[:4000]}

        Your synthesis should include:
        1. Key Insights Relevant to the Task
        2. Important Findings and Data Points
        3. Current State and Trends (include years or timelines if mentioned)
        4. Implications and Strategic Considerations

        Guidelines:
        - Avoid repetition.
        - Use bullet points where appropriate.
        - Keep each section informative and tight.

        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception:
            return "Research synthesis unavailable due to processing error."
    
    def _extract_key_findings(self, synthesis: str) -> list:
        """Extract key bullet points from synthesis"""
        
        prompt = f"""
        From this research synthesis, extract the 5–7 most actionable and insightful findings. Each point should:
        - Be concise
        - Mention relevant stats or timelines (if present)
        - Avoid vague or obvious statements
        {synthesis}
        Return as bullet points starting with a dash (-).

        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            findings = [f.strip()[1:].strip() for f in response.choices[0].message.content.split('\n') if f.strip().startswith('-')]
            return findings
        
        except Exception:
            return ["Key findings extraction unavailable"]
    
