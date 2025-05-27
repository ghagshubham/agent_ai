from typing import List
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class RouterAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def route_task(self, task: str) -> List[str]:
        """Determine which agents are needed for the task"""
        
        prompt = f"""
        Analyze this task and determine which agents are needed:
        Task: {task}
        
        Available agents:
        - research: For gathering information, web search, analysis
        - code: For generating algorithms, code implementations
        - visualization: For creating charts, graphs, visual representations
        
        Respond with only the agent names needed, separated by commas.
        Example: research,code,visualization
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            agent_list = response.choices[0].message.content.strip().split(',')
            return [agent.strip() for agent in agent_list if agent.strip() in ['research', 'code', 'visualization']]
        
        except Exception as e:
            print(f"Router error: {e}")
            # Fallback: analyze keywords
            task_lower = task.lower()
            agents = []
            
            if any(word in task_lower for word in ['analyze', 'research', 'study', 'impact', 'find']):
                agents.append('research')
            
            if any(word in task_lower for word in ['algorithm', 'code', 'implement', 'develop', 'program']):
                agents.append('code')
            
            if any(word in task_lower for word in ['visualiz', 'chart', 'graph', 'plot', 'diagram']):
                agents.append('visualization')
            
            return agents if agents else ['research']  # Default to research