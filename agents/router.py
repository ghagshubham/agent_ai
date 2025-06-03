
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class RouterAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def route_task(self, task: str) -> List[str]:
        """Legacy method for backward compatibility"""
        analysis = self.analyze_task_requirements(task)
        return analysis.get("required_agents", ["code"])  # Default fallback
    
    def analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """Enhanced method to analyze task requirements intelligently"""
        
        # First, try AI-powered analysis
        ai_analysis = self._ai_analyze_task(task)
        
        # Then, apply rule-based analysis as backup/validation
        rule_analysis = self._rule_based_analysis(task)
        
        # Combine both analyses for best results
        final_analysis = self._combine_analyses(ai_analysis, rule_analysis, task)
        
        return final_analysis
    
    def _ai_analyze_task(self, task: str) -> Dict[str, Any]:
        """Use AI to analyze task requirements"""
        
        prompt = f"""
        Analyze this task and determine the execution strategy:
        
        Task: {task}
        
        Available agents:
        - research: For gathering information, web search, data collection, analysis
        - code: For generating algorithms, code implementations, programming solutions
        - visualization: For creating charts, graphs, dashboards, visual representations
        
        Analyze the task and respond in this exact JSON format:
        {{
            "required_agents": ["agent1", "agent2"],
            "execution_mode": "minimal|sequential|parallel",
            "skip_human_review": true|false,
            "dependencies": {{"agent2": ["agent1"]}},
            "reasoning": "Brief explanation of why these agents are needed"
        }}
        
        Guidelines:
        - Use "minimal" mode for single-agent tasks (code only, research only, etc.)
        - Use "sequential" mode when agents depend on each other
        - Use "parallel" mode when agents can run independently
        - Set skip_human_review to true for simple, single-agent tasks
        - Only include agents that are absolutely necessary
        
        Examples:
        - "Write a Python function" → {{"required_agents": ["code"], "execution_mode": "minimal", "skip_human_review": true}}
        - "Research AI trends" → {{"required_agents": ["research"], "execution_mode": "minimal", "skip_human_review": true}}
        - "Research data and create charts" → {{"required_agents": ["research", "visualization"], "execution_mode": "sequential", "dependencies": {{"visualization": ["research"]}}}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            import json
            try:
                # Extract JSON from response if it's wrapped in other text
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    
                    # Validate required fields
                    if "required_agents" in analysis and "execution_mode" in analysis:
                        # Filter valid agents only
                        valid_agents = [agent for agent in analysis["required_agents"] 
                                      if agent in ['research', 'code', 'visualization']]
                        analysis["required_agents"] = valid_agents
                        return analysis
            except json.JSONDecodeError:
                pass
                
        except Exception as e:
            print(f"AI Router analysis error: {e}")
        
        # Return empty dict if AI analysis fails
        return {}
    
    def _rule_based_analysis(self, task: str) -> Dict[str, Any]:
        """Rule-based analysis as fallback"""
        task_lower = task.lower()
        
        # Initialize analysis
        analysis = {
            "required_agents": [],
            "execution_mode": "minimal",
            "skip_human_review": True,
            "dependencies": {},
            "reasoning": "Rule-based analysis"
        }
        
        # Code-only patterns
        code_patterns = [
            "write code", "code only", "generate code", "create function", 
            "implement", "algorithm", "script", "program", "function",
            "coding", "develop", "build code", "write a function"
        ]
        
        # Research-only patterns
        research_patterns = [
            "research", "find information", "search", "investigate", 
            "study", "analyze data", "gather information", "look up",
            "find out", "explore", "examine"
        ]
        
        # Visualization-only patterns
        viz_patterns = [
            "visualize", "chart", "graph", "plot", "dashboard", 
            "create chart", "show graph", "display data", "visual",
            "diagram", "infographic"
        ]
        
        # Multi-agent patterns
        multi_patterns = {
            "research_code": ["research and code", "find data and implement", "analyze and program"],
            "code_viz": ["code and visualize", "implement and chart", "program and plot"],
            "research_viz": ["research and visualize", "find data and chart", "analyze and plot"],
            "all_three": ["complete analysis", "full implementation", "research code visualize", "end-to-end"]
        }
        
        # Check for single-agent patterns first
        if any(pattern in task_lower for pattern in code_patterns):
            # Check if it's ONLY code
            if not any(pattern in task_lower for pattern in research_patterns + viz_patterns):
                analysis["required_agents"] = ["code"]
                analysis["execution_mode"] = "minimal"
                analysis["skip_human_review"] = True
                return analysis
        
        if any(pattern in task_lower for pattern in research_patterns):
            # Check if it's ONLY research
            if not any(pattern in task_lower for pattern in code_patterns + viz_patterns):
                analysis["required_agents"] = ["research"]
                analysis["execution_mode"] = "minimal"
                analysis["skip_human_review"] = True
                return analysis
        
        if any(pattern in task_lower for pattern in viz_patterns):
            # Check if it's ONLY visualization
            if not any(pattern in task_lower for pattern in code_patterns + research_patterns):
                analysis["required_agents"] = ["visualization"]
                analysis["execution_mode"] = "minimal"
                analysis["skip_human_review"] = True
                return analysis
        
        # Check for multi-agent patterns
        if any(pattern in task_lower for pattern in multi_patterns["all_three"]):
            analysis["required_agents"] = ["research", "code", "visualization"]
            analysis["execution_mode"] = "sequential"
            analysis["dependencies"] = {"code": ["research"], "visualization": ["research", "code"]}
            analysis["skip_human_review"] = False
            return analysis
        
        if any(pattern in task_lower for pattern in multi_patterns["research_code"]):
            analysis["required_agents"] = ["research", "code"]
            analysis["execution_mode"] = "sequential"
            analysis["dependencies"] = {"code": ["research"]}
            analysis["skip_human_review"] = False
            return analysis
        
        if any(pattern in task_lower for pattern in multi_patterns["code_viz"]):
            analysis["required_agents"] = ["code", "visualization"]
            analysis["execution_mode"] = "sequential"
            analysis["dependencies"] = {"visualization": ["code"]}
            analysis["skip_human_review"] = False
            return analysis
        
        if any(pattern in task_lower for pattern in multi_patterns["research_viz"]):
            analysis["required_agents"] = ["research", "visualization"]
            analysis["execution_mode"] = "sequential"
            analysis["dependencies"] = {"visualization": ["research"]}
            analysis["skip_human_review"] = False
            return analysis
        
        # Fallback: detect individual agents
        agents = []
        if any(word in task_lower for word in ['data', 'information', 'find', 'search', 'study']):
            agents.append('research')
        if any(word in task_lower for word in ['code', 'function', 'script', 'program', 'algorithm']):
            agents.append('code')
        if any(word in task_lower for word in ['chart', 'graph', 'visual', 'plot', 'diagram']):
            agents.append('visualization')
        
        if agents:
            analysis["required_agents"] = agents
            if len(agents) > 1:
                analysis["execution_mode"] = "sequential"
                analysis["skip_human_review"] = False
            else:
                analysis["execution_mode"] = "minimal"
                analysis["skip_human_review"] = True
        else:
            # Ultimate fallback
            analysis["required_agents"] = ["code"]
            analysis["execution_mode"] = "minimal"
            analysis["skip_human_review"] = True
        
        return analysis
    
    def _combine_analyses(self, ai_analysis: Dict[str, Any], rule_analysis: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Combine AI and rule-based analyses for best results"""
        
        # If AI analysis is valid, use it as primary
        if ai_analysis and "required_agents" in ai_analysis and ai_analysis["required_agents"]:
            # Validate AI analysis with rule-based checks
            final_analysis = ai_analysis.copy()
            
            # Ensure execution mode is appropriate for agent count
            agent_count = len(final_analysis["required_agents"])
            if agent_count == 1 and final_analysis.get("execution_mode") != "minimal":
                final_analysis["execution_mode"] = "minimal"
                final_analysis["skip_human_review"] = True
            elif agent_count > 1 and final_analysis.get("execution_mode") == "minimal":
                final_analysis["execution_mode"] = "sequential"
                final_analysis["skip_human_review"] = False
            
            return final_analysis
        
        # Fallback to rule-based analysis
        return rule_analysis
    
    def get_execution_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the execution plan"""
        agents = analysis.get("required_agents", [])
        mode = analysis.get("execution_mode", "minimal")
        skip_review = analysis.get("skip_human_review", False)
        
        summary = f"Execution Plan:\n"
        summary += f"- Agents: {', '.join(agents)}\n"
        summary += f"- Mode: {mode}\n"
        summary += f"- Human review: {'Skipped' if skip_review else 'Required'}\n"
        
        if analysis.get("dependencies"):
            summary += f"- Dependencies: {analysis['dependencies']}\n"
        
        if analysis.get("reasoning"):
            summary += f"- Reasoning: {analysis['reasoning']}"
        
        return summary