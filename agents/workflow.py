from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict, Set
from .router import RouterAgent
from .research import ResearchAgent
from .code_gen import CodeAgent
from .visualizer import RealDataVisualizationAgent 

class AgentState(TypedDict):
    task: str
    user_id: str
    task_id: str
    messages: List[Dict[str, Any]]
    results: Dict[str, Any]
    human_feedback: Dict[str, Any]
    status: str
    requires_human_input: bool
    agent_plan: List[str]
    completed_agents: Set[str]
    pending_agents: Set[str]
    skip_human_review: bool
    execution_mode: str  # 'sequential', 'parallel', 'minimal'
    dependencies: Dict[str, List[str]]

def enhanced_router_node(state: AgentState) -> AgentState:
    """Intelligently route the task to only required agents"""
    router = RouterAgent()
    
    # Analyze task to determine required agents and dependencies
    analysis = router.analyze_task_requirements(state["task"])
    
    # Update state with analysis results
    state["agent_plan"] = analysis["required_agents"]
    state["execution_mode"] = analysis.get("execution_mode", "minimal")
    state["skip_human_review"] = analysis.get("skip_human_review", False)
    state["dependencies"] = analysis.get("dependencies", {})
    
    # Initialize tracking sets
    state["completed_agents"] = set()
    state["pending_agents"] = set(analysis["required_agents"])
    
    # Log routing decision
    state["messages"].append({
        "agent": "router",
        "message": f"Task analysis complete. Required agents: {', '.join(analysis['required_agents'])}",
        "execution_mode": analysis["execution_mode"],
        "dependencies": analysis.get("dependencies", {}),
        "skip_human_review": analysis.get("skip_human_review", False),
        "reasoning": analysis.get("reasoning", ""),
        "timestamp": "now"
    })
    
    return state

def should_execute_research(state: AgentState) -> str:
    """Determine if research should be executed"""
    if "research" in state["pending_agents"]:
        return "research"
    elif "code" in state["pending_agents"]:
        return "check_code"
    elif "visualization" in state["pending_agents"]:
        return "check_visualization"
    else:
        return "check_completion"

def research_node(state: AgentState) -> AgentState:
    """Execute research only if required"""
    if "research" in state["pending_agents"]:
        agent = ResearchAgent()
        result = agent.execute(state["task"])
        
        state["results"]["research"] = result
        state["completed_agents"].add("research")
        state["pending_agents"].discard("research")
        
        state["messages"].append({
            "agent": "research",
            "message": "Research completed successfully",
            "data": result,
            "timestamp": "now"
        })
    
    return state

def should_execute_code(state: AgentState) -> str:
    """Determine if code generation should be executed"""
    if "code" in state["pending_agents"]:
        # Check dependencies
        dependencies = state["dependencies"].get("code", [])
        if all(dep in state["completed_agents"] for dep in dependencies):
            return "code"
        else:
            # Dependencies not met, skip for now
            return "check_visualization"
    elif "visualization" in state["pending_agents"]:
        return "check_visualization"
    else:
        return "check_completion"

def code_node(state: AgentState) -> AgentState:
    """Execute code generation only if required"""
    if "code" in state["pending_agents"]:
        agent = CodeAgent()
        
        # Build context from completed agents
        context = {}
        if "research" in state["completed_agents"]:
            context["research_data"] = state["results"].get("research", {})
        
        result = agent.execute(state["task"], context)
        
        state["results"]["code"] = result
        state["completed_agents"].add("code")
        state["pending_agents"].discard("code")
        
        state["messages"].append({
            "agent": "code",
            "message": "Code generation completed successfully",
            "data": result,
            "timestamp": "now"
        })
    
    return state

def should_execute_visualization(state: AgentState) -> str:
    """Determine if visualization should be executed"""
    if "visualization" in state["pending_agents"]:
        # Check dependencies
        dependencies = state["dependencies"].get("visualization", [])
        if all(dep in state["completed_agents"] for dep in dependencies):
            return "visualization"
        else:
            # Dependencies not met, check if we can proceed anyway
            return "check_completion"
    else:
        return "check_completion"

def visualization_node(state: AgentState) -> AgentState:
    """Execute visualization only if required"""
    if "visualization" in state["pending_agents"]:
        agent = RealDataVisualizationAgent()
        
        # Build context from completed agents only
        context = {}
        if "research" in state["completed_agents"]:
            context["research"] = state["results"].get("research", {})
        if "code" in state["completed_agents"]:
            context["code"] = state["results"].get("code", {})
        
        result = agent.execute(state["task"], context)
        
        state["results"]["visualization"] = result
        state["completed_agents"].add("visualization")
        state["pending_agents"].discard("visualization")
        
        state["messages"].append({
            "agent": "visualization",
            "message": "Visualization completed successfully",
            "data": result,
            "timestamp": "now"
        })
    
    return state

def check_completion(state: AgentState) -> str:
    """Check if all required agents have completed"""
    if len(state["pending_agents"]) == 0:
        # All required agents completed
        if state.get("skip_human_review", False):
            return "finalize"
        else:
            return "human_checkpoint"
    else:
        # Some agents still pending - this might happen if dependencies aren't met
        # For now, proceed to completion
        state["messages"].append({
            "agent": "system",
            "message": f"Warning: Some agents still pending: {list(state['pending_agents'])}",
            "timestamp": "now"
        })
        return "finalize"

def human_checkpoint(state: AgentState) -> AgentState:
    """Conditional human review - only if needed"""
    if not state.get("skip_human_review", False):
        state["requires_human_input"] = True
        state["status"] = "awaiting_human_feedback"
        
        # Only show results for agents that actually executed
        results_summary = {}
        for agent in state["completed_agents"]:
            results_summary[agent] = bool(state["results"].get(agent))
        
        state["messages"].append({
            "agent": "system",
            "message": "Task completed, awaiting human review",
            "results_summary": results_summary,
            "executed_agents": list(state["completed_agents"]),
            "timestamp": "now"
        })
    else:
        # Skip human review
        state["messages"].append({
            "agent": "system",
            "message": "Human review skipped for simple task",
            "timestamp": "now"
        })
    
    return state

def should_continue_after_human(state: AgentState) -> str:
    """Determine next step after human checkpoint"""
    if state.get("skip_human_review", False):
        return "finalize"
    elif state.get("human_feedback") and state["human_feedback"].get("approved"):
        return "finalize"
    elif state.get("human_feedback") and not state["human_feedback"].get("approved"):
        # Human rejected - could implement retry logic here
        return "finalize"  # For now, finalize anyway
    elif state.get("requires_human_input") and not state.get("human_feedback"):
        return END  # Wait for human input
    else:
        return "finalize"

def finalize_node(state: AgentState) -> AgentState:
    """Finalize results with only executed agents"""
    state["status"] = "completed"
    state["requires_human_input"] = False

    # Compile final results - only include results from executed agents
    final_results = {}
    for agent in state["completed_agents"]:
        if agent in state["results"]:
            final_results[agent] = state["results"][agent]

    final_output = {
        "task": state["task"],
        "completion_time": "now",
        "agents_executed": list(state["completed_agents"]),
        "agents_planned": state["agent_plan"],
        "execution_mode": state["execution_mode"],
        "results": final_results,
        "human_feedback": state.get("human_feedback"),
        "performance_metrics": {
            "total_agents_planned": len(state["agent_plan"]),
            "total_agents_executed": len(state["completed_agents"]),
            "efficiency": len(state["completed_agents"]) / len(state["agent_plan"]) if state["agent_plan"] else 1.0
        }
    }

    state["results"]["final_output"] = final_output
    state["messages"].append({
        "agent": "system",
        "message": f"Task completed successfully. Executed agents: {', '.join(state['completed_agents'])}",
        "efficiency": f"{len(state['completed_agents'])}/{len(state['agent_plan'])} agents executed",
        "timestamp": "now"
    })

    return state

def create_dynamic_workflow():
    """Create the dynamic non-linear LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("router", enhanced_router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)
    workflow.add_node("visualization", visualization_node)
    workflow.add_node("human_checkpoint", human_checkpoint)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Dynamic conditional routing from router
    workflow.add_conditional_edges(
        "router",
        should_execute_research,
        {
            "research": "research",
            "check_code": "code",
            "check_visualization": "visualization",
            "check_completion": "human_checkpoint"
        }
    )

    # After research, determine next step
    workflow.add_conditional_edges(
        "research",
        should_execute_code,
        {
            "code": "code",
            "check_visualization": "visualization",
            "check_completion": "human_checkpoint"
        }
    )

    # After code, determine next step
    workflow.add_conditional_edges(
        "code",
        should_execute_visualization,
        {
            "visualization": "visualization",
            "check_completion": "human_checkpoint"
        }
    )

    # After visualization, check completion
    workflow.add_conditional_edges(
        "visualization",
        check_completion,
        {
            "human_checkpoint": "human_checkpoint",
            "finalize": "finalize"
        }
    )

    # Conditional edge from human checkpoint
    workflow.add_conditional_edges(
        "human_checkpoint",
        should_continue_after_human,
        {
            "finalize": "finalize",
            END: END
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()

# Backward compatibility - keep the old function name
def create_workflow():
    """Legacy function name for backward compatibility"""
    return create_dynamic_workflow()

# Utility function to execute workflow with task
def execute_task(task: str, user_id: str = "default", task_id: str = "auto") -> Dict[str, Any]:
    """Helper function to execute a task through the dynamic workflow"""
    
    workflow = create_dynamic_workflow()
    
    initial_state = {
        "task": task,
        "user_id": user_id,
        "task_id": task_id,
        "messages": [],
        "results": {},
        "human_feedback": {},
        "status": "initialized",
        "requires_human_input": False,
        "agent_plan": [],
        "completed_agents": set(),
        "pending_agents": set(),
        "skip_human_review": False,
        "execution_mode": "minimal",
        "dependencies": {}
    }
    
    try:
        result = workflow.invoke(initial_state)
        return result
    except Exception as e:
        print(f"Workflow execution error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "task": task
        }

