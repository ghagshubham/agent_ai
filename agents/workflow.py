from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict
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

def router_node(state: AgentState) -> AgentState:
    """Route the task to appropriate agents"""
    router = RouterAgent()
    plan = router.route_task(state["task"])
    
    state["agent_plan"] = plan
    state["messages"].append({
        "agent": "router",
        "message": f"Task routed to agents: {', '.join(plan)}",
        "timestamp": "now"
    })
    
    return state

def research_node(state: AgentState) -> AgentState:
    """Execute research if needed"""
    if "research" in state["agent_plan"]:
        agent = ResearchAgent()
        result = agent.execute(state["task"])
        state["results"]["research"] = result
        state["messages"].append({
            "agent": "research",
            "message": "Research completed",
            "data": result
        })
    return state

def code_node(state: AgentState) -> AgentState:
    """Execute code generation if needed"""
    if "code" in state["agent_plan"]:
        agent = CodeAgent()
        context = state["results"].get("research", {})
        result = agent.execute(state["task"], context)
        state["results"]["code"] = result
        state["messages"].append({
            "agent": "code",
            "message": "Code generation completed",
            "data": result
        })
    return state

def visualization_node(state: AgentState) -> AgentState:
    """Execute visualization if needed"""
    if "visualization" in state["agent_plan"]:
        agent = RealDataVisualizationAgent()
        context = {
            "research": state["results"].get("research", {}),
            "code": state["results"].get("code", {})
        }
        result = agent.execute(state["task"], context)
        state["results"]["visualization"] = result
        state["messages"].append({
            "agent": "visualization",
            "message": "Visualization completed",
            "data": result
        })
    return state

def human_checkpoint(state: AgentState) -> AgentState:
    """Pause for human review"""
    state["requires_human_input"] = True
    state["status"] = "awaiting_human_feedback"
    state["messages"].append({
        "agent": "system",
        "message": "Task completed, awaiting human review",
        "results_summary": {
            "research": bool(state["results"].get("research")),
            "code": bool(state["results"].get("code")),
            "visualization": bool(state["results"].get("visualization"))
        }
    })
    return state

def should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or end"""
    if state.get("human_feedback") and state["human_feedback"].get("approved"):
        return "finalize"
    elif state.get("requires_human_input") and not state.get("human_feedback"):
        return END
    else:
        return "finalize"

def finalize_node(state: AgentState) -> AgentState:
    """Finalize results"""
    state["status"] = "completed"
    state["requires_human_input"] = False

    # Compile final results without circular references
    final_output = {
        "task": state["task"],
        "completion_time": "now",
        "agents_used": state["agent_plan"],
        "results": {
            "research": state["results"].get("research"),
            "code": state["results"].get("code"),
            "visualization": state["results"].get("visualization")
        },
        "human_feedback": state.get("human_feedback")
    }

    state["results"]["final_output"] = final_output
    state["messages"].append({
        "agent": "system",
        "message": "Task finalized and completed"
    })

    return state

def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)
    workflow.add_node("visualization", visualization_node)
    workflow.add_node("human_checkpoint", human_checkpoint)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_edge("router", "research")
    workflow.add_edge("research", "code")
    workflow.add_edge("code", "visualization")
    workflow.add_edge("visualization", "human_checkpoint")

    # Conditional edge from human checkpoint
    workflow.add_conditional_edges(
        "human_checkpoint",
        should_continue,
        {
            "finalize": "finalize",
            END: END
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()
