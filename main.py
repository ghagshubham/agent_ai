from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from agents.workflow import create_workflow
from tools.memory import MemoryStore

app = FastAPI(title="Multi-Agent AI Platform", version="1.0.0")
memory = MemoryStore()

class TaskRequest(BaseModel):
    task: str
    user_id: str = "demo_user"

class HumanFeedback(BaseModel):
    task_id: str
    approved: bool
    feedback: str = ""
    modifications: Dict[str, Any] = {}

@app.get("/")
async def root():
    return {"message": "Multi-Agent AI Platform is running"}

@app.post("/execute_task")
async def execute_task(request: TaskRequest):
    try:
        # Create workflow and execute
        workflow = create_workflow()
        
        # Initial state
        initial_state = {
            "task": request.task,
            "user_id": request.user_id,
            "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "messages": [],
            "results": {},
            "human_feedback": None,
            "status": "processing"
        }
        
        # Execute workflow
        result = workflow.invoke(initial_state)
        
        # Store in memory
        memory.store_task(result["task_id"], result)
        
        return {
            "task_id": result["task_id"],
            "status": result["status"],
            "results": result["results"],
            "requires_human_input": result.get("requires_human_input", False)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/human_feedback")
async def submit_human_feedback(feedback: HumanFeedback):
    try:
        # Get task from memory
        task_data = memory.get_task(feedback.task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Add human feedback
        task_data["human_feedback"] = {
            "approved": feedback.approved,
            "feedback": feedback.feedback,
            "modifications": feedback.modifications,
            "timestamp": datetime.now().isoformat()
        }
        
        # Continue workflow if approved
        if feedback.approved:
            workflow = create_workflow()
            task_data["status"] = "completing"
            result = workflow.invoke(task_data)
            memory.store_task(feedback.task_id, result)
            
            return {
                "task_id": feedback.task_id,
                "status": "completed",
                "final_results": result["results"]
            }
        else:
            # Handle feedback and restart
            task_data["status"] = "modified"
            task_data["task"] = f"{task_data['task']} (Modified based on feedback: {feedback.feedback})"
            
            workflow = create_workflow()
            result = workflow.invoke(task_data)
            memory.store_task(feedback.task_id, result)
            
            return {
                "task_id": feedback.task_id,
                "status": "reprocessed",
                "results": result["results"]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task(task_id: str):
    task_data = memory.get_task(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
