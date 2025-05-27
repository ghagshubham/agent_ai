import streamlit as st
import requests
import time
import os
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Multi-Agent AI Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'task_results' not in st.session_state:
    st.session_state.task_results = None
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# API endpoints
API_BASE = "http://localhost:8000"

def execute_task_api(task: str):
    """Call the execute_task API"""
    try:
        response = requests.post(
            f"{API_BASE}/execute_task",
            json={"task": task, "user_id": "demo_user"}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def submit_feedback_api(task_id: str, approved: bool, feedback: str = ""):
    """Submit human feedback"""
    try:
        response = requests.post(
            f"{API_BASE}/human_feedback",
            json={
                "task_id": task_id,
                "approved": approved,
                "feedback": feedback,
                "modifications": {}
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_task_api(task_id: str):
    """Get task details"""
    try:
        response = requests.get(f"{API_BASE}/task/{task_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Main app
st.title(" Multi-Agent AI Platform")
# st.subtitle("Autonomous AI with Human-in-the-Loop Capabilities")

# Sidebar
st.sidebar.title("Demo Controls")
# demo_task = st.sidebar.button(" Run Quantum Computing Demo")
clear_session = st.sidebar.button("🗑️ Clear Session")

if clear_session:
    st.session_state.current_task_id = None
    st.session_state.task_results = None
    st.session_state.demo_mode = False
    st.rerun()

# Demo mode
# if demo_task:
#     st.session_state.demo_mode = True
#     st.session_state.current_task_id = None
#     st.session_state.task_results = None

# Main interface
if st.session_state.demo_mode:
    st.header(" Demo: Quantum Computing Cybersecurity Analysis")
    
    demo_query = "Analyze the impact of quantum computing on cybersecurity, create visualizations of vulnerable sectors, and develop a sample quantum-resistant algorithm implementation"
    
    st.info(f"**Task:** {demo_query}")
    
    if st.button(" Execute Autonomous Analysis"):
        with st.spinner(" Multi-agent system processing..."):
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate agent execution steps
            agents = ["Router Agent", "Research Agent", "Code Generation Agent", "Visualization Agent"]
            for i, agent in enumerate(agents):
                status_text.text(f" {agent} working...")
                progress_bar.progress((i + 1) / len(agents))
                time.sleep(2)  # Simulate processing time
            
            # Execute actual task
            result = execute_task_api(demo_query)
            
            if "error" not in result:
                st.session_state.current_task_id = result["task_id"]
                st.session_state.task_results = result
                st.success(" Autonomous analysis completed!")
                st.rerun()
            else:
                st.error(f" Error: {result['error']}")
else:
    # Custom task input
    st.header(" Custom Task Input")
    
    with st.form("task_form"):
        user_task = st.text_area(
            "Enter your task:",
            placeholder="e.g., Research the latest developments in AI and create a summary with visualizations",
            height=100
        )
        submitted = st.form_submit_button("🚀 Execute Task")
        
        if submitted and user_task:
            with st.spinner(" Processing your task..."):
                result = execute_task_api(user_task)
                
                if "error" not in result:
                    st.session_state.current_task_id = result["task_id"]
                    st.session_state.task_results = result
                    st.success(" Task processing completed!")
                    st.rerun()
                else:
                    st.error(f" Error: {result['error']}")

                

# Display results if available
if st.session_state.task_results and st.session_state.current_task_id:
    st.header(" Autonomous Results")
    
    results = st.session_state.task_results
    
    # Task details
    with st.expander(" Task Details", expanded=True):
        st.write(f"**Task ID:** {results['task_id']}")
        st.write(f"**Status:** {results['status']}")
    
    # Results sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Research Results")
        if results.get("results", {}).get("research"):
            research = results["results"]["research"]
            
            if research.get("synthesis"):
                st.write("**Analysis Summary:**")
                st.write(research["synthesis"])
            
            if research.get("key_findings"):
                st.write("**Key Findings:**")
                for finding in research["key_findings"]:
                    st.write(f"• {finding}")
        else:
            st.info("Research results will appear here")
    
    with col2:
        st.subheader(" Code Generation")
        if results.get("results", {}).get("code"):
            code_result = results["results"]["code"]
            
            if code_result.get("documentation"):
                st.write("**Documentation:**")
                st.write(code_result["documentation"])
            
            if code_result.get("code"):
                st.write("**Generated Code:**")
                st.code(code_result["code"], language="python")
            
            if code_result.get("execution_result"):
                exec_result = code_result["execution_result"]
                if exec_result.get("success"):
                    st.success(" Code executed successfully")
                    if exec_result.get("output"):
                        st.text("Output:")
                        st.code(exec_result["output"])
                else:
                    st.warning(f"⚠️ Execution issue: {exec_result.get('error', 'Unknown error')}")
        else:
            st.info("Code results will appear here")
    
    # Visualizations
    st.subheader(" Visualizations")
    if results.get("results", {}).get("visualization"):
        viz_results = results["results"]["visualization"]
        
        if viz_results.get("visualizations"):
            for viz in viz_results["visualizations"]:
                st.write(f"**{viz.get('title', 'Visualization')}**")
                st.write(viz.get('description', ''))
                
                # Display chart if file exists
                if viz.get('file_path') and os.path.exists(viz['file_path']):
                    try:
                        image = Image.open(viz['file_path'])
                        st.image(image, caption=viz.get('title', ''), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                
                # Display insights
                if viz.get('key_insights'):
                    st.write("**Key Insights:**")
                    for insight in viz['key_insights']:
                        st.write(f"• {insight}")
                
                st.write("---")
        else:
            st.info("Visualizations will appear here")
    else:
        st.info("Visualizations will appear here")
    
    # Human-in-the-Loop Section
    st.header("👤 Human-in-the-Loop Review")
    
    if results.get("requires_human_input", False):
        st.warning(" **System is awaiting your feedback before finalizing results**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            feedback_text = st.text_area(
                "Provide feedback or modifications:",
                placeholder="e.g., Focus more on financial sector implications, add more recent data, etc.",
                height=100
            )
        
        with col2:
            if st.button(" Approve Results", type="primary"):
                with st.spinner("Processing approval..."):
                    feedback_result = submit_feedback_api(
                        st.session_state.current_task_id, 
                        True, 
                        feedback_text
                    )
                    
                    if "error" not in feedback_result:
                        st.success(" Results approved and finalized!")
                        st.session_state.task_results["status"] = "completed"
                        st.session_state.task_results["requires_human_input"] = False
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Error: {feedback_result['error']}")
        
        with col3:
            if st.button(" Request Changes", type="secondary"):
                if feedback_text:
                    with st.spinner("Processing feedback..."):
                        feedback_result = submit_feedback_api(
                            st.session_state.current_task_id, 
                            False, 
                            feedback_text
                        )
                        
                        if "error" not in feedback_result:
                            st.info(" Task being reprocessed with your feedback...")
                            time.sleep(3)
                            st.rerun()
                        else:
                            st.error(f"Error: {feedback_result['error']}")
                else:
                    st.warning("Please provide feedback before requesting changes.")
    else:
        st.success(" **Task completed and approved!**")
        
        if st.button(" Start New Task"):
            st.session_state.current_task_id = None
            st.session_state.task_results = None
            st.session_state.demo_mode = False
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Multi-Agent AI Platform**")
st.sidebar.markdown("Built with LangGraph + FastAPI")
st.sidebar.markdown("Human-in-the-Loop Enabled")
