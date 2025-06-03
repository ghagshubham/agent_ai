# import streamlit as st
# import requests
# import time
# import os
# from PIL import Image

# # Configure page
# st.set_page_config(
#     page_title="Multi-Agent AI Platform",
#     page_icon="🤖",
#     layout="wide"
# )

# # Initialize session state
# if 'current_task_id' not in st.session_state:
#     st.session_state.current_task_id = None
# if 'task_results' not in st.session_state:
#     st.session_state.task_results = None
# if 'demo_mode' not in st.session_state:
#     st.session_state.demo_mode = False

# # API endpoints
# API_BASE = "http://localhost:8000"


# def execute_task_api(task: str):
#     try:
#         response = requests.post(
#             f"{API_BASE}/execute_task",
#             json={"task": task, "user_id": "demo_user"}
#         )
#         result = response.json()
#         print("DEBUG API RESPONSE:", result)  # Useful for debugging
#         return result
#     except Exception as e:
#         return {"error": str(e)}

# def submit_feedback_api(task_id: str, approved: bool, feedback: str = ""):
#     """Submit human feedback"""
#     try:
#         response = requests.post(
#             f"{API_BASE}/human_feedback",
#             json={
#                 "task_id": task_id,
#                 "approved": approved,
#                 "feedback": feedback,
#                 "modifications": {}
#             }
#         )
#         return response.json()
#     except Exception as e:
#         return {"error": str(e)}

# def get_task_api(task_id: str):
#     """Get task details"""
#     try:
#         response = requests.get(f"{API_BASE}/task/{task_id}")
#         return response.json()
#     except Exception as e:
#         return {"error": str(e)}

# # Main app
# st.title(" Multi-Agent AI Platform")
# # st.subtitle("Autonomous AI with Human-in-the-Loop Capabilities")

# # Sidebar
# st.sidebar.title("Demo Controls")
# # demo_task = st.sidebar.button(" Run Quantum Computing Demo")
# clear_session = st.sidebar.button("🗑️ Clear Session")

# if clear_session:
#     st.session_state.current_task_id = None
#     st.session_state.task_results = None
#     st.session_state.demo_mode = False
#     st.rerun()

# # Demo mode
# # if demo_task:
# #     st.session_state.demo_mode = True
# #     st.session_state.current_task_id = None
# #     st.session_state.task_results = None

# # Main interface
# if st.session_state.demo_mode:
#     st.header(" Demo: Quantum Computing Cybersecurity Analysis")
    
#     demo_query = "Analyze the impact of quantum computing on cybersecurity, create visualizations of vulnerable sectors, and develop a sample quantum-resistant algorithm implementation"
    
#     st.info(f"**Task:** {demo_query}")
    
#     if st.button(" Execute Autonomous Analysis"):
#         with st.spinner(" Multi-agent system processing..."):
#             # Show progress
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Simulate agent execution steps
#             agents = ["Router Agent", "Research Agent", "Code Generation Agent", "Visualization Agent"]
#             for i, agent in enumerate(agents):
#                 status_text.text(f" {agent} working...")
#                 progress_bar.progress((i + 1) / len(agents))
#                 time.sleep(2)  # Simulate processing time
            
#             # Execute actual task
#             result = execute_task_api(demo_query)
            
#             if result and isinstance(result, dict) and "task_id" in result:
#                 st.session_state.current_task_id = result["task_id"]
#                 st.session_state.task_results = result
#                 st.success(" Autonomous analysis completed!")
#                 st.rerun()
#             else:
#                 error_msg = result.get("error", "Unknown error or invalid response from API.")
#                 st.error(f"❌ Failed to execute task: {error_msg}")

# else:
#     # Custom task input
#     st.header(" Custom Task Input")
    
#     with st.form("task_form"):
#         user_task = st.text_area(
#             "Enter your task:",
#             placeholder="e.g., Research the latest developments in AI and create a summary with visualizations",
#             height=100
#         )
#         submitted = st.form_submit_button("🚀 Execute Task")
        
#         if submitted and user_task:
#             with st.spinner(" Processing your task..."):
#                 result = execute_task_api(user_task)
                
#                 if result and isinstance(result, dict) and "task_id" in result:
#                     st.session_state.current_task_id = result["task_id"]
#                     st.session_state.task_results = result
#                     st.success(" Task processing completed!")
#                     st.rerun()
#                 else:
#                     error_msg = result.get("error", "Unknown error or invalid response from API.")
#                     st.error(f"❌ Failed to execute task: {error_msg}")


                

# # Display results if available
# if st.session_state.task_results and st.session_state.current_task_id:
#     st.header(" Autonomous Results")
    
#     results = st.session_state.task_results
    
#     # Task details
#     with st.expander(" Task Details", expanded=True):
#         st.write(f"**Task ID:** {results['task_id']}")
#         st.write(f"**Status:** {results['status']}")
    
#     # Results sections
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🔍 Research Results")
#         if results.get("results", {}).get("research"):
#             research = results["results"]["research"]
            
#             if research.get("synthesis"):
#                 st.write("**Analysis Summary:**")
#                 st.write(research["synthesis"])
            
#             if research.get("key_findings"):
#                 st.write("**Key Findings:**")
#                 for finding in research["key_findings"]:
#                     st.write(f"• {finding}")
#         else:
#             st.info("Research results will appear here")
    
#     with col2:
#         st.subheader(" Code Generation")
#         if results.get("results", {}).get("code"):
#             code_result = results["results"]["code"]
            
#             if code_result.get("documentation"):
#                 st.write("**Documentation:**")
#                 st.write(code_result["documentation"])
            
#             if code_result.get("code"):
#                 st.write("**Generated Code:**")
#                 st.code(code_result["code"], language="python")
            
#             if code_result.get("execution_result"):
#                 exec_result = code_result["execution_result"]
#                 if exec_result.get("success"):
#                     st.success(" Code executed successfully")
#                     if exec_result.get("output"):
#                         st.text("Output:")
#                         st.code(exec_result["output"])
#                 else:
#                     st.warning(f"⚠️ Execution issue: {exec_result.get('error', 'Unknown error')}")
#         else:
#             st.info("Code results will appear here")
    
#     # Visualizations
#     st.subheader(" Visualizations")
#     if results.get("results", {}).get("visualization"):
#         viz_results = results["results"]["visualization"]
        
#         if viz_results.get("visualizations"):
#             for viz in viz_results["visualizations"]:
#                 st.write(f"**{viz.get('title', 'Visualization')}**")
#                 st.write(viz.get('description', ''))
                
#                 # Display chart if file exists
#                 if viz.get('file_path') and os.path.exists(viz['file_path']):
#                     try:
#                         image = Image.open(viz['file_path'])
#                         st.image(image, caption=viz.get('title', ''), use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Error loading image: {e}")
                
#                 # Display insights
#                 if viz.get('key_insights'):
#                     st.write("**Key Insights:**")
#                     for insight in viz['key_insights']:
#                         st.write(f"• {insight}")
                
#                 st.write("---")
#         else:
#             st.info("Visualizations will appear here")
#     else:
#         st.info("Visualizations will appear here")
    
#     # Human-in-the-Loop Section
#     st.header("👤 Human-in-the-Loop Review")
    
#     if results.get("requires_human_input", False):
#         st.warning(" **System is awaiting your feedback before finalizing results**")
        
#         col1, col2, col3 = st.columns([2, 1, 1])
        
#         with col1:
#             feedback_text = st.text_area(
#                 "Provide feedback or modifications:",
#                 placeholder="e.g., Focus more on financial sector implications, add more recent data, etc.",
#                 height=100
#             )
        
#         with col2:
#             if st.button(" Approve Results", type="primary"):
#                 with st.spinner("Processing approval..."):
#                     feedback_result = submit_feedback_api(
#                         st.session_state.current_task_id, 
#                         True, 
#                         feedback_text
#                     )
                    
#                     if "error" not in feedback_result:
#                         st.success(" Results approved and finalized!")
#                         st.session_state.task_results["status"] = "completed"
#                         st.session_state.task_results["requires_human_input"] = False
#                         time.sleep(2)
#                         st.rerun()
#                     else:
#                         st.error(f"Error: {feedback_result['error']}")
        
#         with col3:
#             if st.button(" Request Changes", type="secondary"):
#                 if feedback_text:
#                     with st.spinner("Processing feedback..."):
#                         feedback_result = submit_feedback_api(
#                             st.session_state.current_task_id, 
#                             False, 
#                             feedback_text
#                         )
                        
#                         if "error" not in feedback_result:
#                             st.info(" Task being reprocessed with your feedback...")
#                             time.sleep(3)
#                             st.rerun()
#                         else:
#                             st.error(f"Error: {feedback_result['error']}")
#                 else:
#                     st.warning("Please provide feedback before requesting changes.")
#     else:
#         st.success(" **Task completed and approved!**")
        
#         if st.button(" Start New Task"):
#             st.session_state.current_task_id = None
#             st.session_state.task_results = None
#             st.session_state.demo_mode = False
#             st.rerun()

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("**Multi-Agent AI Platform**")
# st.sidebar.markdown("Built with LangGraph + FastAPI")
# st.sidebar.markdown("Human-in-the-Loop Enabled")
import streamlit as st
import requests
import time
import os
from PIL import Image
import json

# Configure page with custom styling
st.set_page_config(
    page_title="Multi-Agent AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .agent-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .status-success {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-pending {
        background: linear-gradient(45deg, #f093fb, #f5576c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .feedback-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-top: 2rem;
    }
    
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #6c757d;
        background: #f8f9fa;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'task_results' not in st.session_state:
    st.session_state.task_results = None

# API endpoints
API_BASE = "http://localhost:8000"

def execute_task_api(task: str):
    try:
        response = requests.post(
            f"{API_BASE}/execute_task",
            json={"task": task, "user_id": "demo_user"}
        )
        result = response.json()
        return result
    except Exception as e:
        return {"error": str(e)}

def submit_feedback_api(task_id: str, approved: bool, feedback: str = ""):
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
    try:
        response = requests.get(f"{API_BASE}/task/{task_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def detect_task_type(task: str):
    """Intelligently detect what type of task the user is requesting"""
    task_lower = task.lower()
    
    task_types = {
        'research': ['research', 'analyze', 'study', 'investigate', 'explore', 'find information', 'summary'],
        'code': ['code', 'programming', 'develop', 'implement', 'algorithm', 'function', 'script', 'build'],
        'visualization': ['visualiz', 'chart', 'graph', 'plot', 'diagram', 'visual', 'image', 'picture']
    }
    
    detected_types = []
    for task_type, keywords in task_types.items():
        if any(keyword in task_lower for keyword in keywords):
            detected_types.append(task_type)
    
    return detected_types if detected_types else ['research']  # Default to research

def render_agent_working_animation(agents):
    """Render a beautiful agent working animation"""
    progress_container = st.container()
    
    with progress_container:
        cols = st.columns(len(agents))
        status_containers = []
        
        for i, (col, agent) in enumerate(zip(cols, agents)):
            with col:
                status_container = st.empty()
                status_containers.append(status_container)
                status_container.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                        <div style="font-weight: bold; color: #667eea;">{agent}</div>
                        <div style="color: #6c757d;">Waiting...</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Animate agents working
        for i, agent in enumerate(agents):
            status_containers[i].markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <div style="font-weight: bold; color: #28a745;">{agent}</div>
                    <div style="color: #28a745;">Working...</div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
        
        # Show completion
        for i, agent in enumerate(agents):
            status_containers[i].markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">✅</div>
                    <div style="font-weight: bold; color: #28a745;">{agent}</div>
                    <div style="color: #28a745;">Complete!</div>
                </div>
            """, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Multi-Agent AI Platform</h1>
    <p style="margin: 0; opacity: 0.9;">Intelligent task execution with human-in-the-loop capabilities</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.markdown("### Quick Actions")
    clear_session = st.button("🗑️ Clear Session", use_container_width=True)
    
    if clear_session:
        st.session_state.current_task_id = None
        st.session_state.task_results = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Platform Info")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><strong>Status</strong><br/>🟢 Online</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><strong>Mode</strong><br/>🤖 Auto</div>', unsafe_allow_html=True)

# Custom task input
st.markdown("## 💬 Task Input")

with st.form("task_form", clear_on_submit=False):
    user_task = st.text_area(
        "What would you like me to help you with?",
        placeholder="Examples:\n• Research the latest AI developments and summarize key findings\n• Write Python code to analyze CSV data\n• Create visualizations showing market trends\n• Build a web scraper for product data",
        height=120
    )
    
    # Show predicted task type
    if user_task:
        predicted_types = detect_task_type(user_task)
        type_badges = []
        for t in predicted_types:
            if t == 'research':
                type_badges.append("🔍 Research")
            elif t == 'code':
                type_badges.append("💻 Code Generation")
            elif t == 'visualization':
                type_badges.append("📊 Visualization")
        
        st.markdown(f"**Detected task types:** {' • '.join(type_badges)}")
    
    submitted = st.form_submit_button("🚀 Execute Task", type="primary", use_container_width=True)
    
    if submitted and user_task:
        # Determine which agents to show based on task type
        task_types = detect_task_type(user_task)
        agents = ["🎯 Router Agent"]
        
        if 'research' in task_types:
            agents.append("🔍 Research Agent")
        if 'code' in task_types:
            agents.append("💻 Code Agent")
        if 'visualization' in task_types:
            agents.append("📊 Visualization Agent")
        
        render_agent_working_animation(agents)
        
        with st.spinner("🔄 Processing your request..."):
            result = execute_task_api(user_task)
            
            if result and isinstance(result, dict) and "task_id" in result:
                st.session_state.current_task_id = result["task_id"]
                st.session_state.task_results = result
                st.success("✅ Task completed successfully!")
                time.sleep(1)
                st.rerun()
            else:
                error_msg = result.get("error", "Unknown error")
                st.error(f"❌ Task failed: {error_msg}")

# Display results if available
if st.session_state.task_results and st.session_state.current_task_id:
    results = st.session_state.task_results
    
    # Task overview
    st.markdown("## 📋 Task Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><strong>Task ID</strong><br/>{results["task_id"][:8]}...</div>', unsafe_allow_html=True)
    with col2:
        status_class = "status-success" if results["status"] == "completed" else "status-pending"
        st.markdown(f'<div class="{status_class}">{results["status"].title()}</div>', unsafe_allow_html=True)
    with col3:
        timestamp = time.strftime("%H:%M:%S")
        st.markdown(f'<div class="metric-card"><strong>Completed</strong><br/>{timestamp}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dynamic result sections based on what was actually generated
    results_data = results.get("results", {})
    
    # Only show sections that have actual content
    if results_data.get("research"):
        st.markdown('<div class="agent-badge">🔍 Research Agent Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        research = results_data["research"]
        
        if research.get("synthesis"):
            st.markdown("**📄 Analysis Summary**")
            st.write(research["synthesis"])
        
        if research.get("key_findings"):
            st.markdown("**🎯 Key Findings**")
            for i, finding in enumerate(research["key_findings"], 1):
                st.write(f"{i}. {finding}")
        
        if research.get("sources"):
            with st.expander("📚 Sources"):
                for source in research["sources"]:
                    st.write(f"• {source}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if results_data.get("code"):
        st.markdown('<div class="agent-badge">💻 Code Generation Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        code_result = results_data["code"]
        
        if code_result.get("documentation"):
            st.markdown("**📖 Documentation**")
            st.write(code_result["documentation"])
        
        if code_result.get("code"):
            st.markdown("**🔧 Generated Code**")
            st.code(code_result["code"], language="python")
        
        if code_result.get("execution_result"):
            exec_result = code_result["execution_result"]
            if exec_result.get("success"):
                st.success("✅ Code executed successfully")
                if exec_result.get("output"):
                    st.markdown("**📤 Output:**")
                    st.code(exec_result["output"])
            else:
                st.warning(f"⚠️ Execution issue: {exec_result.get('error', 'Unknown error')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if results_data.get("visualization"):
        st.markdown('<div class="agent-badge">📊 Visualization Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        viz_results = results_data["visualization"]
        
        if viz_results.get("visualizations"):
            for i, viz in enumerate(viz_results["visualizations"]):
                st.markdown(f"**📈 {viz.get('title', f'Visualization {i+1}')}**")
                
                if viz.get('description'):
                    st.write(viz['description'])
                
                # Display chart if file exists
                if viz.get('file_path') and os.path.exists(viz['file_path']):
                    try:
                        image = Image.open(viz['file_path'])
                        st.image(image, caption=viz.get('title', ''), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading visualization: {e}")
                
                # Display insights
                if viz.get('key_insights'):
                    st.markdown("**💡 Key Insights:**")
                    for insight in viz['key_insights']:
                        st.write(f"• {insight}")
                
                if i < len(viz_results["visualizations"]) - 1:
                    st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show empty state if no results
    if not any([results_data.get("research"), results_data.get("code"), results_data.get("visualization")]):
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤔</div>
            <h3>No Results Generated</h3>
            <p>The task completed but didn't generate any displayable results. This might be normal for certain types of tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Human-in-the-Loop Section
    if results.get("requires_human_input", False):
        st.markdown("""
        <div class="feedback-section">
            <h2 style="color: white; margin-top: 0;">👤 Human Review Required</h2>
            <p style="opacity: 0.9;">The system is awaiting your feedback before finalizing the results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        feedback_text = st.text_area(
            "💬 Provide your feedback:",
            placeholder="e.g., Focus more on recent developments, add error handling to the code, create additional charts showing trends over time...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve Results", type="primary", use_container_width=True):
                with st.spinner("Processing approval..."):
                    feedback_result = submit_feedback_api(
                        st.session_state.current_task_id, 
                        True, 
                        feedback_text
                    )
                    
                    if "error" not in feedback_result:
                        st.success("✅ Results approved and finalized!")
                        st.session_state.task_results["status"] = "completed"
                        st.session_state.task_results["requires_human_input"] = False
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Error: {feedback_result['error']}")
        
        with col2:
            if st.button("🔄 Request Changes", use_container_width=True):
                if feedback_text:
                    with st.spinner("Processing feedback..."):
                        feedback_result = submit_feedback_api(
                            st.session_state.current_task_id, 
                            False, 
                            feedback_text
                        )
                        
                        if "error" not in feedback_result:
                            st.info("🔄 Task being reprocessed with your feedback...")
                            time.sleep(3)
                            st.rerun()
                        else:
                            st.error(f"Error: {feedback_result['error']}")
                else:
                    st.warning("Please provide feedback before requesting changes.")
    else:
        st.markdown("""
        <div class="feedback-section">
            <h3 style="color: white; margin-top: 0;">🎉 Task Completed Successfully!</h3>
            <p style="opacity: 0.9;">All agents have finished their work and the results are ready for use.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start New Task", type="primary", use_container_width=True):
            st.session_state.current_task_id = None
            st.session_state.task_results = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <strong>Multi-Agent AI Platform</strong> • Built with LangGraph + FastAPI<br/>
    🤖 Autonomous • 👤 Human-in-the-Loop • 🚀 Intelligent Task Execution
</div>
""", unsafe_allow_html=True)