# Multi-Agent AI Platform

A sophisticated multi-agent AI system built with LangGraph and FastAPI that autonomously handles complex tasks with human-in-the-loop capabilities.

## Features

- **Multi-Agent Architecture**: Coordinated agents for research, code generation, and visualization
- **Human-in-the-Loop**: Review and approve results before finalization
- **Autonomous Execution**: Agents work together to complete complex tasks
- **Interactive Frontend**: Streamlit-based UI for task submission and result review
- **RESTful API**: FastAPI backend for programmatic access
- **Memory Management**: Task storage and retrieval system

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Tavily API key (for research)

### Installation

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and install:
```bash
git clone https://github.com/ghagshubham/AI_Agents
cd LangGraphAgent
uv venv
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
```

### Running the Application

**Note**: Make sure your virtual environment is activated before running these commands.

1. Start the FastAPI backend:
```bash
uv run python main.py
```

2. Launch the Streamlit frontend (in a new terminal):
```bash
source .venv/bin/activate  # Activate venv first
uv run streamlit run frontend.py
```

3. Open your browser to `http://localhost:8501`

## Usage

### Demo Mode
Click "Run Quantum Computing Demo" to see the system analyze quantum computing's impact on cybersecurity with automated research, code generation, and visualizations.

### Custom Tasks
Enter any complex task requiring research, analysis, and code generation. The system will:
1. Route your task to appropriate agents
2. Conduct research and analysis
3. Generate relevant code and visualizations
4. Present results for your review and approval

### API Endpoints

- `POST /execute_task` - Submit a new task
- `POST /human_feedback` - Provide feedback on results
- `GET /task/{task_id}` - Retrieve task details

## Architecture

The platform uses specialized agents:
- **Router Agent**: Determines task routing and coordination
- **Research Agent**: Conducts web research and analysis
- **Code Generation Agent**: Creates and executes code solutions
- **Visualization Agent**: Generates charts and visual insights

## Example Tasks

- "Analyze market trends and create predictive models"
- "Research AI ethics and generate compliance framework"
- "Study renewable energy technologies and create comparison visualizations"
- "Investigate cybersecurity threats and develop mitigation strategies"

## Technology Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Frontend**: Streamlit
- **AI**: OpenAI GPT models
- **Research**: Tavily API
- **Visualization**: Matplotlib, Seaborn
- **Data**: Pandas, NumPy
