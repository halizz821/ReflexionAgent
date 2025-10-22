# ReflexionAgent
Implementing a Reflexion Agent. This is an interesting starting point for learning Agentic AI using LangGraph

## Overview
This project implements a Reflexion Agent using the following tools:
- **LangGraph** and **LangChain** for agent orchestration and task management
- **OpenAI LLM** for natural language understanding and generation
- **TavilySearch** for searching information on the internet

> ⚠️ Make sure to update your API keys in the `.env` file before running the agent.

## Definition
A **Reflexion Agent** is an AI system designed to iteratively reflect on its own outputs, analyze mistakes or improvements, and update its behavior or responses accordingly. It uses self-feedback to refine decisions and enhance performance over time.

## How to Run
The agent is currently **command-line only** (no graphical interface).

1. Open `main.py` and modify the prompt manually to define the task for the agent.
2. Run the agent:
   ```bash
   python main.py
   ```
3. The number of self-reflection loops is controlled by `MAX_ITERATIONS` inside `graph.py`:
   ```python
   MAX_ITERATIONS = 1  # change this value to increase/decrease loops
   ```

## Installation
To install dependencies and set up the environment:

1. Install **uv** (a project environment manager):
   ```bash
   pip install uv
   ```
2. Navigate to the project folder and run:
   ```bash
   uv sync
   ```
   This will install all dependencies and create a new environment for the project.

## Acknowledgements
Thanks to  [harishneel1](https://github.com/harishneel1) for its course on [LangGraph](https://www.youtube.com/watch?v=Y3dbzuQBnUw&t=11278s)
that inspired this project.
