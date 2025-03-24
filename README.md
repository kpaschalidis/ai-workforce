# SkillFlow – Build Composable AI Agents with Skills

## Overview

**SkillFlow** is a lightweight framework for building modular, skill-based AI agents. It's designed to help you go beyond single-purpose chatbots and create structured, intelligent agents that can reason, use tools, and solve real-world problems.

### Key Features

- **Skill-First Architecture**: Define business logic as plug-and-play skills.
- **Agent Orchestration**: Compose multiple agents or skills into workflows.
- **LLM + Tool Integration**: Use language models and real functions side by side.
- **Transparent State Management**: Every decision, message, and tool call is tracked.
- **Real-World Use Cases**: Built for actual tasks like writing proposals, analyzing contracts, and generating content.

## Core Principles

### 1. Skills are Modular
Each skill is a self-contained capability — like generating markdown, extracting data, or analyzing a smart contract. You can mix and match them in different agents.

### 2. Agents are Composers
Agents manage conversation, state, and routing. You can build single-agent tools or multi-agent workflows with coordinated execution.

### 3. Tools are Real
Every skill can include tools that actually run code, call APIs, validate inputs, or use LLMs. Tools can retry, log, and report failures cleanly.

1. **Skill-Centric Design**: 
   - Each skill represents a specific business capability
   - Easily extensible and customizable
   - Decoupled from underlying AI implementation

2. **Intelligent Routing**:
   - Dynamic decision-making
   - Context-aware task allocation
   - Machine learning-enhanced routing

3. **Comprehensive Security**:
   - Input validation
   - Access control
   - Compliance tracking


##  Installation

### Requirements

- Python 3.11.7+
- OpenAI API key

### Setup

```bash
pip install -r requirements.txt
```

## Examples

**Run the 'EchoAgent'**:
```bash
python -m src.examples.echo.main 
```