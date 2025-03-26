import asyncio
from .echo_agent import EchoAgent
from .echo_skill import EchoSkill
from src.core.state import AgentState

if __name__ == "__main__":
    agent = EchoAgent(skills=[EchoSkill()])
    workflow = agent.create_workflow()

    state = AgentState()
    state.add_user_message("Hello, echo this!")

    # Use asyncio to run the async workflow
    final_state = asyncio.run(workflow.ainvoke(state))

    print("=== Final Messages ===")
    for msg in final_state["messages"]:
        if "type" in msg:
            msg_type = msg["type"]
            if msg_type == "human":
                print(f"[user] {msg.get('content', '')}")
            elif msg_type == "ai":
                print(f"[assistant] {msg.get('content', '')}")
            elif msg_type == "system":
                print(f"[system] {msg.get('content', '')}")
            elif msg_type == "tool":
                print(f"[tool: {msg.get('name', 'unknown')}] {msg.get('content', '')}")
            else:
                print(f"[{msg_type}] {msg.get('content', '')}")
        elif "role" in msg:
            print(f"[{msg['role']}] {msg['content']}")
        else:
            print(f"[unknown] {msg}")
