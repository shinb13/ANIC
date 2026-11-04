from agenthack import AgentRuntime

from src.my_agent import AgentCCAgent


if __name__ == "__main__":
    AgentRuntime(AgentCCAgent()).run()
