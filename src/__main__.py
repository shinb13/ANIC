import logging

from agenthack import AgentRuntime

from src.my_agent import AgentCCAgent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    AgentRuntime(AgentCCAgent()).run()
