import json
import os

import openai
from agenthack import BaseAgent, state, turn

from agentcc_solver import _candidate_splits, solve_input


class AgentCCAgent(BaseAgent):
    async def solve(self, task_input: dict) -> dict:
        task = task_input["task"]
        topology = task_input["topology"]
        prompt_input = {
            "operator": task["operator"],
            "rank_count": len(task["ranks"]),
            "data_size_per_rank_bytes": task["data_size"]["per_rank_bytes"],
            "split_candidates": task["split_policy"]["candidates"],
            "alignment_bytes": task["split_policy"]["alignment_bytes"],
            "max_nonempty_rounds": task["max_nonempty_rounds"],
            "topology": {
                "rank_count": len(topology["ranks"]),
                "switch_count": len(topology.get("switches", [])),
                "edge_count": len(topology["edges"]),
                "resource_count": len(topology.get("resources", [])),
                "resources": [
                    {
                        "id": resource["id"],
                        "capacity": resource.get("capacity", resource.get("bw")),
                        "edge_count": len(resource.get("edges", [])),
                    }
                    for resource in topology.get("resources", [])
                ],
            },
        }
        prompt = (
            "Choose the split candidate for this AgentCC task. "
            "Return only a JSON object in the form {\"split\": integer}.\n"
            + json.dumps(prompt_input, ensure_ascii=False)
        )

        with turn(trigger="user"):
            with state("reasoning"):
                async with openai.AsyncOpenAI(
                    base_url=os.environ["PLATFORM_LLM_URL"],
                    api_key=os.environ["PLATFORM_API_KEY"],
                    timeout=20,
                    max_retries=1,
                ) as client:
                    response = await client.chat.completions.create(
                        model=os.environ["AGENT_MODEL_NAME"],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=32,
                    )
                content = response.choices[0].message.content
                valid_splits = _candidate_splits(task)
                preferred_split = valid_splits[0]
                if content:
                    try:
                        decision = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        decision = None
                    model_split = decision.get("split") if isinstance(decision, dict) else None
                    if type(model_split) is int and model_split in valid_splits:
                        preferred_split = model_split

                answer = solve_input(task_input, preferred_split=preferred_split)

            with state("final_answer"):
                return answer
