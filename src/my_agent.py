import json
import os

import openai
from agenthack import BaseAgent, state, turn

from agentcc_solver import solve_input


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
                if not content:
                    raise ValueError("the model returned an empty split decision")
                decision = json.loads(content)
                preferred_split = decision["split"]
                if type(preferred_split) is not int:
                    raise ValueError("the model split decision must be an integer")
                if preferred_split not in {int(value) for value in task["split_policy"]["candidates"]}:
                    raise ValueError("the model selected a split outside the task candidates")
                data_size = int(task["data_size"]["per_rank_bytes"])
                alignment = int(task["split_policy"]["alignment_bytes"])
                if data_size % preferred_split != 0 or (data_size // preferred_split) % alignment != 0:
                    raise ValueError("the model selected a split that violates alignment")

                answer = solve_input(task_input, preferred_split=preferred_split)

            with state("final_answer"):
                return answer
