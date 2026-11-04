import json
from pathlib import Path
import unittest

from agentcc_solver import estimate_schedule_time_us, solve_request


ROOT = Path(__file__).parent
PUBLIC = ROOT / "agentcc_zip_tmp" / "examples"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class AgentCCBaselineTests(unittest.TestCase):
    def test_shared_resource_examples(self):
        base = PUBLIC / "shared-resource"
        topology = read_json(base / "topology.json")
        for operator in ("all-gather", "all-reduce"):
            task = read_json(base / f"{operator}-task.json")
            response = solve_request(task, topology)
            self.assertEqual(response["status"], "success")
            answer = response["output"]["answer"]
            self.assertEqual(answer["operator"], operator)
            self.assertEqual(answer["split"], 1)
            self.assertAlmostEqual(estimate_schedule_time_us(task, topology, answer), 66.836, places=3)

    def test_all_public_256_rank_examples(self):
        families = (
            "uneven_pods",
            "directed_ring",
            "sparse_fabric",
            "cloudmatrix",
            "fullmesh_clos",
            "mesh2d",
            "nvlink_like",
        )
        for family in families:
            for short_name, operator in (("ag", "all-gather"), ("ar", "all-reduce")):
                base = PUBLIC / "256-rank" / family / short_name
                task = read_json(base / "task.json")
                topology = read_json(base / "topology.json")
                response = solve_request(task, topology)
                self.assertEqual(response["status"], "success")
                answer = response["output"]["answer"]
                self.assertEqual(answer["operator"], operator)
                self.assertEqual(answer["split"], 1)
                self.assertEqual(answer["round_count"], 8)


if __name__ == "__main__":
    unittest.main()
