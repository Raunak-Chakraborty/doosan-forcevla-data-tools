import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


@unittest.skipUnless(
    _has_module("pyarrow") and _has_module("av") and _has_module("imageio"),
    "requires pyarrow, av, and imageio",
)
class SmokeForceVLAObservationBuilderTests(unittest.TestCase):
    def test_smoke_builder_reads_minimal_export(self):
        import imageio.v2 as imageio
        import pyarrow as pa
        import pyarrow.parquet as pq

        from doosan_forcevla_data.inspect.smoke_forcevla_observation_builder import (
            smoke_forcevla_observation_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "videos" / "observation.image").mkdir(parents=True)
            (root / "videos" / "observation.wrist_image").mkdir(parents=True)

            info = {
                "codebase_version": "v2.1",
                "robot_type": "doosan_m1013",
                "total_episodes": 1,
                "total_frames": 2,
                "total_tasks": 1,
                "fps": 30,
                "chunks_size": 1000,
                "total_chunks": 1,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.image": {
                        "dtype": "video",
                        "shape": [16, 16, 3],
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "video",
                        "shape": [16, 16, 3],
                        "names": ["height", "width", "channels"],
                    },
                    "observation.state": {"dtype": "float32", "shape": [13]},
                    "action": {"dtype": "float32", "shape": [7]},
                    "timestamp": {"dtype": "float64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                    "index": {"dtype": "int64", "shape": [1]},
                    "task": {"dtype": "string", "shape": [1]},
                    "prompt": {"dtype": "string", "shape": [1]},
                },
            }
            (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
            _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "Insert the peg."}])
            _write_jsonl(
                root / "meta" / "episodes.jsonl",
                [{"episode_index": 0, "task_index": 0, "length": 2}],
            )
            _write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": 0, "stats": {}}])

            table = pa.table(
                {
                    "observation.state": pa.array([[0.0] * 13, [1.0] * 13], type=pa.list_(pa.float32())),
                    "action": pa.array([[0.0] * 7, [0.1] * 7], type=pa.list_(pa.float32())),
                    "timestamp": pa.array([0.0, 1.0 / 30.0], type=pa.float64()),
                    "frame_index": pa.array([0, 1], type=pa.int64()),
                    "episode_index": pa.array([0, 0], type=pa.int64()),
                    "task_index": pa.array([0, 0], type=pa.int64()),
                    "index": pa.array([0, 1], type=pa.int64()),
                    "task": pa.array(["Insert the peg.", "Insert the peg."], type=pa.string()),
                    "prompt": pa.array(["Insert the peg.", "Insert the peg."], type=pa.string()),
                }
            )
            pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")

            frame_a = [[[0, 0, 0] for _ in range(16)] for _ in range(16)]
            frame_b = [[[255, 255, 255] for _ in range(16)] for _ in range(16)]
            for key in ["observation.image", "observation.wrist_image"]:
                imageio.mimsave(
                    root / "videos" / key / "episode_000000.mp4",
                    [frame_a, frame_b],
                    fps=30,
                )

            report = smoke_forcevla_observation_builder(root, row_index=0)
            self.assertTrue(report["ok"])
            self.assertEqual(report["parquet_rows"], 2)
            self.assertEqual(report["sample"]["state_dim"], 13)
            self.assertEqual(report["sample"]["action_dim"], 7)
            self.assertEqual(report["videos"]["observation.image"]["decoded_frames"], 2)
            self.assertEqual(report["forcevla_observation_summary"]["observation.image.shape"], [16, 16, 3])


if __name__ == "__main__":
    unittest.main()
