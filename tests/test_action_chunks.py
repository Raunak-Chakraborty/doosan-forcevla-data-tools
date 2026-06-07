import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.action_chunks import (
    build_future_action_chunk,
    build_future_action_chunk_from_lerobot_export,
)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class ActionChunkTests(unittest.TestCase):
    def test_future_chunk_inside_sequence(self):
        actions = [[float(i)] * 7 for i in range(5)]
        chunk = build_future_action_chunk(actions, start_index=1, horizon=3)

        self.assertEqual(chunk.actions, [[1.0] * 7, [2.0] * 7, [3.0] * 7])
        self.assertEqual(chunk.valid_mask, [True, True, True])
        self.assertEqual(chunk.padded_count, 0)

    def test_future_chunk_repeat_last_padding(self):
        actions = [[float(i)] * 7 for i in range(4)]
        chunk = build_future_action_chunk(actions, start_index=2, horizon=5, pad_mode="repeat_last")

        self.assertEqual(chunk.actions, [[2.0] * 7, [3.0] * 7, [3.0] * 7, [3.0] * 7, [3.0] * 7])
        self.assertEqual(chunk.valid_mask, [True, True, False, False, False])
        self.assertEqual(chunk.padded_count, 3)

    def test_future_chunk_zero_padding(self):
        actions = [[float(i)] * 7 for i in range(3)]
        chunk = build_future_action_chunk(actions, start_index=2, horizon=4, pad_mode="zero")

        self.assertEqual(chunk.actions, [[2.0] * 7, [0.0] * 7, [0.0] * 7, [0.0] * 7])
        self.assertEqual(chunk.valid_mask, [True, False, False, False])
        self.assertEqual(chunk.padded_count, 3)

    def test_invalid_action_dim_raises(self):
        with self.assertRaises(ValueError):
            build_future_action_chunk([[1.0, 2.0]], start_index=0, horizon=1)


@unittest.skipUnless(_has_module("pyarrow"), "requires pyarrow")
class ActionChunkParquetTests(unittest.TestCase):
    def test_load_chunk_from_minimal_lerobot_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)

            info = {
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
            (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

            table = pa.table(
                {
                    "action": pa.array(
                        [[0.0] * 7, [1.0] * 7, [2.0] * 7],
                        type=pa.list_(pa.float32()),
                    )
                }
            )
            pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")

            chunk = build_future_action_chunk_from_lerobot_export(
                root,
                episode_index=0,
                row_index=1,
                horizon=4,
            )

            self.assertEqual(chunk.actions, [[1.0] * 7, [2.0] * 7, [2.0] * 7, [2.0] * 7])
            self.assertEqual(chunk.valid_mask, [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
