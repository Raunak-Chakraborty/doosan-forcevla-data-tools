import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.ingest.ros2_mcap import (
    McapIngestError,
    discover_episode_dirs,
    load_episode_descriptor,
    read_episode_metadata,
)


def _metadata(
    *,
    storage_identifier: str = "mcap",
    message_count: int = 3,
    relative_file_paths: (
        list[str] | None
    ) = None,
) -> dict:
    if relative_file_paths is None:
        relative_file_paths = [
            "episode_0.mcap"
        ]

    return {
        "rosbag2_bagfile_information": {
            "version": 9,
            "storage_identifier": (
                storage_identifier
            ),
            "message_count": message_count,
            "relative_file_paths": (
                relative_file_paths
            ),
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/a",
                        "type": (
                            "std_msgs/msg/String"
                        ),
                        "serialization_format": (
                            "cdr"
                        ),
                    },
                    "message_count": 1,
                },
                {
                    "topic_metadata": {
                        "name": "/b",
                        "type": (
                            "sensor_msgs/msg/Joy"
                        ),
                        "serialization_format": (
                            "cdr"
                        ),
                    },
                    "message_count": 2,
                },
            ],
            "custom_data": {
                "episode_index": 10,
                "raw_contract": (
                    "doosan_two_camera_"
                    "rosbag_raw_v1"
                ),
            },
        }
    }


def _write_episode(
    root: Path,
    *,
    metadata: dict | None = None,
    write_operator: bool = True,
    write_validation: bool = True,
) -> None:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if metadata is None:
        metadata = _metadata()

    (
        root / "metadata.yaml"
    ).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    relative_paths = (
        metadata[
            "rosbag2_bagfile_information"
        ].get(
            "relative_file_paths",
            [],
        )
    )

    for relative_path in relative_paths:
        relative = Path(relative_path)

        if ".." in relative.parts:
            continue

        path = root / relative

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(b"mcap")

    if write_operator:
        (
            root
            / "episode_operator.json"
        ).write_text(
            json.dumps(
                {
                    "schema_version": (
                        "operator_v3"
                    )
                }
            ),
            encoding="utf-8",
        )

    if write_validation:
        (
            root
            / "episode_validation.json"
        ).write_text(
            json.dumps(
                {
                    "schema_version": (
                        "validation_v1"
                    ),
                    "passed": True,
                }
            ),
            encoding="utf-8",
        )


class Ros2McapIngestMetadataTests(
    unittest.TestCase
):
    def test_metadata_and_sidecars_parse(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir)
                / "episode_000010"
            )

            _write_episode(episode)

            descriptor = (
                load_episode_descriptor(
                    episode
                )
            )

            self.assertEqual(
                descriptor
                .metadata
                .storage_identifier,
                "mcap",
            )

            self.assertEqual(
                descriptor
                .metadata
                .message_count,
                3,
            )

            self.assertEqual(
                {
                    item.name: (
                        item.message_count
                    )
                    for item in (
                        descriptor
                        .metadata
                        .topics
                    )
                },
                {
                    "/a": 1,
                    "/b": 2,
                },
            )

            self.assertEqual(
                descriptor.operator[
                    "schema_version"
                ],
                "operator_v3",
            )

            self.assertTrue(
                descriptor.validation[
                    "passed"
                ]
            )

    def test_discovery_direct_and_children(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            episode_b = (
                root / "episode_000002"
            )

            episode_a = (
                root / "episode_000001"
            )

            _write_episode(episode_b)
            _write_episode(episode_a)

            self.assertEqual(
                discover_episode_dirs(
                    root
                ),
                [
                    episode_a.resolve(),
                    episode_b.resolve(),
                ],
            )

            self.assertEqual(
                discover_episode_dirs(
                    episode_a
                ),
                [
                    episode_a.resolve()
                ],
            )

    def test_rejects_non_mcap_storage(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir) / "episode"
            )

            _write_episode(
                episode,
                metadata=_metadata(
                    storage_identifier=(
                        "sqlite3"
                    )
                ),
            )

            with self.assertRaisesRegex(
                McapIngestError,
                "storage_identifier",
            ):
                read_episode_metadata(
                    episode
                )

    def test_rejects_total_count_disagreement(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir) / "episode"
            )

            _write_episode(
                episode,
                metadata=_metadata(
                    message_count=99
                ),
            )

            with self.assertRaisesRegex(
                McapIngestError,
                "per-topic counts",
            ):
                read_episode_metadata(
                    episode
                )

    def test_rejects_mcap_path_escape(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir) / "episode"
            )

            _write_episode(
                episode,
                metadata=_metadata(
                    relative_file_paths=[
                        "../escape.mcap"
                    ]
                ),
            )

            with self.assertRaisesRegex(
                McapIngestError,
                "escapes episode",
            ):
                read_episode_metadata(
                    episode
                )

    def test_requires_both_sidecars(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir) / "episode"
            )

            _write_episode(
                episode,
                write_validation=False,
            )

            with self.assertRaisesRegex(
                McapIngestError,
                "episode_validation.json",
            ):
                load_episode_descriptor(
                    episode
                )

    def test_rejects_unlisted_mcap_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir) / "episode"
            )

            _write_episode(episode)

            (
                episode / "stray.mcap"
            ).write_bytes(b"stray")

            with self.assertRaisesRegex(
                McapIngestError,
                "MCAP set mismatch",
            ):
                read_episode_metadata(
                    episode
                )

    def test_rejects_non_cdr_topic(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = (
                Path(tmpdir) / "episode"
            )

            metadata = _metadata()

            metadata[
                "rosbag2_bagfile_information"
            ][
                "topics_with_message_count"
            ][0][
                "topic_metadata"
            ][
                "serialization_format"
            ] = "not_cdr"

            _write_episode(
                episode,
                metadata=metadata,
            )

            with self.assertRaisesRegex(
                McapIngestError,
                "serialization_format",
            ):
                read_episode_metadata(
                    episode
                )


if __name__ == "__main__":
    unittest.main()
