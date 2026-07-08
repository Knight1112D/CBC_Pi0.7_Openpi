"""生成 CBC RECAP/MEM/RL token sidecar 标签文件。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from openpi.training import cbc_training


def _load_episode_metadata(path: Path) -> dict[int, dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload["episodes"] if isinstance(payload, dict) and "episodes" in payload else payload
    else:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    output = {}
    for row in rows:
        episode = int(row["episode_index"])
        output[episode] = dict(row)
    return output


def _load_interventions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    return payload["records"] if isinstance(payload, dict) and "records" in payload else payload


def _in_intervention(episode: int, frame: int, intervals: list[dict[str, Any]]) -> bool:
    for interval in intervals:
        if int(interval.get("episode_index", -1)) != episode:
            continue
        start = int(interval.get("start_frame", interval.get("start_index", 0)))
        end = int(interval.get("end_frame", interval.get("end_index", start)))
        if start <= frame <= end:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interventions", type=Path)
    parser.add_argument("--frames-per-episode", type=int, default=1)
    parser.add_argument("--success-field", default="success")
    args = parser.parse_args()

    episodes = _load_episode_metadata(args.episode_metadata)
    interventions = _load_interventions(args.interventions)
    records = []
    for episode, metadata in sorted(episodes.items()):
        success = str(metadata.get(args.success_field, "0")).lower() in {"1", "true", "yes", "y"}
        for frame in range(args.frames_per_episode):
            is_intervention = _in_intervention(episode, frame, interventions)
            advantage = 1.0 if success and not is_intervention else 0.0
            records.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "advantage_indicator": advantage,
                    "use_advantage": True,
                    "is_human_intervention": is_intervention,
                    "rl_token_weight": 1.0,
                    "memory": metadata.get("memory", ""),
                    "next_memory": metadata.get("next_memory", ""),
                }
            )
    cbc_training.write_jsonl(records, args.output)
    print(f"wrote {len(records)} sidecar rows to {args.output}")


if __name__ == "__main__":
    main()
