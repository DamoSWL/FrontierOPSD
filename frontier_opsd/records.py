"""Per-task epoch records for analysis only; never a training replay buffer."""

import hashlib
import json
from pathlib import Path


def task_id(environment, identity):
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
    return environment + ':' + hashlib.sha256(encoded.encode()).hexdigest()[:24]


def compare(current, previous):
    result = {"previous_epoch": previous["epoch"] if previous else None,
              "result": "uncertain", "reason": "No preceding epoch record for this task."}
    if previous is None:
        return result
    if current["success"] != previous["success"]:
        result.update(result="improved" if current["success"] else "regressed",
                      reason="Task outcome changed from failure to success." if current["success"]
                      else "Task outcome changed from success to failure.")
    elif current["success"]:
        result.update(result="unchanged", reason="Both sampled rollouts succeeded.")
    else:
        old = previous.get("diagnosis")
        new = current.get("diagnosis")
        if old is None or new is None:
            result["reason"] = "A failed rollout lacks a valid mistake diagnosis."
            return result
        a, b = old["mistaken_step_index"], new["mistaken_step_index"]
        prefix = min(a, b)
        def transitions(record):
            return [(s["observation_before"], s["action"], s["observation_after"])
                    for s in record["steps"]]
        old_steps, new_steps = transitions(previous), transitions(current)
        if old_steps[:prefix] != new_steps[:prefix]:
            result["reason"] = "Different reasoning/action paths; mistake indices alone are not comparable."
        elif a == b:
            result.update(result="unchanged", reason="First mistake remains at the same aligned prefix depth.")
        else:
            longer = new_steps[:b] if b > a else old_steps[:a]
            if len(set(longer)) != len(longer) or any(before == after for before, _, after in longer[prefix:]):
                result["reason"] = "Additional steps include repetition or unchanged observations."
            else:
                result.update(result="improved" if b > a else "regressed",
                    reason="First diagnosed mistake moved along an aligned, non-repeating prefix; single-rollout evidence only.")
    return result


class RolloutRecords:
    def __init__(self, path=None, latest=None):
        self.path = Path(path) if path else None
        self.latest = latest or {}

    def rewind(self, offset):
        """Archive records newer than a restored model checkpoint."""
        if self.path is None or not self.path.exists():
            if offset:
                raise RuntimeError("Checkpoint rollout journal is missing")
            return
        data = self.path.read_bytes()
        if not 0 <= offset <= len(data):
            raise RuntimeError("Checkpoint rollout journal offset is invalid")
        if offset < len(data):
            archive = self.path.with_suffix(self.path.suffix + ".uncheckpointed")
            with archive.open("ab") as stream:
                stream.write(data[offset:])
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_bytes(data[:offset])
            temporary.replace(self.path)

    def add(self, record):
        previous = self.latest.get(record["task_id"])
        if previous is not None and previous["epoch"] >= record["epoch"]:
            raise ValueError("A task may have only one rollout record per epoch")
        record["comparison"] = compare(record, previous)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.latest[record["task_id"]] = record
        return record
