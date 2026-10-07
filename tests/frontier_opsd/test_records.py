import json
from pathlib import Path
import tempfile
import unittest
from frontier_opsd.records import RolloutRecords, compare, task_id


def record(epoch, success=False, frontier=0, steps=None):
    return {"task_id": "A", "epoch": epoch, "policy_version": epoch - 1,
            "success": success, "steps": steps or [],
            "diagnosis": None if success else {"mistaken_step_index": frontier, "hint": "hint"}}


class RecordTests(unittest.TestCase):
    def test_jsonl_records_and_outcome_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            store = RolloutRecords(Path(folder) / 'rollouts.jsonl')
            store.add(record(1))
            offset = store.path.stat().st_size
            improved = store.add(record(2, True))
            self.assertEqual(improved['comparison']['result'], 'improved')
            self.assertEqual(improved['comparison']['previous_epoch'], 1)
            regressed = store.add(record(3))
            self.assertEqual(regressed['comparison']['result'], 'regressed')
            lines = [json.loads(line) for line in store.path.read_text().splitlines()]
            self.assertEqual(len(lines), 3)
            restored = RolloutRecords(latest=store.latest)
            self.assertEqual(restored.add(record(4, True))['comparison']['previous_epoch'], 3)
            with self.assertRaises(ValueError):
                restored.add(record(4))
            store.rewind(offset)
            self.assertEqual(len(store.path.read_text().splitlines()), 1)
            self.assertEqual(len(store.path.with_suffix('.jsonl.uncheckpointed').read_text().splitlines()), 2)

    def test_later_index_on_different_or_repetitive_path_is_uncertain(self):
        def step(before, action, after):
            return dict(observation_before=before, action=action, observation_after=after)
        old = record(1, frontier=1, steps=[step('a', 'go', 'b')])
        new = record(2, frontier=2, steps=[step('x', 'go', 'y'), step('y', 'go', 'z')])
        self.assertEqual(compare(new, old)['result'], 'uncertain')
        new['steps'] = [step('a', 'go', 'b'), step('b', 'wait', 'b')]
        self.assertEqual(compare(new, old)['result'], 'uncertain')
        new['steps'][1] = step('b', 'go', 'c')
        self.assertEqual(compare(new, old)['result'], 'improved')

    def test_task_ids_do_not_depend_on_dict_order(self):
        self.assertEqual(task_id('search', {'q': 'abc', 'id': 1}),
                         task_id('search', {'id': 1, 'q': 'abc'}))
