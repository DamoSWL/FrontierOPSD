"""Run a configured frontier experiment: python -m frontier_opsd config.json."""

import argparse
import importlib
import json

from .trainer import FrontierConfig, FrontierTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON with factory, backend, and frontier configuration")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    module_name, function_name = config["factory"].split(":")
    factory = getattr(importlib.import_module(module_name), function_name)
    policy, env, critic, tasks = factory(config.get("backend", {}))
    trainer = FrontierTrainer(policy, env, critic, FrontierConfig(**config.get("frontier", {})))
    for result in trainer.fit(tasks):
        print(json.dumps({"task_id": result["record"]["task_id"], "epoch": result["record"]["epoch"],
                          "status": result["status"], "events": result["events"],
                          "comparison": result["record"]["comparison"],
                          "policy_version": trainer.policy_version}), flush=True)


if __name__ == "__main__":
    main()
