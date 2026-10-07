#!/usr/bin/env bash
set -euo pipefail

# Reuse SDAR's complete launch configuration and original data preparation.
# Usage: bash examples/frontier_opsd/run_sdar_dataset.sh alfworld [SDAR arguments]
environment=${1:?Choose alfworld, webshop, or search}
shift
case "$environment" in
    alfworld|webshop|search) ;;
    *) echo "Supported SDAR environments: alfworld, webshop, search" >&2; exit 2 ;;
esac
export TRAINER_MODULE=verl.trainer.main_frontier_opsd
bash "examples/sdar_trainer/run_${environment}_3b.sh" "$@"
