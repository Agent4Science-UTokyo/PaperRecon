#!/bin/bash
# Rubric evaluation for all papers with all configs
configs=(
    configs/cc_sonnet4.yaml
    configs/cc_sonnet46.yaml
    configs/cc_teams_sonnet46.yaml
    configs/codex_gpt5.yaml
    configs/codex_gpt54.yaml
)

for config in "${configs[@]}"; do
    echo "=== Evaluating with $config ==="
    python run_evaluation.py --config-path "$config" --all --eval-mode rubric
done
