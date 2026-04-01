#!/bin/bash
# Write all papers with all configs
configs=(
    configs/cc_sonnet4.yaml
    configs/cc_sonnet46.yaml
    configs/cc_teams_sonnet46.yaml
    configs/codex_gpt5.yaml
    configs/codex_gpt54.yaml
)

for config in "${configs[@]}"; do
    echo "=== Running with $config ==="
    python launch_writing.py --config "$config" --all
done
