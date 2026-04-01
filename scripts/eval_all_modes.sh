#!/bin/bash
# Run all evaluation modes for a specific paper
PAPER=${1:-paper_1}
CONFIG=${2:-configs/cc_sonnet4.yaml}

for mode in rubric hallucination citation; do
    echo "=== Evaluating $PAPER with mode=$mode ==="
    python run_evaluation.py --config-path "$CONFIG" --paper "$PAPER" --eval-mode "$mode"
done
