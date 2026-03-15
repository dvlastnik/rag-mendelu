#!/usr/bin/env bash
set -euo pipefail

trap 'echo ""; echo "Interrupted — stopping."; kill 0; exit 130' INT TERM

# ── Defaults ──────────────────────────────────────────────────────────────────
COLLECTION="drough"
QUESTIONS=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --collection)
            COLLECTION="$2"; shift 2 ;;
        --questions)
            QUESTIONS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--collection <name>] [--questions <path>]"
            exit 1 ;;
    esac
done

# ── Discover models (sorted by size ascending: smallest first) ────────────────
# Note: mapfile requires bash 4+; macOS ships bash 3.2, so use while-read instead
MODELS=()
while IFS= read -r line; do
    MODELS+=("$line")
done < <(
    ollama list | tail -n +2 | awk '{
        val = $3 + 0
        unit = $4
        if (unit == "MB") val = val / 1024
        print val, $1
    }' | sort -k1 -n | awk '{print $2}'
)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No models found in 'ollama list'. Make sure Ollama is running and at least one model is loaded."
    exit 1
fi

echo "═══════════════════════════════════════════════"
echo " Collection : $COLLECTION"
echo " Questions  : ${QUESTIONS:-<not set>}"
echo " Models     : ${MODELS[*]}"
echo "═══════════════════════════════════════════════"

# Pre-compute date once — mirrors generate_answers.get_results_filepath() format
DATE=$(date +%y%m%d)

# ── Run tests per model ───────────────────────────────────────────────────────
# Parallel indexed arrays (bash 3.2 compatible — no declare -A)
RESULTS=()
REPORT_PATHS=()

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    echo ""
    echo "▶ Testing model: $MODEL"
    echo "───────────────────────────────────────────────"

    CMD=(uv run pytest tests/rag/test_rag_custom.py
         --model "$MODEL"
         --collection-name "$COLLECTION")

    if [[ -n "$QUESTIONS" ]]; then
        CMD+=(--questions "$QUESTIONS")
    fi

    if "${CMD[@]}"; then
        RESULTS[$i]="PASS"
    else
        RESULTS[$i]="FAIL"
    fi

    # Replicate get_results_filepath() + evaluation_logger path logic:
    #   directory : / and : → _   (get_results_filepath replaces both)
    #   filename  : only / → _    (evaluation_logger only replaces /)
    DIR_MODEL=$(echo "$MODEL" | sed 's|[/:]|_|g')
    FILE_MODEL=$(echo "$MODEL" | sed 's|/|_|g')

    if [[ -n "$QUESTIONS" ]]; then
        QUESTIONS_STEM=$(basename "$QUESTIONS" .json)
        REPORT_DIR="tests/rag/results/${QUESTIONS_STEM}_answers_${DATE}_${DIR_MODEL}"
    else
        REPORT_DIR="tests/rag/results/answers_${DATE}_${DIR_MODEL}"
    fi

    REPORT_PATHS[$i]="${REPORT_DIR}/judgement_report_${FILE_MODEL}.json"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo " SUMMARY"
echo "═══════════════════════════════════════════════"
OVERALL=0
for i in "${!MODELS[@]}"; do
    if [[ "${RESULTS[$i]}" == "PASS" ]]; then
        echo "  ✅  ${MODELS[$i]}"
    else
        echo "  ❌  ${MODELS[$i]}"
        OVERALL=1
    fi
done
echo "═══════════════════════════════════════════════"

# ── Generate evaluation_matrix.csv ────────────────────────────────────────────
echo ""
echo "Generating evaluation_matrix.csv..."

# Build flat arg list: model1 path1 model2 path2 ...
python_args=()
for i in "${!MODELS[@]}"; do
    python_args+=("${MODELS[$i]}" "${REPORT_PATHS[$i]}")
done

uv run python - "${python_args[@]}" << 'PYEOF'
import json, csv, os, sys

pairs = sys.argv[1:]  # model path model path ...
rows = []
for i in range(0, len(pairs), 2):
    model = pairs[i]
    path  = pairs[i + 1]
    if not os.path.exists(path):
        print(f"  ⚠  Report not found, skipping: {path}", file=sys.stderr)
        continue
    with open(path) as f:
        meta = json.load(f)['metadata']
        ml_metrics = meta['ml_metrics']
    rows.append({
        'model':             model,
        'success_rate':      meta.get('success_rate', '')
        'accuracy_percent':  ml_metrics.get('accuracy_percent',  ''),
        'precision_percent': ml_metrics.get('precision_percent', ''),
        'recall_percent':    ml_metrics.get('recall_percent',    ''),
        'f1_percent':        ml_metrics.get('f1_percent',        ''),
        'duration_minutes':  meta.get('duration_minutes',  ''),
    })

out = 'evaluation_matrix.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'model', 'success_rate', 'accuracy_percent', 'precision_percent',
        'recall_percent', 'f1_percent', 'duration_minutes',
    ])
    w.writeheader()
    w.writerows(rows)
print(f"  CSV saved → {out}")
PYEOF

exit $OVERALL
