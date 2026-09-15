MODEL=$1
DEFAULT_N=4

COMPS=(
  "arxiv/april"
  "arxiv/may"
  "arxiv/june"
  "arxiv_false/april"
  "arxiv_false/may"
  "arxiv_false/june"
)

# Per-comp n overrides
declare -A N_VALUES=(
  # add more overrides here if needed
  ["arxiv/april"]=3
  ["arxiv_false/april"]=2
  ["arxiv/may"]=3
  ["arxiv_false/may"]=2
  ["arxiv/june"]=3
  ["arxiv_false/june"]=2
)

COMP_N_OVERRIDES=()
for comp in "${COMPS[@]}"; do
  if [[ -n "${N_VALUES[$comp]}" ]]; then
    COMP_N_OVERRIDES+=("${comp}=${N_VALUES[$comp]}")
  fi
done

echo "Running on comps: ${COMPS[*]} with model $MODEL (default n=$DEFAULT_N, overrides: ${COMP_N_OVERRIDES[*]})"
uv run python scripts/run.py \
  --comp "${COMPS[@]}" \
  --models "$MODEL" \
  --n "$DEFAULT_N" \
  --comp-n "${COMP_N_OVERRIDES[@]}"

for comp in "${COMPS[@]}"; do
  uv run python scripts/curation/check.py  --comp "$comp" --model-config gemini/gemini-38-flash-low
done

for comp in "${COMPS[@]}"; do
  if [[ "$comp" == arxiv_false/* ]]; then
    uv run python scripts/judge/judge.py --comp "$comp" --models "$MODEL"
  fi
done
