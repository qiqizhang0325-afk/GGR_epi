#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

set -euo pipefail

PYMOD="plantcad2.pipeline"
# legacy script kept under scripts/ for reference
LOGDIR="logs"

usage() {
  cat <<'EOF'
Usage:
  run_pipeline.sh <reference|sample|all> \
    --fasta FASTA --vcf VCF --gff GFF \
    --outdir OUTDIR --chrs Chr1,Chr2,... \
    --gpus 0[,1,2,...] \
    [--samples samples.txt] \
    [--per_chr_outdir 0|1] \
    [--ref_bg 0|1] \
    [--wait_timeout_sec 0|SECONDS] \
    [--poll_sec SECONDS]

Key rules:
  - sample MUST run AFTER reference outputs are ready.
  - Reference is considered ready only when:
      ref_gene_embeddings.pkl and intergenic_embeddings.pkl
    exist AND are stable (size>0, and size/mtime unchanged across polls).
  - sample will auto-skip samples whose output pickle already exists.

Options:
  --per_chr_outdir 1  : write outputs into OUTDIR/ChrX/ to avoid overwrite risk
  --ref_bg 1          : run reference in background (nohup). The script will still WAIT before starting sample.
  --wait_timeout_sec  : 0 means wait forever (default: 0)
  --poll_sec          : polling interval in seconds (default: 60)

Examples:
  bash run_pipeline.sh reference --fasta A --vcf B --gff C --outdir out --chrs Chr1,Chr2,Chr3,Chr4,Chr5 --gpus 0
  bash run_pipeline.sh sample    --fasta A --vcf B --gff C --outdir out --chrs Chr1,Chr2,Chr3,Chr4,Chr5 --gpus 0,1 --samples samples.txt
  bash run_pipeline.sh all       --fasta A --vcf B --gff C --outdir out --chrs Chr1,Chr2,Chr3,Chr4,Chr5 --gpus 0,1 --samples samples.txt --per_chr_outdir 1 --ref_bg 1
EOF
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi
MODE="$1"; shift

FASTA=""
VCF=""
GFF=""
OUTDIR=""
CHRS=""
GPUS=""
SAMPLES_FILE=""
PER_CHR_OUTDIR="0"
REF_BG="0"
WAIT_TIMEOUT_SEC="0"
POLL_SEC="60"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fasta) FASTA="$2"; shift 2;;
    --vcf) VCF="$2"; shift 2;;
    --gff) GFF="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --chrs) CHRS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --samples) SAMPLES_FILE="$2"; shift 2;;
    --per_chr_outdir) PER_CHR_OUTDIR="$2"; shift 2;;
    --ref_bg) REF_BG="$2"; shift 2;;
    --wait_timeout_sec) WAIT_TIMEOUT_SEC="$2"; shift 2;;
    --poll_sec) POLL_SEC="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$FASTA" || -z "$VCF" || -z "$GFF" || -z "$OUTDIR" || -z "$CHRS" || -z "$GPUS" ]]; then
  echo "Missing required args."
  usage
  exit 1
fi

python -c "import importlib; importlib.import_module('${PYMOD}')" >/dev/null 2>&1 || { echo "Cannot import module: $PYMOD (did you install the package / set PYTHONPATH?)"; exit 1; }

mkdir -p "$LOGDIR"
CHRS_ARR=($(echo "$CHRS" | tr ',' ' '))
GPUS_ARR=($(echo "$GPUS" | tr ',' ' '))
NGPU="${#GPUS_ARR[@]}"

outdir_for_chr() {
  local chr="$1"
  if [[ "$PER_CHR_OUTDIR" == "1" ]]; then
    echo "${OUTDIR}/${chr}"
  else
    echo "${OUTDIR}"
  fi
}

# ---------- reference file stability check ----------
# We consider a reference file "stable" if:
# - it exists
# - size > 0
# - and its (size, mtime) signature stays the same for 2 consecutive polls
declare -A _prev_sig
declare -A _stable

_file_sig() {
  local f="$1"
  # Linux stat: size + mtime(epoch)
  stat -c "%s %Y" "$f" 2>/dev/null || echo ""
}

_update_stability() {
  local f="$1"
  local sig
  sig="$(_file_sig "$f")"

  if [[ -z "$sig" ]]; then
    _stable["$f"]="0"
    _prev_sig["$f"]=""
    return
  fi

  local size
  size="${sig%% *}"
  if [[ "$size" -le 0 ]]; then
    _stable["$f"]="0"
    _prev_sig["$f"]="$sig"
    return
  fi

  if [[ "${_prev_sig["$f"]:-}" == "$sig" ]]; then
    _stable["$f"]="1"
  else
    _stable["$f"]="0"
    _prev_sig["$f"]="$sig"
  fi
}

_is_stable() {
  local f="$1"
  [[ "${_stable["$f"]:-0}" == "1" ]]
}

ref_ready_for_chr() {
  local chr="$1"
  local odir
  odir="$(outdir_for_chr "$chr")"

  local f1="${odir}/ref_gene_embeddings.pkl"
  local f2="${odir}/intergenic_embeddings.pkl"

  _update_stability "$f1"
  _update_stability "$f2"

  _is_stable "$f1" && _is_stable "$f2"
}

wait_for_reference_all_chrs() {
  local start_ts
  start_ts="$(date +%s)"

  echo "[wait] Waiting for reference outputs for all chromosomes..."
  echo "[wait] Expecting stable files: ref_gene_embeddings.pkl + intergenic_embeddings.pkl (size/mtime stable across polls)"

  while true; do
    local missing=0
    for chr in "${CHRS_ARR[@]}"; do
      local odir
      odir="$(outdir_for_chr "$chr")"
      if ! ref_ready_for_chr "$chr"; then
        missing=1
        echo "[wait] NOT ready: ${chr} @ ${odir}"
      fi
    done

    if [[ "$missing" -eq 0 ]]; then
      echo "[wait] Reference ready for all chromosomes."
      return 0
    fi

    if [[ "$WAIT_TIMEOUT_SEC" != "0" ]]; then
      local now
      now="$(date +%s)"
      local elapsed=$(( now - start_ts ))
      if [[ "$elapsed" -ge "$WAIT_TIMEOUT_SEC" ]]; then
        echo "[wait] TIMEOUT after ${elapsed}s. Reference not ready."
        exit 2
      fi
    fi

    sleep "$POLL_SEC"
  done
}

run_reference_foreground() {
  local gpu="$1"
  echo "[ref] Running reference on GPU ${gpu} (foreground)..."
  for chr in "${CHRS_ARR[@]}"; do
    local odir
    odir="$(outdir_for_chr "$chr")"
    mkdir -p "$odir"
    echo "[ref] ${chr} -> ${odir}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
      python -m "$PYMOD" \
        --mode reference \
        --chrom "$chr" \
        --fasta "$FASTA" \
        --vcf "$VCF" \
        --gff "$GFF" \
        --output "$odir" \
      > "${LOGDIR}/ref_${chr}.gpu${gpu}.log" 2>&1
  done
  echo "[ref] Reference finished."
}


run_reference_background() {
  local gpu="$1"
  echo "[ref] Submitting reference on GPU ${gpu} (background nohup)..."
  nohup bash -c "
set -euo pipefail
for CHR in ${CHRS_ARR[*]}; do
  if [[ \"${PER_CHR_OUTDIR}\" == \"1\" ]]; then
    ODIR=\"${OUTDIR}/\${CHR}\"
  else
    ODIR=\"${OUTDIR}\"
  fi
  mkdir -p \"\${ODIR}\"

  CUDA_VISIBLE_DEVICES=${gpu} python -m ${PYMOD} \
    --mode reference \
    --chrom \${CHR} \
    --fasta \"${FASTA}\" \
    --vcf \"${VCF}\" \
    --gff \"${GFF}\" \
    --output \"\${ODIR}\" \
    > ${LOGDIR}/ref_\${CHR}.gpu${gpu}.log 2>&1
done
" > "${LOGDIR}/reference.gpu${gpu}.nohup.log" 2>&1 &

  echo "[ref] Reference submitted: ${LOGDIR}/reference.gpu${gpu}.nohup.log"
}

# ---------- sample helpers ----------
# sample output naming rule:
# sample_<id>_<id>_embeddings.pkl
split_samples_evenly() {
  local infile="$1"
  local n="$2"
  local outprefix="$3"

  [[ -f "$infile" ]] || { echo "Samples file not found: $infile"; exit 1; }

  local tmp="${outprefix}.clean.txt"
  awk 'NF{print $0}' "$infile" > "$tmp"

  local total
  total=$(wc -l < "$tmp" | tr -d ' ')
  [[ "$total" -gt 0 ]] || { echo "Samples file is empty: $infile"; exit 1; }

  local per=$(( (total + n - 1) / n ))

  rm -f "${outprefix}".part_*.txt
  split -d -l "$per" "$tmp" "${outprefix}.part_"

  for i in $(seq 0 $((n-1))); do
    local f
    f=$(printf "%s.part_%02d" "$outprefix" "$i")
    [[ -f "$f" ]] || : > "$f"
    mv "$f" "${outprefix}.part_${i}.txt"
  done
  rm -f "$tmp"
}

run_sample_worker_nohup() {
  local gpu="$1"
  local part_file="$2"

  [[ -f "$part_file" ]] || { echo "[sample] part file not found: $part_file"; exit 1; }

  # Robust GPU mapping:
  # bind physical GPU via CUDA_VISIBLE_DEVICES, and use cuda:0 inside the process.
  nohup bash -c "
set -euo pipefail
for CHR in ${CHRS_ARR[*]}; do
  if [[ \"${PER_CHR_OUTDIR}\" == \"1\" ]]; then
    ODIR=\"${OUTDIR}/\${CHR}\"
  else
    ODIR=\"${OUTDIR}\"
  fi
  mkdir -p \"\${ODIR}\"

  FILTERED=\"${LOGDIR}/samples_gpu${gpu}_\${CHR}.todo.txt\"
  awk 'NF{print \$0}' \"${part_file}\" | while read -r SID; do
    if [[ -f \"\${ODIR}/sample_\${SID}_\${SID}_embeddings.pkl\" ]]; then
      echo \"[sample] skip done: \${SID} (found in \${ODIR})\" >&2
    else
      echo \"\${SID}\"
    fi
  done > \"\${FILTERED}\"

  if [[ ! -s \"\${FILTERED}\" ]]; then
    echo \"[sample] GPU ${gpu} \${CHR}: no remaining samples.\" >&2
    continue
  fi

  CUDA_VISIBLE_DEVICES=${gpu} python -m ${PYMOD} \
    --mode sample \
    --chrom \${CHR} \
    --device cuda:0 \
    --samples \$(xargs -a \"\${FILTERED}\") \
    --fasta \"${FASTA}\" \
    --vcf \"${VCF}\" \
    --gff \"${GFF}\" \
    --output \"\${ODIR}\" \
    > ${LOGDIR}/sample_\${CHR}.gpu${gpu}.log 2>&1
done
" > "${LOGDIR}/sample.gpu${gpu}.nohup.log" 2>&1 &

  echo "[sample] Submitted GPU ${gpu} (part: ${part_file}) -> ${LOGDIR}/sample.gpu${gpu}.nohup.log"
}

run_sample_multigpu() {
  [[ -n "$SAMPLES_FILE" ]] || { echo "--samples is required for sample mode."; exit 1; }

  local split_prefix="${LOGDIR}/samples_split"
  split_samples_evenly "$SAMPLES_FILE" "$NGPU" "$split_prefix"

  for idx in "${!GPUS_ARR[@]}"; do
    local gpu="${GPUS_ARR[$idx]}"
    local part="${split_prefix}.part_${idx}.txt"
    run_sample_worker_nohup "$gpu" "$part"
  done
}

# ---------- main ----------
case "$MODE" in
  reference)
    if [[ "$REF_BG" == "1" ]]; then
      run_reference_background "${GPUS_ARR[0]}"
    else
      run_reference_foreground "${GPUS_ARR[0]}"
    fi
    ;;
  sample)
    wait_for_reference_all_chrs
    run_sample_multigpu
    ;;
  all)
    if [[ "$REF_BG" == "1" ]]; then
      run_reference_background "${GPUS_ARR[0]}"
    else
      run_reference_foreground "${GPUS_ARR[0]}"
    fi
    wait_for_reference_all_chrs
    run_sample_multigpu
    ;;
  *)
    echo "Unknown mode: $MODE"
    usage
    exit 1
    ;;
esac
