#!/usr/bin/env bash
# Unified pipeline runner for PlantCAD2 reference + sample embedding
# - Ensures PYTHONPATH is set for module execution
# - Normalizes reference outputs into per-chr directories and creates alias filenames
# - Runs sample embedding (optionally multi-GPU) with correct output naming + skip logic

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

set -euo pipefail
# --- HARD GPU REQUIREMENT CHECK ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FATAL] NVIDIA GPU not detected (nvidia-smi not found)."
  echo "        This pipeline REQUIRES a GPU-enabled workspace."
  echo "        On SURF Research Cloud: please start a GPU workspace."
  exit 10
fi

python - <<'PY'
import sys, torch
print("[GPU CHECK] torch version:", torch.__version__)
print("[GPU CHECK] torch.version.cuda:", torch.version.cuda)
print("[GPU CHECK] cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("[FATAL] CUDA is NOT available to PyTorch.")
    print("        Likely driver / CUDA mismatch.")
    sys.exit(11)
print("[GPU CHECK] GPU name:", torch.cuda.get_device_name(0))
PY

PYMOD="plantcad2.pipeline"
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
  --per_chr_outdir 1  : write outputs into OUTDIR/ChrX/ to avoid overwrite risk (recommended)
  --ref_bg 1          : run reference in background (nohup). The script will still WAIT before starting sample.
  --wait_timeout_sec  : 0 means wait forever (default: 0)
  --poll_sec          : polling interval in seconds (default: 60)

Sample output naming (current pipeline):
  sample_<SID>.<CHR>_embeddings.pkl
Example:
  outputs_test/Chr1/sample_9381_9381.Chr1_embeddings.pkl

Examples:
  bash run_pipeline.sh reference --fasta A --vcf B --gff C --outdir out --chrs Chr1,Chr2,Chr3,Chr4,Chr5 --gpus 0 --per_chr_outdir 1
  bash run_pipeline.sh sample    --fasta A --vcf B --gff C --outdir out --chrs Chr1,Chr2,Chr3,Chr4,Chr5 --gpus 0,1 --samples samples.txt --per_chr_outdir 1
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


# --- ensure VCF is bgzipped (.vcf.gz) and indexed (.tbi/.csi) ---
ensure_vcf_gz() {
  # If user passed .vcf, compress to .vcf.gz. If user passed .vcf.gz, keep as-is.
  if [[ "$VCF" == *.vcf ]]; then
    local gz="${VCF}.gz"

    if [[ ! -f "$VCF" ]]; then
      echo "[ERROR] VCF not found: $VCF"
      exit 1
    fi

    if [[ -f "$gz" ]]; then
      echo "[INFO] Found existing compressed VCF: $gz"
    else
      echo "[INFO] Compressing VCF -> $gz"
      if command -v bgzip >/dev/null 2>&1; then
        bgzip -c "$VCF" > "$gz"
      elif command -v bcftools >/dev/null 2>&1; then
        bcftools view -Oz -o "$gz" "$VCF"
      else
        echo "[ERROR] Need bgzip or bcftools to create .vcf.gz"
        echo "        Try: module load htslib (for bgzip/tabix) or install bcftools"
        exit 1
      fi
    fi

    VCF="$gz"
  fi

  # At this point, require .vcf.gz
  if [[ "$VCF" != *.vcf.gz ]]; then
    echo "[ERROR] --vcf must be .vcf or .vcf.gz, got: $VCF"
    exit 1
  fi

  if [[ ! -f "$VCF" ]]; then
    echo "[ERROR] VCF not found: $VCF"
    exit 1
  fi

  # Ensure an index exists (best-effort)
  if [[ -f "${VCF}.tbi" || -f "${VCF}.csi" ]]; then
    echo "[INFO] VCF index exists."
    return 0
  fi

  echo "[INFO] Building VCF index for: $VCF"
  if command -v tabix >/dev/null 2>&1; then
    tabix -p vcf "$VCF"
  elif command -v bcftools >/dev/null 2>&1; then
    bcftools index -f "$VCF"
  else
    echo "[WARN] No tabix/bcftools found; skipping index creation."
    echo "       Some tools may fail without .tbi/.csi index."
  fi
}
ensure_vcf_gz


# Import check (must be valid Python)
python -c "import importlib; importlib.import_module('${PYMOD}')" >/dev/null 2>&1 || {
  echo "Cannot import module: ${PYMOD} (did you install the package / set PYTHONPATH?)"
  echo "Hint: export PYTHONPATH=\"${REPO_ROOT}/src:${REPO_ROOT}:\$PYTHONPATH\""
  exit 1
}

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

# ---------- normalize / link reference outputs ----------
# The pipeline may output reference pickles as:
#   OUTDIR/ref_gene_embeddings.Chr1.pkl
#   OUTDIR/intergenic_embeddings.Chr1.pkl
# or in per-chr dir:
#   OUTDIR/Chr1/ref_gene_embeddings.Chr1.pkl
#   OUTDIR/Chr1/intergenic_embeddings.Chr1.pkl
#
# We standardize to per-chr dir and create aliases expected by wait/sample:
#   OUTDIR/Chr1/ref_gene_embeddings.pkl  -> ref_gene_embeddings.Chr1.pkl
#   OUTDIR/Chr1/intergenic_embeddings.pkl -> intergenic_embeddings.Chr1.pkl
ensure_reference_aliases_for_chr() {
  local chr="$1"
  local chrdir="${OUTDIR}/${chr}"
  mkdir -p "${chrdir}"

  local ref_chr="${chrdir}/ref_gene_embeddings.${chr}.pkl"
  local int_chr="${chrdir}/intergenic_embeddings.${chr}.pkl"
  local ref_root="${OUTDIR}/ref_gene_embeddings.${chr}.pkl"
  local int_root="${OUTDIR}/intergenic_embeddings.${chr}.pkl"

  # If reference outputs exist in OUTDIR root, copy them into per-chr dir
  if [[ ! -s "${ref_chr}" && -s "${ref_root}" ]]; then
    cp -f "${ref_root}" "${ref_chr}"
  fi
  if [[ ! -s "${int_chr}" && -s "${int_root}" ]]; then
    cp -f "${int_root}" "${int_chr}"
  fi

  # Create no-suffix aliases (used by wait/sample)
  # Try symlink first; if it fails, fallback to copy
  if [[ -s "${ref_chr}" ]]; then
    ln -sf "ref_gene_embeddings.${chr}.pkl" "${chrdir}/ref_gene_embeddings.pkl" 2>/dev/null || true
    if [[ ! -s "${chrdir}/ref_gene_embeddings.pkl" ]]; then
      cp -f "${ref_chr}" "${chrdir}/ref_gene_embeddings.pkl"
    fi
  fi

  if [[ -s "${int_chr}" ]]; then
    ln -sf "intergenic_embeddings.${chr}.pkl" "${chrdir}/intergenic_embeddings.pkl" 2>/dev/null || true
    if [[ ! -s "${chrdir}/intergenic_embeddings.pkl" ]]; then
      cp -f "${int_chr}" "${chrdir}/intergenic_embeddings.pkl"
    fi
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

  # Always normalize / alias first
  ensure_reference_aliases_for_chr "$chr"

  local chrdir="${OUTDIR}/${chr}"

  local f1a="${chrdir}/ref_gene_embeddings.pkl"
  local f2a="${chrdir}/intergenic_embeddings.pkl"
  local f1b="${chrdir}/ref_gene_embeddings.${chr}.pkl"
  local f2b="${chrdir}/intergenic_embeddings.${chr}.pkl"

  # Prefer no-suffix alias; fallback to suffixed files
  local f1="$f1a"; [[ -e "$f1" ]] || f1="$f1b"
  local f2="$f2a"; [[ -e "$f2" ]] || f2="$f2b"

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
      local chrdir="${OUTDIR}/${chr}"
      if ! ref_ready_for_chr "$chr"; then
        missing=1
        echo "[wait] NOT ready: ${chr} @ ${chrdir}"
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

    # normalize outputs for sample/wait
    ensure_reference_aliases_for_chr "$chr"
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
  mkdir -p \"${OUTDIR}/\${CHR}\"

  CUDA_VISIBLE_DEVICES=${gpu} python -m ${PYMOD} \
    --mode reference \
    --chrom \${CHR} \
    --fasta \"${FASTA}\" \
    --vcf \"${VCF}\" \
    --gff \"${GFF}\" \
    --output \"\${ODIR}\" \
    > ${LOGDIR}/ref_\${CHR}.gpu${gpu}.log 2>&1

  # Best-effort: normalize aliases (wait will also do this)
  if [[ -s \"${OUTDIR}/ref_gene_embeddings.\${CHR}.pkl\" && ! -s \"${OUTDIR}/\${CHR}/ref_gene_embeddings.\${CHR}.pkl\" ]]; then
    cp -f \"${OUTDIR}/ref_gene_embeddings.\${CHR}.pkl\" \"${OUTDIR}/\${CHR}/ref_gene_embeddings.\${CHR}.pkl\" 2>/dev/null || true
  fi
  if [[ -s \"${OUTDIR}/intergenic_embeddings.\${CHR}.pkl\" && ! -s \"${OUTDIR}/\${CHR}/intergenic_embeddings.\${CHR}.pkl\" ]]; then
    cp -f \"${OUTDIR}/intergenic_embeddings.\${CHR}.pkl\" \"${OUTDIR}/\${CHR}/intergenic_embeddings.\${CHR}.pkl\" 2>/dev/null || true
  fi
  if [[ -s \"${OUTDIR}/\${CHR}/ref_gene_embeddings.\${CHR}.pkl\" ]]; then
    ln -sf \"ref_gene_embeddings.\${CHR}.pkl\" \"${OUTDIR}/\${CHR}/ref_gene_embeddings.pkl\" 2>/dev/null || true
  fi
  if [[ -s \"${OUTDIR}/\${CHR}/intergenic_embeddings.\${CHR}.pkl\" ]]; then
    ln -sf \"intergenic_embeddings.\${CHR}.pkl\" \"${OUTDIR}/\${CHR}/intergenic_embeddings.pkl\" 2>/dev/null || true
  fi
done
" > "${LOGDIR}/reference.gpu${gpu}.nohup.log" 2>&1 &

  echo "[ref] Reference submitted: ${LOGDIR}/reference.gpu${gpu}.nohup.log"
}

# ---------- sample helpers ----------
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

  # bind physical GPU via CUDA_VISIBLE_DEVICES, and use cuda:0 inside the process.
  nohup bash -c "
set -euo pipefail
for CHR in ${CHRS_ARR[*]}; do
  # Ensure reference outputs are normalized before running sample
  # (wait should have done this, but this makes worker standalone)
  mkdir -p \"${OUTDIR}/\${CHR}\"

  if [[ \"${PER_CHR_OUTDIR}\" == \"1\" ]]; then
    ODIR=\"${OUTDIR}/\${CHR}\"
  else
    ODIR=\"${OUTDIR}\"
  fi
  mkdir -p \"\${ODIR}\"

  FILTERED=\"${LOGDIR}/samples_gpu${gpu}_\${CHR}.todo.txt\"

  # Filter out already-done samples based on ACTUAL output naming:
  #   sample_<SID>.<CHR>_embeddings.pkl
  awk 'NF{print \$0}' \"${part_file}\" | while read -r SID; do
    OUTPKL=\"\${ODIR}/sample_\${SID}.\${CHR}_embeddings.pkl\"
    if [[ -f \"\${OUTPKL}\" ]]; then
      echo \"[sample] skip done: \${SID} (found: \${OUTPKL})\" >&2
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

  # clear stale todo to avoid confusion after completion
  : > \"\${FILTERED}\" || true
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
