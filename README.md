# GGR_epi
`GGR_epi` is a genome-wide embedding pipeline based on **PlantCAD2**, designed for
plant genomic regions including **genes** and **intergenic regions**, with support
for **per-sample mutant embeddings** using population-scale VCFs.

The pipeline is optimized for **HPC / GPU environments** and supports
single-GPU and multi-GPU execution.

---

## Features
- Gene and intergenic region embeddings from reference genome
- SNP-aware per-sample mutant embeddings
- Chromosome-wise execution (Chr1–Chr5, etc.)
- Automatic resume / skip for completed samples
- Unified shell-based pipeline management
- Safe execution order: **reference → sample (enforced)**

---

## Main Script
All computations are performed via:

`PlantCAD2_Reference_gene_Intergenic_embedding_Sample_Windows_single_reference_Chr.py`


The script supports:
- `--mode reference` : generate reference embeddings
- `--mode sample`    : generate mutant embeddings per sample
- `--chrom ChrX`     : chromosome-wise execution

---

## Environment Setup
### Conda (recommended for HPC)
```bash
conda env create -f environment.yml
conda activate PlantCAD_offical
```

### Pip (optional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This pipeline REQUIRES an NVIDIA GPU with CUDA.
CPU-only environments are NOT supported.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Input Files
You need the following inputs:

- Reference genome (FASTA)
- Population variants (VCF, bgzip + tabix indexed)
- Gene annotation (GFF3)

Example (Arabidopsis, HPC paths):
```text
FASTA: /lustre/BIF/nobackup/zhang479/genomes/arabidopsis/TAIR10_chr_all.fas
VCF:   /lustre/BIF/nobackup/zhang479/genomes/arabidopsis/arb_ld.vcf.gz
GFF:   /lustre/BIF/nobackup/zhang479/genomes/arabidopsis/TAIR10_GFF3_genes.gff
```

## Output Structure
All outputs are written under `--outdir`.

Typical outputs:
- ref_gene_embeddings.pkl
- intergenic_embeddings.pkl
- sample_<ID>_<ID>_embeddings.pkl


Optional / auto-created directories:
- logs/
- checkpoints/
- embeddings/
- sequences/

## Resume Behavior
If `sample_<ID>_<ID>_embeddings.pkl` already exists, that sample will be skipped.
Safe for interrupted or incremental runs

## Important Dependency Rule
Sample mutant embeddings MUST be generated after reference embeddings exist.

Required reference files:
- ref_gene_embeddings.pkl
- intergenic_embeddings.pkl

To avoid mistakes, always use the provided pipeline script.

## Running the Pipeline (Recommended)
All runs are managed by:
`run_pipeline.sh`

Activate environment
```bash
conda activate PlantCAD_offical
```

## Reference only (single GPU)
```bash
bash run_pipeline.sh reference \
  --fasta /path/to/genome.fas \
  --vcf /path/to/pop.vcf.gz \
  --gff /path/to/genes.gff3 \
  --outdir arb_ld_vcf_out \
  --chrs Chr1,Chr2,Chr3,Chr4,Chr5 \
  --gpus 0
```

## Sample embeddings (single GPU)
Prepare a sample list file:
- samples.txt   # one sample ID per line

```bash
bash run_pipeline.sh sample \
  --fasta /path/to/genome.fas \
  --vcf /path/to/pop.vcf.gz \
  --gff /path/to/genes.gff3 \
  --outdir arb_ld_vcf_out \
  --chrs Chr1,Chr2,Chr3,Chr4,Chr5 \
  --gpus 0 \
  --samples samples.txt
```

## Sample embeddings (multi-GPU, auto-split)
Example with 2 GPUs (0 and 1):
```bash
bash run_pipeline.sh sample \
  --fasta /path/to/genome.fas \
  --vcf /path/to/pop.vcf.gz \
  --gff /path/to/genes.gff3 \
  --outdir arb_ld_vcf_out \
  --chrs Chr1,Chr2,Chr3,Chr4,Chr5 \
  --gpus 0,1 \
  --samples samples.txt
```
Samples are automatically split across GPUs.

## All-in-one (reference → wait → sample)  Recommended
```bash
bash run_pipeline.sh all \
  --fasta /path/to/genome.fas \
  --vcf /path/to/pop.vcf.gz \
  --gff /path/to/genes.gff3 \
  --outdir arb_ld_vcf_out \
  --chrs Chr1,Chr2,Chr3,Chr4,Chr5 \
  --gpus 0,1 \
  --samples samples.txt \
  - `--per_chr_outdir 1`
  - `--ref_bg 1`
```

This mode:
- runs reference embeddings first
- waits until reference outputs are ready
- then starts sample jobs automatically

## Recommended Flags
--per_chr_outdir 1
→ write outputs to OUTDIR/ChrX/, prevents overwrite and is safer for team use

--ref_bg 1
→ run reference in background (nohup), pipeline will still wait correctly

## Notes for HPC Users
- The pipeline is designed for long-running jobs on HPC clusters
- Logs are written to `logs/`
- Compatible with both `nohup` and scheduler-based workflows
- Easy to adapt to `sbatch` / Slurm if needed

## Citation
This project uses PlantCAD2 via HuggingFace Transformers.
Please cite PlantCAD2 and Transformers accordingly if you use this pipeline in your work.


## Directory Structure
A typical `GGR_epi` repository layout:

```text
GGR_epi/
├── README.md
├── run_pipeline.sh
├── environment.yml
├── requirements.txt
├── PlantCAD2_Reference_gene_Intergenic_embedding_Sample_Windows_single_reference_Chr.py
│
├── arb_ld_vcf_out/                 # output directory (--outdir)
│   ├── ref_gene_embeddings.pkl     # reference gene embeddings
│   ├── intergenic_embeddings.pkl   # reference intergenic embeddings
│   ├── sample_5829_5829_embeddings.pkl
│   ├── sample_6189_6189_embeddings.pkl
│   ├── sample_6252_6252_embeddings.pkl
│   ├── sample_<ID>_<ID>_embeddings.pkl
│   │
│   ├── checkpoints/                # model checkpoints (if enabled)
│   ├── embeddings/                 # intermediate embedding files
│   ├── sequences/                  # cached sequences
│   └── logs/                       # runtime logs
│
├── logs/                            # pipeline-level logs (nohup, GPU workers)
│   ├── reference.gpu0.nohup.log
│   ├── sample.gpu0.nohup.log
│   ├── sample.gpu1.nohup.log
│   └── samples_split.part_*.txt
│
└── samples.txt                     # list of sample IDs (one per line)
```

### Optional: per-chromosome output layout
When running with --per_chr_outdir 1, outputs are written per chromosome to avoid overwrite risk:

```text
arb_ld_vcf_out/
├── Chr1/
│   ├── ref_gene_embeddings.pkl
│   ├── intergenic_embeddings.pkl
│   ├── sample_5829_5829_embeddings.pkl
│   └── ...
├── Chr2/
├── Chr3/
├── Chr4/
└── Chr5/
```
## Quick Demo (Chr1, tiny sample set)

```bash
bash run_pipeline.sh all \
  --fasta data/genome/TAIR10_chr1_all.fas \
  --vcf   data/genome/test_template.vcf \
  --gff   data/genome/TAIR10_GFF3_genes_chr1.gff \
  --outdir outputs_demo \
  --chrs Chr1 \
  --gpus 0 \
  --samples data/pheno/samples_test.txt \
  --per_chr_outdir 1
