# Run GGR_epi on SURF Research Cloud (SRC)

This repo is designed to be reproducible on SURF Research Cloud, following the infrastructure-as-code idea: the environment can be rebuilt from versioned scripts. (SRC) :contentReference[oaicite:1]{index=1}

## What you need (training-friendly)
- A SURF Research Cloud workspace with internet access (to clone this repo).
- GPU workspace recommended for `--device cuda:*` runs.
- Test data included in this repo under `data/` (small files for training/demo).

## Option A (Recommended): Docker / Docker Compose (SRC Catalog Item)

### A1. Create a Docker Compose component in SRC
In the SRC portal:
1. Go to **Catalog → Components → +**
2. Choose **Docker Compose** as Script Type
3. Fill in:
   - **URL**: this GitHub repo URL
   - **Path**: `src_catalog/docker-compose.yml`
   - **Tag**: `plain`
Follow the wizard to finish. :contentReference[oaicite:2]{index=2}

### A2. Run in a workspace
Once the component is available in your collaboration, create a workspace and run the catalog item.
Default command in compose runs a quick demo on `Chr1`.

You can also override the command after the workspace is started:
```bash
cd /workspace   # (the compose mounts repo here)
bash scripts/run_pipeline.sh all \
  --fasta data/genome/TAIR10_chr1_all.fas \
  --vcf   data/genome/test_template.vcf \
  --gff   data/genome/TAIR10_GFF3_genes_chr1.gff \
  --samples data/pheno/samples_test.txt \
  --outdir outputs_test \
  --chrs Chr1 \
  --gpus 0
