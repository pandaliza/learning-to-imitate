#!/bin/bash
#SBATCH --job-name=oops-manifest
#SBATCH --output=logs/oops_manifest_%j.out
#SBATCH --error=logs/oops_manifest_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
# Policy-free physical-case manifest + validity (CPU only). Build it ONCE, before any rollout; never rebuild
# after outcomes are seen (rebuild only with a bumped VALIDITY_VERSION/CONFIG_VERSION and a written reason).
source slurm-scripts/oops/_env.sh
test -e "$MANIFEST/cases.jsonl" && { echo "manifest exists at $MANIFEST -- refusing to overwrite"; exit 1; }
python examples/openpi/oops/build_case_manifest.py --out-dir "$MANIFEST"
