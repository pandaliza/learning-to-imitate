#!/bin/bash
#SBATCH --job-name=oops-eval
#SBATCH --output=logs/oops_eval_%A_%a.out
#SBATCH --error=logs/oops_eval_%A_%a.err
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --array=0-15
# One (task, arm) block per array index, arms INTERLEAVED (even=baseline, odd=intent) so both arms of a task
# start together and an interruption never leaves a whole arm missing. Resumable: resubmit the same array.
#   ARMS="M0 M1" sbatch slurm-scripts/oops/oops_20_eval.sbatch
#   sanity batch:  MAX_CASES=2 sbatch --array=0-1 slurm-scripts/oops/oops_20_eval.sbatch
source slurm-scripts/oops/_env.sh
read -r -a ARMV <<< "${ARMS:-M0 M1}"
TASKS=(put_the_bowl_on_the_stove put_the_wine_bottle_on_top_of_the_cabinet
       open_the_top_drawer_and_put_the_bowl_inside put_the_bowl_on_top_of_the_cabinet
       push_the_plate_to_the_front_of_the_stove put_the_cream_cheese_in_the_bowl
       put_the_bowl_on_the_plate put_the_wine_bottle_on_the_rack)
i=${SLURM_ARRAY_TASK_ID:?}
TASK=${TASKS[$((i / ${#ARMV[@]}))]}
ARM=${ARMV[$((i % ${#ARMV[@]}))]}
echo "array $i -> task=$TASK arm=$ARM"
python examples/openpi/oops/eval_spatial_shifts.py --preset "$ARM" --manifest-dir "$MANIFEST" \
    --out-dir "$EPISODES" --tasks "$TASK" --policy-seeds 0 --video-dir "$OOPS/videos_pv1" \
    ${MAX_CASES:+--max-cases $MAX_CASES}
