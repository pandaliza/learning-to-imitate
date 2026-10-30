#!/bin/bash
# usage: launch_all60.sh <seed> <gpu>      (resumable: rerun the same command; finished runs print ALREADY_DONE)
# Torch-only training for the B1K all-60 matched-condition check (protocol addendum 2026-10-30). Refuses to start
# unless the GPU has >= 16 GB free. nice 10, 1 BLAS thread, 4 loader workers. Status: runs/b1k/all60/status/<run>.status
SEED=$1; GPU=$2
O=/edrive1/lizad/oopsieverse_full_study
RUN=amix_add_firewood_all60_s$SEED
LOG=$O/runs/b1k/all60/logs/$RUN.log
ST=$O/runs/b1k/all60/status/$RUN.status
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
if [ "$FREE" -lt 16000 ]; then echo "$(date -Is) REFUSED gpu=$GPU free=${FREE}MiB" >> $ST; exit 3; fi
export TMPDIR=$O/cache/tmp HF_HOME=$O/cache/hf XDG_CACHE_HOME=$O/cache/xdg TORCH_HOME=$O/cache/torch \
  MPLCONFIGDIR=$O/cache/mpl NUMBA_CACHE_DIR=$O/cache/numba PYNPUT_BACKEND=dummy \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU MATRIX_LOG=$LOG
mkdir -p $TMPDIR $HF_HOME $XDG_CACHE_HOME $TORCH_HOME $MPLCONFIGDIR $NUMBA_CACHE_DIR
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/b1k
echo "$(date -Is) START seed=$SEED gpu=$GPU run=$RUN" >> $ST
nice -n 10 $O/env/oopsie/bin/python -u train_b1k_all60.py --arm amix --task add_firewood --seed $SEED \
  --run-id $RUN --runs-root $O/runs/b1k/all60/train --workers 4 --act-ckpt --save-every 1000 >> $LOG 2>&1
RC=$?
echo "$(date -Is) EXIT rc=$RC" >> $ST
exit $RC
