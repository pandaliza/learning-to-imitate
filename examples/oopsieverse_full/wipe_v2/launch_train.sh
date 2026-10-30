#!/bin/bash
# usage: launch_train.sh <gpu> <cpu-list> <log> -- <train_v2.py args...>
# Refuses to start unless the GPU has >= 16 GB free (sanity; queue_v2.py applies need + 8 GB). nice 10, 1 BLAS thread.
GPU=$1; CPUS=$2; LOG=$3; shift 4
O=/edrive1/lizad/oopsieverse_full_study
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
if [ "$FREE" -lt 16000 ]; then echo "GPU $GPU only ${FREE} MiB free; refusing" | tee -a $LOG; exit 3; fi
export TMPDIR=$O/cache/tmp HF_HOME=$O/cache/hf XDG_CACHE_HOME=$O/cache/xdg TORCH_HOME=$O/cache/torch \
  MPLCONFIGDIR=$O/cache/mpl NUMBA_CACHE_DIR=$O/cache/numba PYNPUT_BACKEND=dummy \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU V2_LOG=$LOG
cd /scr/lizad/personal/learning-to-imitate/examples/oopsieverse_full/wipe_v2
exec taskset -c $CPUS nice -n 10 $O/env/oopsie/bin/python -u ${V2_SCRIPT:-train_v2.py} "$@" >> $LOG 2>&1
