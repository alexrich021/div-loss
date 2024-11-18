DTU_EVAL="$HOME/dtu_eval"
LOGDIR="outputs/DIV-MVS"
OUTDIRNAME="dtu"

python eval.py \
--testpath $DTU_EVAL \
--loadckpt $LOGDIR/model_000014.ckpt \
--outdir $LOGDIR/$OUTDIRNAME \
--model weighted \
--ngroups 8,4,2 \
--depth_thres 0.001 \
--num_worker 2
