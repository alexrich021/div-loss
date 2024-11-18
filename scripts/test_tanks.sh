LOGDIR="outputs/DIV-MVS"
TANKS_DIR="$HOME/tanksandtemples/"
OUTDIRNAME="tanks"
PLYDIRNAME="tanks_ply"

SPLIT="intermediate"
python eval_tanks.py \
  --model weighted \
  --ngroups 8,4,2 \
  --testpath $TANKS_DIR \
  --split $SPLIT \
  --loadckpt $LOGDIR/model_000014.ckpt \
  --outdir $LOGDIR/$OUTDIRNAME \
  --plydir $LOGDIR/$PLYDIRNAME \
  --ndepths "64,32,8" \
  --depth_inter_r "3,2,1" \
  --num_view 11 \
  --filter normal


SPLIT="advanced"
python eval_tanks.py \
  --model weighted \
  --ngroups 8,4,2 \
  --testpath $TANKS_DIR \
  --split $SPLIT \
  --loadckpt $LOGDIR/model_000014.ckpt \
  --outdir $LOGDIR/$OUTDIRNAME \
  --plydir $LOGDIR/$PLYDIRNAME \
  --ndepths "64,32,8" \
  --depth_inter_r "3,2,1" \
  --num_view 11 \
  --filter dypcd \
  --num_worker 1

