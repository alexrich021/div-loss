DATASET_DIR="$HOME/dtu/"   # path to dataset folder
LOGDIR="outputs/DIV-MVS-2"
MASTER_ADDR="localhost"
MASTER_PORT=1234
NNODES=1
NGPUS=4
NODE_RANK=0

if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi

python -m torch.distributed.launch \
--nnodes=$NNODES \
--nproc_per_node=$NGPUS \
--node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
train.py \
  --logdir $LOGDIR \
  --trainpath $DATASET_DIR \
  --testpath $DATASET_DIR \
  --ngroups 8,4,2 \
  --batch_size 2 \
  --lr 0.0005 | tee -a $LOGDIR/log.txt
