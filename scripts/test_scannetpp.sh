LOGDIR="outputs/DIV-MVS"
OUTDIRNAME="scannetpp"
SCANNET_EVAL="$HOME/scannetpp_mvs"

python eval.py \
--testpath $SCANNET_EVAL \
--dataset general_eval \
--testlist lists/scannetpp/nvs_sem_val.txt \
--loadckpt $LOGDIR/model_000014.ckpt \
--outdir $LOGDIR/$OUTDIRNAME \
--model weighted \
--ngroups 8,4,2 \
--num_view 11 \
--img_dist_thres 1.0 \
--depth_thres 0.01 \
--num_consistency 5 \
--max_w 1750 \
--max_h 740 \
--num_worker 4 \
--mask_using_gt   # prevents penalizing the reconstruction for filling holes in the GT

