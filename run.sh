GPUS=$1
python -m torch.distributed.launch --master_port 1345 --nproc_per_node $GPUS\
    train_dist.py --num_point 50000 --num_decoder_layers 6 \
    --size_delta 0.111111111111 --center_delta 0.04 \
    --learning_rate 0.006 --decoder_learning_rate 0.0006 --weight_decay 0.0005 \
    --dataset scannet --data_root scannet ${@:2} 

