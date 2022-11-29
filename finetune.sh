now=$(date +"%Y%m%d_%H%M%S")

ckpt="./pretrain/checkpoint.pth"

python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=997\
	main.py \
	--model dhvt_small_imagenet_patch16 \
	--input-size 224 \
	--sched cosine \
        --lr 5e-4 \
	--min-lr 1e-6 \
        --num_workers 8 \
	--weight-decay 1e-8 \
	--batch-size 128 \
	--warmup-epochs 2 \
	--epochs 100 \
        --no-repeated-aug \
	--dist-eval \
        --data-set clipart \
	--data-path ./data/domain/clipart \
        --finetune $ckpt \
	--output_dir ./output/finetune/$now \

