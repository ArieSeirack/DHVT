now=$(date +"%Y%m%d_%H%M%S")

python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 997\
	main.py \
	--model dhvt_tiny_imagenet_patch16 \
	--input-size 224 \
	--batch-size 128 \
    --lr 5e-4 \
	--warmup-epochs 10 \
	--epochs 300 \
   --num_workers 8 \
	--dist-eval \
   --data-set IMNET \
	--data-path ./data/imagenet \
	--output_dir ./output/imagenet/$now \
