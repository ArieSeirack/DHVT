now=$(date +"%Y%m%d_%H%M%S")

python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=9977 \
	main.py \
	--model dhvt_tiny_domain_patch16 \
	--input-size 224 \
	--batch-size 128\
	--warmup-epochs 20 \
   --lr 1e-3 \
   --num_workers 8 \
	--epochs 300 \
	--dist-eval \
   --data-set quickdraw \
	--data-path ./data/domain/quickdraw \
	--output_dir ./output/domain/$now \

