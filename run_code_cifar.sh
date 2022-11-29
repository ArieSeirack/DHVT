now=$(date +"%Y%m%d_%H%M%S")

python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=990\
	main.py \
	--model dhvt_tiny_cifar_patch4 \
	--input-size 32 \
	--batch-size 256 \
	--warmup-epochs 5 \
   --lr 1e-3 \
   --num_workers 8 \
	--epochs 300 \
	--dist-eval \
   --data-set CIFAR \
	--data-path ./data/cifar-100-python \
	--output_dir ./output/cifar/$now

