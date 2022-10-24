CUDA_VISIBLE_DEVICES=0 WANDB_RUN_ID=test python main.py --project ve_test --mode train \
--config configs/ve/resnet_default.py --workdir workdir/test/ --config.data.cache_rate=0.0 \
--config.model.attention_heads=1
