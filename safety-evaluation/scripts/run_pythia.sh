export CUDA_VISIBLE_DEVICES=
export NCCL_P2P_DISABLE=
export HF_HOME=
export WANDB_API_KEY=

torchrun --nnodes 1 --nproc_per_node 2 --master-port 29100 finetuning_dense.py \
--batch_size_training 64 --lr 5e-5 \
--gradient_accumulation_steps 1 --weight_decay 0 \
--num_epochs 1 \
--dataset alpaca_dataset \
--enable_fsdp \
--report \
--dist_checkpoint_root_folder PATH/TO/DIR \
--model_name EleutherAI/pythia-1.4b --pure_bf16 \
--dist_checkpoint_folder NAME_OF_CKPT