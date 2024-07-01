export CUDA_VISIBLE_DEVICES=4,5,6,7
export gpu_num=2
export NCCL_P2P_DISABLE=1

# export CUDA_LAUNCH_BLOCKING=1
MTYPE=switch
model_path=PATH_2_MODEL/switch-base-32
dataset=ood_sa_dataset_seq2seq
scheduler_type=linear

weight_decay=1e-3
dropout_rate=1e-1
expert_dropout_rate=1e-1
aux_loss_weight=0
batch_size_training=64
val_batch_size=256
lr=1e-4
epoch=6

project_name=icml-ood-sa-${MTYPE}
exp_name=freeze-${MTYPE}-wd:${weight_decay}-dp:${dropout_rate},${expert_dropout_rate}-aux:${aux_loss_weight}-epoch:${epoch}-lr:${lr}

# freeze router
torchrun --nnodes 1 --nproc_per_node ${gpu_num} --master-port 29112 finetuning_moe.py \
--model_type ${MTYPE} \
--scheduler_type ${scheduler_type} \
--project_name ${project_name} \
--expname ${exp_name} \
--run_ood \
--run_validation \
--freeze_router \
--save_optimizer \
--run_validation_steps 10 \
--batch_size_training ${batch_size_training} \
--val_batch_size ${val_batch_size} \
--gradient_accumulation_steps 1 \
--weight_decay ${weight_decay} \
--dropout_rate ${dropout_rate} \
--expert_dropout_rate ${expert_dropout_rate} \
--lr ${lr} \
--aux_loss_weight ${aux_loss_weight} \
--router_z_loss_coef ${aux_loss_weight} \
--router_aux_loss_coef ${aux_loss_weight} \
--num_epochs ${epoch} \
--dataset ${dataset} \
--enable_fsdp \
--model_name ${model_path} \
--dist_checkpoint_root_folder PATH_2_SAVEDIR \
--dist_checkpoint_folder NAME_OF_CKPT