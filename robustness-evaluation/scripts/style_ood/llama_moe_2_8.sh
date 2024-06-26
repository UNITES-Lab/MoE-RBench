export CUDA_VISIBLE_DEVICES=4,5,6,7
export gpu_num=4
export NCCL_P2P_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
MTYPE=llama_moe
model_path=PATH_2_MODEL/LLaMA-MoE-v1-3_5B-2_8
dataset=ood_style_dataset_seq2class
weight_decay=1e-3
dropout_rate=0
expert_dropout_rate=0
aux_loss_weight=1e-3
batch_size_training=16
val_batch_size=64


lr=5e-5
epoch=2
scheduler_type=step
task_name=rebuttal-style-ood-2-8
project_name=${task_name}-${MTYPE}
# free router
aux_loss_weight=1e-3
exp_name=bsize-64-free-${MTYPE}-wd:${weight_decay}-dp:${dropout_rate},${expert_dropout_rate}-aux:${aux_loss_weight}-epoch:${epoch}-lr:${lr}
echo ${project_name}+${exp_name}
torchrun --nnodes 1 --nproc_per_node ${gpu_num} --master-port 29182 finetuning_ood_style.py \
--task_name ${task_name} \
--model_type ${MTYPE} \
--scheduler_type ${scheduler_type} \
--project_name ${project_name} \
--expname ${exp_name} \
--run_ood \
--run_validation \
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
--model_name ${model_path} --pure_bf16 \
--dist_checkpoint_root_folder PATH_2_SAVEDIR \
--dist_checkpoint_folder NAME_OF_CKPT
