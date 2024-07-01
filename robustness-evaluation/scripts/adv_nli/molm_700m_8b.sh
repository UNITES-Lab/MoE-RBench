export CUDA_VISIBLE_DEVICES=4,5,6,7
export gpu_num=4
export NCCL_P2P_DISABLE=1

# export CUDA_LAUNCH_BLOCKING=1
MTYPE=molm
model_path=PATH_2_MODEL/MoLM-700M-8B
scheduler_type=cosine

weight_decay=1e-3
dropout_rate=1e-1
aux_loss_weight=0
batch_size_training=32
val_batch_size=128
lr=1e-4
epoch=3
dataset=adv_dataset_seq2class

project_name=rebuttal-mixed-adv-nli-${MTYPE}-700m8b
expert_dropout_rate=2e-1
exp_name=free-700m8b-${MTYPE}-wd:${weight_decay}-dp:${dropout_rate},${expert_dropout_rate}-aux:${aux_loss_weight}-epoch:${epoch}-lr:${lr}
echo ${project_name}+${exp_name}
# freeze router
torchrun --nnodes 1 --nproc_per_node ${gpu_num} --master-port 29142 finetuning_anli.py \
--freeze_router \
--begin_round 2 \
--model_type ${MTYPE} \
--scheduler_type ${scheduler_type} \
--project_name ${project_name} \
--expname ${exp_name} \
--run_ood \
--run_validation \
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
--dist_checkpoint_root_folder /data/data3/ood \
--dist_checkpoint_folder ${exp_name}-sharded > ./running/${project_name}+${exp_name}.log 2>&1

