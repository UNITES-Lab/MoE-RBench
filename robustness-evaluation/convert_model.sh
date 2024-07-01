MTYPE=molm

python checkpoint_converter_fsdp_hf.py \
    -mtype $MTYPE \
    -fsdp_checkpoint_path "PATH/to/SHARDED_CKPT" \
    -consolidated_model_path "PATH/to/SAVE/COMPLETE_CKPT" \
    -HF_model_path_or_name ""Huggingface path(e.g. ibm/MoLM-350M-4B)"

