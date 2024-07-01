MTYPE=pythia

python inference/checkpoint_converter_fsdp_hf.py \
    -mtype $MTYPE \
    -fsdp_checkpoint_path "path/to/fsdp/checkpoint" \
    -consolidated_model_path "path/to/converted/model" \
    -HF_model_path_or_name "EleutherAI/pythia-410m"
