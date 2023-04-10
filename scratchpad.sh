# Train GPT2-Small Infinity
./train_gpt2_infinity.sh gpt2

# Train GPT2-Large Infinity
./train_gpt2_infinity.sh gpt2-large

# Train GPT2-XL Infinity
./train_gpt2_infinity.sh gpt2-xl


# Bootstrap host
TRANSFORMERS_HOST_NAME="129.213.25.170"
./bootstrap_host.sh $TRANSFORMERS_HOST_NAME ubuntu ~/.ssh/id_ed25519-lambda "gpt2_infinity"


# SSH into host
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME


# Upload core dataset files
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME \
    "mkdir -p \$HOME/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"

HUGGINGFACE_DATASET_PATH="pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"
du -sh $HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/*

HUGGINGFACE_DATASET_PATH="pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"
scp -i ~/.ssh/id_ed25519-lambda \
    "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/dataset_info.json" \
    "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/"

HUGGINGFACE_DATASET_PATH="pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"
ls -1 "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH" \
    | grep "pg19" \
    | xargs -I{} scp -i ~/.ssh/id_ed25519-lambda \
        "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/{}" \
        "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/"


# Copy cache files from remote host locally
HUGGINGFACE_DATASET_PATH="pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"
rsync -Prv -e "ssh -i $HOME/.ssh/id_ed25519-lambda" \
    "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/cache-*.arrow" \
    "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/"


# Monitor files on remote host
du -sh /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/*
ls -al /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52
ls -1 /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52


# Monitor GPU memory usage
nvidia-smi -f nvidia.smi -l 1

scp -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/transformers/nvidia.smi .


# Copy Lambda Labs SSH key to host
scp -i ~/.ssh/id_ed25519-lambda ~/.ssh/id_ed25519-lambda "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/.ssh"


# Testing get_model_path_for_evaluation.sh
./get_model_path_for_evaluation.sh ../models/gpt2-large_infinity/focused/checkpoints/ checkpoint

scp -i ~/.ssh/id_ed25519-lambda get_model_path_for_evaluation.sh ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/transformers/


# Download tarred model from host
TRANSFORMERS_HOST_NAME="129.213.25.170"
export LAST_CHECKPOINT_ON_HOST=$(
    ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME \
        "\$HOME/transformers/get_model_path_for_evaluation.sh \$HOME/models/gpt2-large_infinity/focused/checkpoints/"
)

ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME \
        "tar -czvf \$HOME/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \$HOME/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST"

scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \
    ../models/gpt2-large_infinity/focused/checkpoints

# Extract tarred model
tar -xzvf ../models/gpt2-large_infinity/focused/checkpoints/checkpoint-599000.tar.gz


# Watch tar status for file
cd $HOME/models/gpt2-large_infinity/focused/checkpoints
watch -n1 du -sh *


# Copy remote model locally
mkdir -p "../models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000"
rsync -Prv -e "ssh -i $HOME/.ssh/id_ed25519-lambda" \
    "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000/*.json" \
    "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000/pytorch_model.bin" \
    "../models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000"




# Upload local datasets downloads to host [DEPRECATED]
ls ~/.cache/huggingface/datasets/downloads | wc -l

tar -czvf ~/.cache/huggingface/datasets/downloads.tar.gz ~/.cache/huggingface/datasets/downloads
du -sh ~/.cache/huggingface/datasets/downloads.tar.gz
scp -i ~/.ssh/id_ed25519-lambda ~/.cache/huggingface/datasets/downloads.tar.gz ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/.cache/huggingface/datasets/

# Upload local cached dataset to host
cd ~/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/ && du -sh *

ls -alt ~/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/


# raw_datasets["train"]
# Dataset({
#     features: ['text'],
#     num_rows: 1801350
# })
# tokenized_datasets["train"]
# Dataset({
#     features: ['input_ids', 'attention_mask'],
#     num_rows: 1801350
# })

cat input.result.json | jq -r '.input_ids | length'
cat input.result.json | jq -r '.input_ids[] | length'
cat input.result.json | jq -r '.example_idx[]'

