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

# Train GPT2-Small Infinity
./train_gpt2_infinity.sh gpt2

# Train GPT2-Large Infinity
./train_gpt2_infinity.sh gpt2-large

# Train GPT2-XL Infinity
./train_gpt2_infinity.sh gpt2-xl

# Bootstrap host
./bootstrap_host.sh 129.213.25.170 ubuntu ~/.ssh/id_ed25519-lambda "gpt2-large_infinity"

# Test setup.sh on host
scp -i ~/.ssh/id_ed25519-lambda get_model_path_for_evaluation.sh ubuntu@129.213.25.170:/home/ubuntu/transformers/

# Download model from host
export LAST_CHECKPOINT_ON_HOST=$(
    ssh -i ~/.ssh/id_ed25519-lambda ubuntu@129.213.25.170 \
        "/home/ubuntu/transformers/get_model_path_for_evaluation.sh /home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/"
)

ssh -i ~/.ssh/id_ed25519-lambda ubuntu@129.213.25.170 \
        "tar -czvf /home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz /home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST"

scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@129.213.25.170:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \
    ../models/gpt2-large_infinity/focused/checkpoints

# SSH into host
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@129.213.25.170

ssh -i ~/.ssh/id_ed25519-lambda ubuntu@129.213.25.170 "mkdir -p /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"

# Upload core dataset files
du -sh $HOME/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/*

HUGGINGFACE_DATASET_PATH="pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"
scp -i ~/.ssh/id_ed25519-lambda \
    "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/dataset_info.json" \
    "ubuntu@129.213.25.170:/home/ubuntu/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/"

ls -1 "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH" \
    | grep "pg19" \
    | xargs -I{} scp -i ~/.ssh/id_ed25519-lambda \
        "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/{}" \
        "ubuntu@129.213.25.170:/home/ubuntu/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/"

# Copy cache files from remote host locally
HUGGINGFACE_DATASET_PATH="pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52"
rsync -Prv -e "ssh -i $HOME/.ssh/id_ed25519-lambda" \
    "ubuntu@129.213.25.170:/home/ubuntu/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/cache-*.arrow" \
    "$HOME/.cache/huggingface/datasets/$HUGGINGFACE_DATASET_PATH/"

# Monitor files on remote host
du -sh /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/*
ls -al /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52
ls -1 /home/ubuntu/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52


# Monitor GPU memory usage
nvidia-smi -f nvidia.smi -l 1

scp -i ~/.ssh/id_ed25519-lambda ubuntu@129.213.25.170:/home/ubuntu/transformers/nvidia.smi .

# Copy remote model locally
mkdir -p "../models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000"
rsync -Prv -e "ssh -i $HOME/.ssh/id_ed25519-lambda" \
    "ubuntu@129.213.25.170:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000/*.json" \
    "ubuntu@129.213.25.170:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000/pytorch_model.bin" \
    "../models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000"



# Upload local datasets downloads to host (DEPRECATED)
ls ~/.cache/huggingface/datasets/downloads | wc -l

tar -czvf ~/.cache/huggingface/datasets/downloads.tar.gz ~/.cache/huggingface/datasets/downloads
du -sh ~/.cache/huggingface/datasets/downloads.tar.gz
scp -i ~/.ssh/id_ed25519-lambda ~/.cache/huggingface/datasets/downloads.tar.gz ubuntu@129.213.25.170:/home/ubuntu/.cache/huggingface/datasets/

# Upload local cached dataset to host
cd ~/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/ && du -sh *

ls -alt ~/.cache/huggingface/datasets/pg19/pg19/0.1.0/27ec2bf19d4783d6380fa725bb6664a91e8016ef4dd616de4d63570ff9aeaf52/
