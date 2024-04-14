# Train GPT2-Small Infinity on PG19
export DATASET_NAME="pg19"
export DATASET_CONFIG_NAME="pg19"
./train_gpt2_infinity.sh gpt2

# Train GPT2-Large Infinity on PG19
export DATASET_NAME="pg19"
export DATASET_CONFIG_NAME="pg19"
./train_gpt2_infinity.sh gpt2-large

# Train GPT2-XL Infinity on PG19
export DATASET_NAME="pg19"
export DATASET_CONFIG_NAME="pg19"
./train_gpt2_infinity.sh gpt2-xl

# Train GPT2-Small Infinity on Wikipedia
export DATASET_NAME="wikimedia/wikipedia"
export DATASET_CONFIG_NAME="20231101.en"
./train_gpt2_infinity.sh gpt2

# Train GPT2-Large Infinity on Wikipedia
export DATASET_NAME="wikimedia/wikipedia"
export DATASET_CONFIG_NAME="20231101.en"
./train_gpt2_infinity.sh gpt2-large

# Train GPT2-XL Infinity on Wikipedia
export DATASET_NAME="wikimedia/wikipedia"
export DATASET_CONFIG_NAME="20231101.en"
./train_gpt2_infinity.sh gpt2-xl

# SSH into host
TRANSFORMERS_HOST_NAME="129.213.25.170"
mosh --ssh="ssh -i ~/.ssh/id_ed25519-lambda" ubuntu@$TRANSFORMERS_HOST_NAME

TRANSFORMERS_HOST_NAME="192.9.243.187"
mosh --ssh="ssh -i ~/.ssh/id_ed25519-lambda" ubuntu@$TRANSFORMERS_HOST_NAME

# Tar, download, and extract model snapshot from training host
TRANSFORMERS_HOST_NAME="129.213.25.170"
./ontology/tar_download_and_extract_model_snapshot.sh $TRANSFORMERS_HOST_NAME

# Watch tar status for file
cd $HOME/models/gpt2-large_infinity/focused/checkpoints
watch -n1 du -sh *

# Monitor GPU memory usage
nvidia-smi -f nvidia.smi -l 1

scp -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/transformers/nvidia.smi .

##
##  Training host setup
##

# Copy Lambda Labs SSH key to host
TRANSFORMERS_HOST_NAME="192.9.243.187"
scp -i ~/.ssh/id_ed25519-lambda ~/.ssh/id_ed25519-lambda "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/.ssh"


# Bootstrap host
TRANSFORMERS_HOST_NAME="129.213.25.170"
./bootstrap_host.sh $TRANSFORMERS_HOST_NAME ubuntu ~/.ssh/id_ed25519-lambda "gpt2_infinity"

TRANSFORMERS_HOST_NAME="192.9.243.187"
./bootstrap_host.sh $TRANSFORMERS_HOST_NAME ubuntu ~/.ssh/id_ed25519-lambda "gpt2_infinity"


# Dataset tests
source .env/bin/activate
python examples/pytorch/language-modeling/sequential_iterable_dataset_test.py
accelerate launch examples/pytorch/language-modeling/sequential_iterable_dataset_test.py
accelerate launch --num_processes=2 examples/pytorch/language-modeling/sequential_iterable_dataset_test.py


##
##  Dataset cache copy
##

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

# Tar and download model from host
TRANSFORMERS_HOST_NAME="129.213.25.170"
export LAST_CHECKPOINT_ON_HOST=$(
    ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME \
        "\$HOME/transformers/get_model_path_for_evaluation.sh \$HOME/models/gpt2-large_infinity/focused/checkpoints/"
)

# Tar entire training state of model
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME \
        "cd \$HOME/models/gpt2-large_infinity/focused/checkpoints && tar -czvf $LAST_CHECKPOINT_ON_HOST.tar.gz $LAST_CHECKPOINT_ON_HOST"

# Download model from host with training state for wikipedia
mkdir -p ../models/gpt2-large_infinity/focused/wikipedia/checkpoints
export LAST_CHECKPOINT_ON_HOST="checkpoint-2547000"
scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \
    ../models/gpt2-large_infinity/focused/wikipedia/checkpoints

# Download model from host with training state for pg19
export LAST_CHECKPOINT_ON_HOST="checkpoint-14332000"
scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \
    ../models/gpt2-large_infinity/focused/pg19/checkpoints

# Tar only model files
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME \
        "cd \$HOME/models/gpt2-large_infinity/focused/checkpoints && \
            tar -czvf $LAST_CHECKPOINT_ON_HOST.tar.gz \
                $LAST_CHECKPOINT_ON_HOST/*.json \
                $LAST_CHECKPOINT_ON_HOST/pytorch_model.bin"

scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \
    ../models/gpt2-large_infinity/focused/checkpoints

# Download previously tarred model for pg19
TRANSFORMERS_HOST_NAME="129.213.25.170"
LAST_CHECKPOINT_ON_HOST="checkpoint-7898000"
scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz \
    ../models/gpt2-large_infinity/focused/pg19/checkpoints

# Extract tarred model on desktop
tar -xzvf "../models/gpt2-large_infinity/focused/wikipedia/checkpoints/$LAST_CHECKPOINT_ON_HOST.tar.gz" \
    --directory "../models/gpt2-large_infinity/focused/wikipedia/checkpoints/"

cd ../models/gpt2-large_infinity/focused/checkpoints/
tar -xzvf checkpoint-14882000.tar.gz

# Extract tarred model on host
cd $HOME/models/gpt2-large_infinity/focused/checkpoints/
tar -xzvf checkpoint-7898000.tar.gz

##
##  Download final model after training completion
##

# Tar final model after training
TRANSFORMERS_HOST_NAME="129.213.25.170"
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME
cd $HOME/models/gpt2-large_infinity/focused/checkpoints

tar -czvf pg19.tar.gz \
    README.md \
    all_results.json \
    config.json \
    merges.txt \
    pytorch_model.bin \
    special_tokens_map.json \
    tokenizer.json \
    tokenizer_config.json \
    train_results.json \
    trainer_state.json \
    training_args.bin \
    vocab.json

# Download final model after training
TRANSFORMERS_HOST_NAME="129.213.25.170"
FINAL_MODEL_FILENAME="pg19"
scp -i ~/.ssh/id_ed25519-lambda \
    ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/$FINAL_MODEL_FILENAME.tar.gz \
    ../models/gpt2-large_infinity/focused/checkpoints

# Reset traininer state
ssh -i ~/.ssh/id_ed25519-lambda ubuntu@$TRANSFORMERS_HOST_NAME
cd $HOME/models/gpt2-large_infinity/focused/checkpoints/checkpoint-0
mv trainer_state.json trainer_state.json.bak
cat trainer_state.json.bak \
    | jq -r '.log_history = []' \
    | jq -r '.global_step = 0' \
    | jq -r '.epoch = 0' \
    | jq -r '.total_flos = 0' \
    | jq -r 'del(.max_steps)' \
    | tee trainer_state.json


##
##  Testing
##

# Copy remote model locally
mkdir -p "../models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000"
rsync -Prv -e "ssh -i $HOME/.ssh/id_ed25519-lambda" \
    "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000/*.json" \
    "ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000/pytorch_model.bin" \
    "../models/gpt2-large_infinity/focused/checkpoints/checkpoint-41000"


# Testing get_model_path_for_evaluation.sh
./get_model_path_for_evaluation.sh ../models/gpt2-large_infinity/focused/checkpoints/ checkpoint

scp -i ~/.ssh/id_ed25519-lambda get_model_path_for_evaluation.sh ubuntu@$TRANSFORMERS_HOST_NAME:/home/ubuntu/transformers/


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

