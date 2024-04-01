#!/bin/bash

TRANSFORMERS_HOST_NAME="$1"
MODEL_NAME="${2:-"gpt2-large_infinity"}"
SSH_IDENTITY_PATH="${SSH_IDENTITY_PATH:-"~/.ssh/id_ed25519-lambda"}"
SSH_USER="${SSH_USER:-"ubuntu"}"

REMOTE_MODEL_CHECKPOINT_DIR="${REMOTE_MODEL_CHECKPOINT_DIR:-"focused/checkpoints"}"
LOCAL_MODEL_CHECKPOINT_DIR="${LOCAL_MODEL_CHECKPOINT_DIR:-"focused/wikipedia/checkpoints"}"

REMOTE_MODEL_CHECKPOINT_PATH="/home/$SSH_USER/models/$MODEL_NAME/$REMOTE_MODEL_CHECKPOINT_DIR"
LOCAL_MODEL_CHECKPOINT_PATH="../models/$MODEL_NAME/$LOCAL_MODEL_CHECKPOINT_DIR"

LAST_CHECKPOINT_ON_HOST=$(
    ssh -i "$SSH_IDENTITY_PATH" $SSH_USER@$TRANSFORMERS_HOST_NAME \
        "\$HOME/transformers/get_model_path_for_evaluation.sh \$HOME/models/$MODEL_NAME/focused/checkpoints/"
)

echo "Last model checkpoint on training host for $MODEL_NAME: $LAST_CHECKPOINT_ON_HOST ($(date))"

echo "Taring model checkpoint on training host for $MODEL_NAME: $LAST_CHECKPOINT_ON_HOST ($(date))"
ssh -i "$SSH_IDENTITY_PATH" $SSH_USER@$TRANSFORMERS_HOST_NAME \
    "cd $REMOTE_MODEL_CHECKPOINT_PATH && tar -czvf $LAST_CHECKPOINT_ON_HOST.tar.gz $LAST_CHECKPOINT_ON_HOST"

echo "Downloading tar for model checkpoint from training host for $MODEL_NAME: $LAST_CHECKPOINT_ON_HOST ($(date))"
scp -i "$SSH_IDENTITY_PATH" \
	"$SSH_USER@$TRANSFORMERS_HOST_NAME:$REMOTE_MODEL_CHECKPOINT_PATH/$LAST_CHECKPOINT_ON_HOST.tar.gz" \
	"$LOCAL_MODEL_CHECKPOINT_PATH"

echo "Extracting tar for model checkpoint on this host for $MODEL_NAME: $LAST_CHECKPOINT_ON_HOST ($(date))"
tar -xzvf "$LOCAL_MODEL_CHECKPOINT_PATH/$LAST_CHECKPOINT_ON_HOST.tar.gz" \
    --directory "$LOCAL_MODEL_CHECKPOINT_PATH/"
