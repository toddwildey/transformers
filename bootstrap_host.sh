#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

REMOTE_HOST="$1"
REMOTE_USER="$2"
REMOTE_IDENTITY_FILE="$3"
REMOTE_BRANCH_NAME="$4"

REMOTE_HOME_PATH="/home/$REMOTE_USER"

ssh-keygen -F "$REMOTE_HOST" || ssh-keyscan "$REMOTE_HOST" >> ~/.ssh/known_hosts

find "$SCRIPT_DIR/bootstrap_host/" -mindepth 1 -maxdepth 1 \
    | xargs -I{} scp -r -i "$REMOTE_IDENTITY_FILE" "{}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_HOME_PATH"
scp -i "$REMOTE_IDENTITY_FILE" ~/.ssh/id_ed25519-github-remote "$REMOTE_USER@$REMOTE_HOST:$REMOTE_HOME_PATH/.ssh"

ssh -i "$REMOTE_IDENTITY_FILE" "$REMOTE_USER@$REMOTE_HOST" "ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts"

ssh -i "$REMOTE_IDENTITY_FILE" "$REMOTE_USER@$REMOTE_HOST" "sudo apt install mosh"
ssh -i "$REMOTE_IDENTITY_FILE" "$REMOTE_USER@$REMOTE_HOST" "cd -- $REMOTE_HOME_PATH && git clone git@github.com:toddwildey/transformers.git"
ssh -i "$REMOTE_IDENTITY_FILE" "$REMOTE_USER@$REMOTE_HOST" "cd -- $REMOTE_HOME_PATH/transformers && git checkout $REMOTE_BRANCH_NAME"
ssh -i "$REMOTE_IDENTITY_FILE" "$REMOTE_USER@$REMOTE_HOST" "cd -- $REMOTE_HOME_PATH/transformers && ./setup.python.sh"
ssh -i "$REMOTE_IDENTITY_FILE" "$REMOTE_USER@$REMOTE_HOST" "cd -- $REMOTE_HOME_PATH/transformers && ./setup.sh"

echo "ssh -i $REMOTE_IDENTITY_FILE $REMOTE_USER@$REMOTE_HOST"
