#!/bin/bash

# Define the folders to copy
folders=(
    "/mnt/myshareddir/huggingface/question-answering/deepset-roberta-base-squad2"
    "/mnt/myshareddir/huggingface/question-answering/deepset-xlm-roberta-large-squad2"
    "/mnt/myshareddir/huggingface/question-answering/deepset-xlm-roberta-base-squad2"
    "/mnt/myshareddir/huggingface/question-answering/distilbert-base-cased-distilled-squad"
)

REPOS=(
    sdghafouri
)

dockerfile="Dockerfile"
IMAGE_NAME=audio-qa-final-with-model:nlpqa

# Create the models directory
mkdir models

# Copy the folders to the models directory
for folder in "${folders[@]}"; do
    cp -r "$folder" models/
done

# Generate the Dockerfile
mlserver dockerfile --include-dockerignore .
sed -i 's/seldonio/sdghafouri/g' Dockerfile
sed -i 's/1.3.0.dev4-slim/custom-2-slim/g' Dockerfile

# Add the copy commands to the Dockerfile using the contents of the models directory
# copy_commands=""
# for folder in "${folders[@]}"; do
#     target_path="/mnt/models/$(basename "$folder")"
#     copy_commands+="COPY models/$(basename "$folder") $target_path\n"
# done
# awk -v copy_commands="$copy_commands" '!done && /^COPY \\$/ {print copy_commands; done=1} 1' "$dockerfile" > Dockerfile.tmp
# mv Dockerfile.tmp "$dockerfile"

# Build and push the Docker image
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in "${REPOS[@]}"; do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done

# Delete the models directory
rm -r models
