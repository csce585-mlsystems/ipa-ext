download_data(){
    # download experiment logs
    gsutil cp -r gs://ipa-results-1/results.zip ~/ipa/data
    unzip ~/ipa/data/results.zip
    mv results ~/ipa/data
    rm ~/ipa/data/results.zip

    # download ml models
    mc mb --ignore-existing minio/huggingface
    mc mb --ignore-existing minio/torchhub
    mkdir ~/temp-model-dir
    gsutil cp -r 'gs://ipa-models/myshareddir/torchhub' ~/temp-model-dir
    gsutil cp -r 'gs://ipa-models/myshareddir/huggingface' ~/temp-model-dir
    mc cp -r ~/temp-model-dir/huggingface/* minio/huggingface
    mc cp -r ~/temp-model-dir/torchhub/* minio/torchhub
    rm -r ~/temp-model-dir

    # download lstm trained model
    gsutil cp -r gs://ipa-models/lstm ~/ipa/data
}

download_data
