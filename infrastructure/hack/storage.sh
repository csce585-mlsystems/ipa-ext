#!/bin/bash

# Function to retrieve the public IP address
get_public_ip() {
    # Using dig to get the public IP address
    dig +short myip.opendns.com @resolver1.opendns.com
}

# Function to retrieve the private IP address
get_private_ip() {
    # Grabbing the first non-loopback IP address
    hostname -I | awk '{print $1}'
}

# Function to set up storage
setup_storage() {
    local PUBLIC_IP="$1"
    local PRIVATE_IP="$2"

    echo "Setting up storage: Installing NFS..."
    sudo apt-get update
    sudo apt-get install -y nfs-kernel-server

    echo "Creating shared directory..."
    sudo mkdir -p /mnt/myshareddir
    sudo chown nobody:nogroup /mnt/myshareddir
    sudo chmod 777 /mnt/myshareddir

    echo "Configuring NFS exports..."
    echo "/mnt/myshareddir ${PRIVATE_IP}/24(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
    sudo exportfs -a
    sudo systemctl restart nfs-kernel-server

    echo "Applying PersistentVolume configuration..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
  namespace: default
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 200Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: ${PRIVATE_IP}
    path: "/mnt/myshareddir"
EOF

    echo "Creating namespace for MinIO..."
    kubectl create namespace minio-system || true

    echo "Applying PersistentVolumeClaim configuration..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-nfs
  namespace: minio-system
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
EOF

    local MINIOUSER="minioadmin"
    local MINIOPASSWORD="minioadmin"

    echo "Adding MinIO Helm repository..."
    helm repo add minio https://charts.min.io/
    helm repo update

    echo "Deploying MinIO using Helm..."
    helm upgrade --install minio minio/minio \
      --namespace minio-system \
      --set rootUser=${MINIOUSER} \
      --set rootPassword=${MINIOPASSWORD} \
      --set mode=standalone \
      --set persistence.enabled=true \
      --set persistence.existingClaim=pvc-nfs \
      --set persistence.storageClass=- \
      --set replicas=1

    echo "Patching MinIO service to use LoadBalancer..."
    kubectl patch svc minio -n minio-system --type='json' -p '[{"op":"replace","path":"/spec/type","value":"LoadBalancer"}]'
    kubectl patch svc minio -n minio-system --patch '{"spec": {"type": "LoadBalancer", "ports": [{"port": 9000, "nodePort": 31900}]}}'

    echo "Retrieving MinIO access credentials..."
    local ACCESS_KEY
    local SECRET_KEY
    ACCESS_KEY=$(kubectl get secret minio -n minio-system -o jsonpath="{.data.rootUser}" | base64 --decode)
    SECRET_KEY=$(kubectl get secret minio -n minio-system -o jsonpath="{.data.rootPassword}" | base64 --decode)

    echo "Downloading and configuring MinIO client..."
    wget https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x mc
    sudo mv mc /usr/local/bin/

    echo "Setting up MinIO client alias..."
    mc alias set minio http://localhost:31900 "${ACCESS_KEY}" "${SECRET_KEY}" --api s3v4
    mc ls minio

    echo "Creating RClone secret for Seldon..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
  namespace: default
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: Minio
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: ${MINIOUSER}
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: ${MINIOPASSWORD}
  RCLONE_CONFIG_S3_ENDPOINT: http://${PUBLIC_IP}:31900
EOF

    echo "Cleaning up..."
    rm -f mc

    echo "Storage setup completed successfully."
}

# Main script execution
echo "Running script..."

# Retrieve public and private IP addresses
PUBLIC_IP=$(get_public_ip)
PRIVATE_IP=$(get_private_ip)

# Ensure both IP addresses were retrieved successfully
if [[ -z "${PUBLIC_IP}" || -z "${PRIVATE_IP}" ]]; then
    echo "Error: Unable to retrieve IP addresses."
    exit 1
fi

# Call the setup_storage function with the retrieved IP addresses
setup_storage "${PUBLIC_IP}" "${PRIVATE_IP}"

echo "Script execution complete."