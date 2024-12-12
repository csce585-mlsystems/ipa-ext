#!/bin/bash

set -euo pipefail

install_kubectl() {
    echo "Install kubectl"
    curl -LO https://dl.k8s.io/release/v1.23.2/bin/linux/amd64/kubectl
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    sudo microk8s config > $HOME/.kube/config
    # sudo ufw allow 16443
    # echo "y" | sudo ufw enable
    echo "End Install kubectl"
    rm -rf kubectl
}

install_istio() {
    echo "Installing Istio..."
    sudo microk8s enable community
    sudo microk8s enable istio
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/prometheus.yaml
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
    echo "Istio installed successfully."
    echo
}

install_seldon_core() {
    echo "Installing Seldon Core..."
    kubectl create namespace seldon-system || true
    helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts \
        --set usageMetrics.enabled=true \
        --set istio.enabled=true \
        --namespace seldon-system

    cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF

    kubectl patch svc istio-ingressgateway -n istio-system --patch '{"spec": {"ports": [{"name": "http2", "nodePort": 32000, "port": 80, "protocol": "TCP", "targetPort": 8080}]}}'
    echo "Seldon Core installed successfully."
}

configure_monitoring() {
    echo "Configuring monitoring..."
    sudo microk8s enable prometheus

    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: seldon-podmonitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/managed-by: seldon-core
  podMetricsEndpoints:
    - port: metrics
      interval: 1s
      path: /prometheus
  namespaceSelector:
    any: true
EOF

    kubectl apply -f ~/infrastructure/istio-monitoring.yaml
    kubectl patch svc prometheus-k8s -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
    kubectl patch svc grafana -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
    kubectl patch svc prometheus-k8s -n monitoring --patch '{"spec": {"type": "NodePort", "ports": [{"port": 9090, "nodePort": 30090}]}}'
    kubectl patch svc grafana -n monitoring --patch '{"spec": {"type": "NodePort", "ports": [{"port": 3000, "nodePort": 30300}]}}'
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
    kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/jaeger.yaml
    echo "Monitoring configuration complete."
}

install_docker() {
    echo "Installing Docker..."
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo groupadd docker || true
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker.service
    sudo systemctl enable containerd.service
    rm get-docker.sh
    echo "Docker installed successfully. Please log out and log back in to use Docker as a non-root user."
}

install_kubectl
install_istio
install_seldon_core
configure_monitoring
install_docker
