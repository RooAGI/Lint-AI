# Run Lint-AI with Docker

Docker Compose builds and runs the Lint-AI HTTP server, keeps its memory index
in a persistent volume, and publishes the API on localhost.

## Requirements

- Docker Engine with the Compose plugin
- A terminal in the Lint-AI repository

## Start the server

Set a private API token in your shell, then build and start the container:

```bash
export SERVER_TOKEN="$(openssl rand -hex 32)"
docker compose up --build -d
```

The first start builds the release server image. Compose stores the index in
the `lint-ai-data` volume, so memories remain available when the container is
restarted or upgraded.

The server is available at `http://127.0.0.1:8080`. The Compose port mapping
binds the host side to localhost. The container listens on its own network
interface with `--allow-non-loopback`; that option requires authentication
through `SERVER_TOKEN` or `JWT_SECRET` and cannot be combined with
`--allow-unauthenticated`.

The dashboard is included and served by default. Open
[`http://127.0.0.1:8080/dashboard`](http://127.0.0.1:8080/dashboard) to view
memory index health, search activity, and agent activity. No separate
container, port, or dashboard flag is needed. See the
[Observability guide](observability.md) for what it displays.

Check that it started:

```bash
curl http://127.0.0.1:8080/health
```

Add and search a memory:

```bash
curl -X POST http://127.0.0.1:8080/add \
  -H "X-Api-Key: $SERVER_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{
    "request_id": "docker-demo:session-1:chunk-1",
    "messages": [{"role": "user", "content": "I prefer concise answers."}],
    "user_id": "docker-demo-user",
    "session_id": "docker-demo-session"
  }'

curl -X POST http://127.0.0.1:8080/search \
  -H "X-Api-Key: $SERVER_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{
    "query": "How should I format answers?",
    "user_id": "docker-demo-user",
    "top_k": 5
  }'
```

The Compose setup uses the same routes as the [HTTP server API](server.md),
including `POST /add/batch` for adding up to 128 memories in one request.

## Stop and manage data

Stop the server and keep its saved memories:

```bash
docker compose down
```

Start it again with `docker compose up -d`. To permanently delete the stored
index as well as stop the server, remove the named volume:

```bash
docker compose down --volumes
```

The Helm chart also retains its persistent volume when the release is
uninstalled. To permanently delete the Helm-managed index, remove the claim
after uninstalling:

```bash
helm uninstall lint-ai --namespace lint-ai
kubectl delete pvc lint-ai --namespace lint-ai
```

Keep `SERVER_TOKEN` private. The supplied Compose file publishes the port only
on the local machine. If you change that port binding to expose the service on
another interface, put it behind an authenticated, encrypted access layer.

## Install on Kubernetes with Helm

The chart runs one server replica with a persistent volume and an internal
`ClusterIP` Service. The single replica is required because one Lint-AI server
process owns writes to its index. The dashboard is included at `/dashboard`.

Create a namespace and token Secret:

```bash
kubectl create namespace lint-ai
token="$(openssl rand -hex 32)"
kubectl create secret generic lint-ai-auth \
  --namespace lint-ai \
  --from-literal=SERVER_TOKEN="$token"
```

The chart defaults to `ghcr.io/rooagi/lint-ai-server:0.3.0`; the repository
does not currently publish this image automatically. Build and push the image
to a registry you control, then set `image.repository` and `image.tag` during
installation. For example:

```bash
docker build -t registry.example.com/lint-ai-server:0.3.0 .
docker push registry.example.com/lint-ai-server:0.3.0
helm upgrade --install lint-ai ./charts/lint-ai \
  --namespace lint-ai \
  --set image.repository=registry.example.com/lint-ai-server \
  --set image.tag=0.3.0
```

The default chart provisions a 10 GiB persistent volume claim using the
cluster's default StorageClass; set `persistence.size` or
`persistence.storageClassName` to change it. For clusters without a default
StorageClass, set the storage class explicitly.

Forward the internal Service to your local machine to try the API and
dashboard:

```bash
kubectl port-forward --namespace lint-ai service/lint-ai 8080:8080
```

Open [`http://127.0.0.1:8080/dashboard`](http://127.0.0.1:8080/dashboard).
In-cluster agents can call `http://lint-ai.lint-ai.svc.cluster.local:8080` and
send the token in the `X-Api-Key` header. The chart does not create a public
Ingress. Add TLS and authenticated access at your cluster's gateway if
outside-cluster clients need to connect.

The chart source is versioned in `charts/lint-ai`; it can be installed on any
Kubernetes cluster, including AWS EKS. Amazon ECR supports hosting images and
Helm charts as OCI artifacts; see the
[AWS ECR chart instructions](https://docs.aws.amazon.com/AmazonECR/latest/userguide/push-oci-artifact.html)
to publish your image and chart in your AWS account. For a private image
registry, configure `imagePullSecrets` in the chart values.
