FROM nvidia/cuda:13.0.0-devel-ubi9 AS builder

RUN dnf install -y gcc make git python python-pip nss_wrapper gettext tar gzip unzip git dnsutils skopeo wget iputils nmap-ncat screen btop nginx jq && dnf clean all

WORKDIR /build
COPY . .

# sm_89 = L40S / RTX 4090 (Ada Lovelace, compute cap 8.9)
RUN make cuda CUDA_ARCH=sm_89

# Stage 2 – minimal runtime image
# nvidia/cuda:*-runtime-* images already include libcublas.so; no extra install needed.
FROM nvidia/cuda:13.0.0-runtime-ubi9

WORKDIR /app

COPY --from=builder /build/ds4-server .
COPY --from=builder /build/run-nvidia-tp-server.sh .

# Model and KV-cache are supplied via a PVC mounted at /data at runtime.
# Required env vars:
#   DS4_MODEL      – path to the GGUF file inside the PVC
#   DS4_KV_DIR     – KV-cache directory (default /data/ds4-kv)
ENV XDG_RUNTIME_DIR=/tmp

ENV DS4_SERVER_HOST=0.0.0.0 \
    DS4_SERVER_PORT=8000 \
    DS4_CTX=65536 \
    DS4_KV_DIR=/data/ds4-kv \
    DS4_KV_SPACE_MB=8192 \
    DS4_BATCHED_SESSIONS=16 \
    DS4_MODEL=/data/model/ds4flash.gguf

EXPOSE 8000

# Run as non-root (OpenShift drops privileges by default)
USER 1001

# Launch the server directly with 2-GPU tensor parallelism.
# --gpu-devices 0,1 maps the two nvidia.com/gpu slots allocated by the pod.
#ENTRYPOINT ["./ds4-server", \
#    "--cuda", \
#    "--cuda-tensor-parallel", \
#    "--gpu-devices", "0,1", \
#    "--gpu-vram", "auto"]

#CMD ["--model", "/data/model/ds4flash.gguf", \
#     "--ctx",   "65536", \
#     "--host",  "0.0.0.0", \
#     "--port",  "8000", \
#     "--batched-session", "16", \
#     "--kv-disk-dir",     "/data/ds4-kv", \
#     "--kv-disk-space-mb","8192"]
ENTRYPOINT ["/bin/sh", "-c", "--" , "while true; do sleep 30; done;"]
