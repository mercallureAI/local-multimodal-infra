FROM rust:1-trixie AS builder

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build --release --bin controller --bin worker

# The official ONNX Runtime 1.30.0 CPU build, pinned by digest and loaded at
# run time (ort `load-dynamic`).
ARG ORT_VERSION=1.30.0
ARG ORT_SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd
RUN set -eux; \
    curl -fsSL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
    echo "${ORT_SHA256}  /tmp/ort.tgz" | sha256sum -c -; \
    mkdir -p /tmp/ort /ort-libs; \
    tar -xzf /tmp/ort.tgz -C /tmp/ort --strip-components=1; \
    cp -av /tmp/ort/lib/libonnxruntime.so* /ort-libs/

FROM debian:trixie-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/workdir/models

COPY --from=builder /app/target/release/controller /usr/local/bin/controller
COPY --from=builder /app/target/release/worker /usr/local/bin/worker
COPY --from=builder /ort-libs/ /usr/local/lib/
COPY configs ./configs
RUN ldconfig

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV RUST_LOG=info
