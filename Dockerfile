FROM rust:1-trixie AS builder

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# The official ONNX Runtime 1.30.0 CPU build, pinned by digest and loaded at
# run time (ort `load-dynamic`). Before the sources, so a code change does not
# download it again.
ARG ORT_VERSION=1.30.0
ARG ORT_SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd
RUN set -eux; \
    curl -fsSL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
    echo "${ORT_SHA256}  /tmp/ort.tgz" | sha256sum -c -; \
    mkdir -p /tmp/ort /ort-libs; \
    tar -xzf /tmp/ort.tgz -C /tmp/ort --strip-components=1; \
    cp -av /tmp/ort/lib/libonnxruntime.so* /ort-libs/; \
    rm -rf /tmp/ort /tmp/ort.tgz

COPY . .
# The crate downloads and the target directory are BuildKit caches kept
# between builds: a code change rebuilds only what it touches.
RUN --mount=type=cache,id=lmi-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=lmi-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=lmi-target-cpu,target=/app/target,sharing=locked \
    cargo build --release --bin controller --bin worker \
    && cp target/release/controller target/release/worker /usr/local/bin/

FROM debian:trixie-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/workdir/models

COPY --from=builder /usr/local/bin/controller /usr/local/bin/controller
COPY --from=builder /usr/local/bin/worker /usr/local/bin/worker
COPY --from=builder /ort-libs/ /usr/local/lib/
COPY configs ./configs
RUN ldconfig

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV RUST_LOG=info
