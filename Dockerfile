FROM rust:1-bookworm AS builder

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build --release --bins
RUN set -eux; \
    mkdir -p /ort-libs; \
    find /app/target -name 'libonnxruntime*.so*' -exec cp -Lv {} /ort-libs/ \;; \
    find /ort-libs -name 'libonnxruntime*.so*' | grep -q .

FROM debian:bookworm-slim

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

ENV RUST_LOG=info
