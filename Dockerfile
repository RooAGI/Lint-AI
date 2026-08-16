# syntax=docker/dockerfile:1

FROM rust:1.96-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY data/lexical ./data/lexical

RUN cargo build --release --bin server

FROM debian:bookworm-slim AS runtime

RUN groupadd --system lintai \
    && useradd --system --gid lintai --home-dir /data --no-create-home lintai \
    && mkdir -p /data/index \
    && chown -R lintai:lintai /data

COPY --from=builder /build/target/release/server /usr/local/bin/lint-ai-server

USER lintai
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080

ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/lint-ai-server"]
CMD ["--bind", "0.0.0.0:8080", "--index", "/data/index"]
