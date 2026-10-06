# syntax=docker/dockerfile:1

FROM rust:1.96-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY data/lexical ./data/lexical
COPY dashboard ./dashboard

RUN cargo build --release --bin server

FROM debian:bookworm-slim AS runtime

RUN groupadd --gid 10001 lintai \
    && useradd --uid 10001 --gid lintai --home-dir /data --no-create-home --shell /usr/sbin/nologin lintai \
    && mkdir -p /data/index \
    && chown -R lintai:lintai /data

COPY --from=builder /build/target/release/server /usr/local/bin/lint-ai-server

USER lintai
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080

ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/lint-ai-server"]
CMD ["--bind", "0.0.0.0:8080", "--allow-non-loopback", "--index", "/data/index"]
