# syntax=docker/dockerfile:1

FROM rust:1.96-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY data/lexical ./data/lexical
COPY dashboard ./dashboard

RUN cargo build --release --bin server

FROM debian:bookworm-slim AS runtime

RUN groupadd --gid 10001 membrick \
    && useradd --uid 10001 --gid membrick --home-dir /data --no-create-home --shell /usr/sbin/nologin membrick \
    && mkdir -p /data/index \
    && chown -R membrick:membrick /data

COPY --from=builder /build/target/release/server /usr/local/bin/membrick-server

USER membrick
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/membrick-server"]
CMD ["--bind", "127.0.0.1:8080", "--index", "/data/index"]
