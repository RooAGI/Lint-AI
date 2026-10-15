# Membrick for Node.js

`membrick` is the Node.js binding for the same Rust `MemoryService` used
by the Rust and Python APIs. It has the same two backend modes as Python:

- Local: `path` selects a persistent store; without it the service is in memory.
- Remote: `base_url` connects to the existing HTTP server, with optional
  `api_key` bearer authentication.

## Install

```sh
npm install membrick
```

This installs both the Node.js `MemoryService` binding and the `membrick` CLI
on supported platforms. For example:

```sh
npm install --global membrick
membrick install codex .
```

The CLI is the Rust release executable. Platform packages contain it alongside
the N-API addon, so agent integration behavior stays in the Rust CLI.

Prebuilt native packages are selected for macOS arm64/x64, Linux arm64/x64
glibc, and Windows x64. Building from a checkout requires Rust 1.88+ and
Node.js 20.17+, 22.13+, or 23.5+ for the N-API build CLI.

The package is published from the matching `v<version>` Git tag after all
platform builds succeed. The repository needs an `NPM_TOKEN` Actions secret
with permission to publish `membrick` and its `@rooagi` platform packages.

## Local service

```js
const { Memory } = require('membrick')

const memory = new Memory({ path: './.membrick/memory' })
await memory.add('write-1', 'alice', 'session-1', [
  { role: 'user', content: 'The launch is on Friday.' },
])
const hits = await memory.search('When is the launch?', 'alice', 5)
```

## Remote service

```js
const memory = new Memory({
  baseUrl: 'http://127.0.0.1:8080',
  apiKey: process.env.MEMBRICK_API_KEY,
})
const hits = await memory.search('When is the launch?', 'alice')
```

The remote mode talks to the same `/v1/memories` endpoints as Python. Use it
when another process already owns the workspace. Local mode is intended for an
application that owns its store; do not open the same persistent workspace in
multiple writer processes.

The public methods are `add`, `addBatch`, `search`, `get`, `list`, `update`,
`delete`, and `refresh`. Method arguments follow the Python `Memory` binding;
message fields and `addBatch` request objects retain the snake_case JSON
contract used by Python and the HTTP API.

## Build from source

```sh
cd bindings/node
npm install
npm run build
npm test
```
