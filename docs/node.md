# Node.js binding

The `membrick` package exposes the Rust `MemoryService` to Node.js. It
follows the same local and remote backend model as the Python `Memory` binding.

Install it from npm:

```bash
npm install membrick
```

## Local memory service

Use `path` to persist a local memory store, or omit it for in-memory operation:

```js
const { Memory } = require('membrick')

const memory = new Memory({ path: './.membrick/memory' })
await memory.add('write-1', 'alice', 'session-1', [
  { role: 'user', content: 'The launch is on Friday.' },
])

const results = await memory.search('When is the launch?', 'alice', 5)
```

## Remote memory service

When the workspace already has a resident server owner, connect to that owner:

```js
const memory = new Memory({
  baseUrl: 'http://127.0.0.1:8080',
  apiKey: process.env.MEMBRICK_API_KEY,
})
const results = await memory.search('When is the launch?', 'alice')
```

The remote backend uses the versioned `/v1/memories` HTTP API. Both local and
remote modes provide `add`, `addBatch`, `search`, `get`, `list`, `update`,
`delete`, and `refresh`. Method arguments follow the Python `Memory` binding.
Message fields and `addBatch` request objects retain the snake_case JSON
contract used by Python and the HTTP API. Methods return promises so Rust work
and HTTP requests do not block the JavaScript event loop.

Local mode owns its `MemoryService`. Do not open the same persistent path from
multiple writer processes; connect those processes to the existing server with
`baseUrl` instead.

See [`bindings/node/README.md`](https://github.com/RooAGI/Membrick/tree/main/bindings/node)
for source builds and package details.
