const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const { createServer } = require('node:http')
const { mkdtemp, rm } = require('node:fs/promises')
const path = require('node:path')
const { test } = require('node:test')

const { Memory, version } = require('../index.js')
const { version: packageVersion } = require('../package.json')

function addRequest(requestId, userId, sessionId, content) {
  return {
    request_id: requestId,
    user_id: userId,
    session_id: sessionId,
    messages: [{ role: 'user', content }],
  }
}

test('reports the Rust package version and rejects conflicting backends', () => {
  assert.equal(version(), packageVersion)
  assert.throws(
    () => new Memory({ path: '/tmp/membrick-test', baseUrl: 'http://127.0.0.1:8080' }),
    /path and base_url cannot both be set/,
  )
})

test('local Memory supports add, batch add, search, get, list, update, delete, and refresh', async () => {
  const memory = new Memory()
  const userId = 'node-test-user'

  const added = await memory.add('single-write', userId, 'session-a', [
    { role: 'user', content: 'The quartz lighthouse inspection is scheduled for Thursday.' },
  ])
  assert.equal(added.success, true)
  assert.equal(added.request_id, 'single-write')

  const batch = await memory.addBatch([
    addRequest('batch-write-1', userId, 'session-a', 'The backup window starts at noon.'),
    addRequest('batch-write-2', userId, 'session-b', 'The release review is on Monday.'),
  ])
  assert.equal(batch.length, 2)
  assert.ok(batch.every((response) => response.success))
  await assert.rejects(memory.addBatch([]), /between 1 and 128/)
  await assert.rejects(
    memory.addBatch(Array.from({ length: 129 }, (_, i) => addRequest(`too-many-${i}`, userId, 's', 'x'))),
    /between 1 and 128/,
  )

  const listed = await memory.list(userId)
  assert.equal(listed.data.length, 3)
  assert.equal(listed.next_cursor, null)
  const first = listed.data.find((record) => record.content.includes('quartz lighthouse'))
  assert.ok(first)

  const fetched = await memory.get(first.id, userId)
  assert.equal(fetched.id, first.id)
  assert.equal(await memory.get(first.id, 'another-user'), null)
  assert.equal((await memory.list('another-user')).data.length, 0)

  const updated = await memory.update(first.id, userId, 'The quartz lighthouse inspection moved to Friday.')
  assert.equal(updated.id, first.id)
  assert.match(updated.content, /moved to Friday/)

  const hits = await memory.search('quartz lighthouse inspection Friday', userId, 5)
  assert.ok(hits.some((hit) => hit.id === first.id))
  assert.deepEqual(await memory.search('quartz lighthouse inspection Friday', 'another-user', 5), [])

  assert.equal(await memory.delete(userId, first.id), true)
  assert.equal(await memory.delete(userId, first.id), false)
  assert.equal(await memory.get(first.id, userId), null)
  await memory.refresh()
})

test('path-backed Memory persists records across Node processes', async (t) => {
  const directory = await mkdtemp(path.join(__dirname, '.membrick-node-test-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const modulePath = require.resolve('../index.js')

  const writeScript = `
    const { Memory } = require(${JSON.stringify(modulePath)});
    (async () => {
      const memory = new Memory({ path: ${JSON.stringify(directory)} });
      await memory.add('persistent-write', 'persistent-user', 'persistent-session', [
        { role: 'user', content: 'The durable copper bridge opens at sunrise.' }
      ]);
      await memory.refresh();
    })().catch((error) => { console.error(error); process.exitCode = 1; });
  `
  const write = spawnSync(process.execPath, ['-e', writeScript], { encoding: 'utf8', timeout: 60_000 })
  assert.equal(write.status, 0, write.stderr || write.error?.message)

  const readScript = `
    const { Memory } = require(${JSON.stringify(modulePath)});
    (async () => {
      const memory = new Memory({ path: ${JSON.stringify(directory)} });
      const result = await memory.list('persistent-user');
      console.log(JSON.stringify(result));
    })().catch((error) => { console.error(error); process.exitCode = 1; });
  `
  const read = spawnSync(process.execPath, ['-e', readScript], { encoding: 'utf8', timeout: 60_000 })
  assert.equal(read.status, 0, read.stderr || read.error?.message)
  const result = JSON.parse(read.stdout.trim())
  assert.equal(result.data.length, 1)
  assert.match(result.data[0].content, /copper bridge opens at sunrise/)
})

test('remote Memory maps all methods to the versioned HTTP API with bearer auth', async (t) => {
  const requests = []
  const record = {
    id: 'remote-memory-1',
    content: 'user: remote memory content',
    role: 'user',
    user_id: 'alice',
    session_id: 'session-1',
    created_at: null,
    expires_at_ms: null,
    supersedes_id: null,
    metadata: {},
  }
  const server = createServer(async (request, response) => {
    const chunks = []
    for await (const chunk of request) chunks.push(chunk)
    const rawBody = Buffer.concat(chunks).toString('utf8')
    const url = new URL(request.url, 'http://127.0.0.1')
    const body = rawBody ? JSON.parse(rawBody) : undefined
    requests.push({ method: request.method, path: url.pathname, query: url.searchParams, body, auth: request.headers.authorization })

    const send = (status, value) => {
      response.writeHead(status, value === undefined ? {} : { 'content-type': 'application/json' })
      response.end(value === undefined ? '' : JSON.stringify(value))
    }

    if (request.method === 'POST' && url.pathname === '/v1/memories') {
      send(201, { success: true, request_id: body.request_id, user_id: body.user_id, session_id: body.session_id })
    } else if (request.method === 'POST' && url.pathname === '/v1/memories/search') {
      send(200, { data: [{ ...record, score: 0.91 }] })
    } else if (request.method === 'GET' && url.pathname === '/v1/memories') {
      send(200, { data: [record], next_cursor: null })
    } else if (request.method === 'GET' && url.pathname === '/v1/memories/not-found') {
      send(404)
    } else if (request.method === 'GET' && url.pathname === '/v1/memories/remote-memory-1') {
      send(200, record)
    } else if (request.method === 'PATCH' && url.pathname === '/v1/memories/remote-memory-1') {
      send(200, { ...record, content: `user: ${body.content}` })
    } else if (request.method === 'PATCH' && url.pathname === '/v1/memories/not-found') {
      send(404)
    } else if (request.method === 'DELETE' && url.pathname === '/v1/memories/remote-memory-1') {
      send(200)
    } else if (request.method === 'DELETE' && url.pathname === '/v1/memories/not-found') {
      send(404)
    } else if (request.method === 'POST' && url.pathname === '/v1/memories/refresh') {
      send(204)
    } else {
      send(404, { error: `unexpected route ${request.method} ${url.pathname}` })
    }
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  t.after(() => new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve())))

  const address = server.address()
  const memory = new Memory({ baseUrl: `http://127.0.0.1:${address.port}/`, apiKey: 'test-secret' })

  assert.equal((await memory.add('remote-add', 'alice', 'session-1', [{ role: 'user', content: 'remote write' }])).success, true)
  const batch = await memory.addBatch([
    addRequest('remote-batch-1', 'alice', 'session-1', 'one'),
    addRequest('remote-batch-2', 'alice', 'session-1', 'two'),
  ])
  assert.equal(batch.length, 2)

  const hits = await memory.search('remote query', 'alice', 4, 'session-1', { provider: 'node' }, 'workspace')
  assert.equal(hits[0].id, 'remote-memory-1')
  assert.ok(Math.abs(hits[0].score - 0.91) < 1e-6)
  assert.equal((await memory.get('remote-memory-1', 'alice')).id, 'remote-memory-1')
  assert.equal(await memory.get('not-found', 'alice'), null)
  assert.equal((await memory.list('alice', 'session-1', 7)).data[0].id, 'remote-memory-1')
  assert.equal((await memory.update('remote-memory-1', 'alice', 'updated remote content')).content, 'user: updated remote content')
  assert.equal(await memory.update('not-found', 'alice', 'ignored'), null)
  assert.equal(await memory.delete('alice', 'remote-memory-1'), true)
  assert.equal(await memory.delete('alice', 'not-found'), false)
  await memory.refresh()

  assert.ok(requests.every((request) => request.auth === 'Bearer test-secret'))
  const searchRequest = requests.find((request) => request.path === '/v1/memories/search')
  assert.deepEqual(searchRequest.body, {
    query: 'remote query',
    options: null,
    user_id: 'alice',
    top_k: 4,
    session_id: 'session-1',
    scope: 'workspace',
    filters: { provider: 'node' },
    reference_date: null,
  })
  assert.equal(requests.filter((request) => request.method === 'POST' && request.path === '/v1/memories').length, 3)
  assert.equal(requests.at(-1).path, '/v1/memories/refresh')
})
