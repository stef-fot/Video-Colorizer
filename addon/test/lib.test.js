'use strict'
const test = require('node:test')
const assert = require('node:assert')
const lib = require('../lib')

const SERVER = 'http://127.0.0.1:7002/'

test('only direct http(s) streams are wrapped', () => {
  const streams = [
    { url: 'https://cdn.example.com/a.mp4', name: 'A' },
    { url: 'http://192.168.1.5/b.mkv' },
    { infoHash: 'abcdef', fileIdx: 0 },
    { ytId: 'xyz' },
    { externalUrl: 'https://example.com' },
    { url: 'magnet:?xt=urn:btih:abc' },
    { url: 'file:///etc/passwd' }
  ]
  const out = lib.wrapStreams(streams, SERVER)
  assert.strictEqual(out.length, 2)
  assert.ok(out.every(s => s.name === 'Colorized (AI)'))
})

test('server url is built and encoded', () => {
  const u = lib.buildServerUrl(SERVER, 'https://x.com/a b.mp4?token=1&x=2')
  assert.strictEqual(u, 'http://127.0.0.1:7002/stream?src=https%3A%2F%2Fx.com%2Fa%20b.mp4%3Ftoken%3D1%26x%3D2')
})

test('proxy headers are forwarded to the server', () => {
  const s = { url: 'https://x.com/a.mp4', behaviorHints: { proxyHeaders: { request: { Referer: 'https://x.com' } } } }
  const out = lib.colorizedStream(s, SERVER)
  const hdr = new URL(out.url).searchParams.get('hdr')
  assert.deepStrictEqual(JSON.parse(hdr), { Referer: 'https://x.com' })
})

test('originals are kept first when requested', () => {
  const s = [{ url: 'https://x.com/a.mp4', name: 'Orig' }]
  const out = lib.wrapStreams(s, SERVER, { keepOriginals: true })
  assert.strictEqual(out.length, 2)
  assert.strictEqual(out[0].name, 'Orig')
  assert.strictEqual(out[1].name, 'Colorized (AI)')
  assert.match(out[1].title, /Orig/)
})

test('addonBase strips manifest.json', () => {
  assert.strictEqual(lib.addonBase('https://h.com/p/manifest.json'), 'https://h.com/p')
  assert.strictEqual(lib.addonBase('https://h.com/p/'), 'https://h.com/p')
})

test('library stream uses the server', () => {
  const s = lib.libraryStream({ id: 'x', name: 'Film', url: 'C:/v/f.mkv' }, SERVER)
  assert.ok(s.url.startsWith('http://127.0.0.1:7002/stream?src=C%3A%2Fv%2Ff.mkv'))
})
