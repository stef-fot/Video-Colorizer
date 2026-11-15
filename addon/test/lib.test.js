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

test('burn mode asks the server to draw the first language', () => {
  const u = new URL(lib.colorizedStream({ url: 'https://x.com/a.mp4' }, SERVER, { mode: 'burn', langs: ['el', 'en'] }).url)
  assert.strictEqual(u.searchParams.get('subs'), 'burn')
  assert.strictEqual(u.searchParams.get('lang'), 'el')
})

test('soft mode adds one Stremio subtitle per language (Greek is ell)', () => {
  const s = lib.colorizedStream({ url: 'https://x.com/a.mp4' }, SERVER, { mode: 'soft', langs: ['el', 'en'], source: 'auto' })
  assert.deepStrictEqual(s.subtitles.map(x => x.lang), ['ell', 'eng'])
  const u = new URL(s.subtitles[0].url)
  assert.strictEqual(u.pathname, '/subs.vtt')
  assert.strictEqual(u.searchParams.get('src'), 'https://x.com/a.mp4')
  assert.strictEqual(u.searchParams.get('lang'), 'el')
  assert.ok(!new URL(s.url).searchParams.has('subs'))
})

test('no subtitle options leaves the stream unchanged', () => {
  const s = lib.colorizedStream({ url: 'https://x.com/a.mp4' }, SERVER, { mode: 'off', langs: ['el'] })
  assert.strictEqual(s.subtitles, undefined)
  assert.ok(!s.url.includes('subs='))
})

test('default subtitle languages: 20 hot languages plus Greek', () => {
  assert.strictEqual(lib.DEFAULT_LANGS.length, 21)
  assert.ok(lib.DEFAULT_LANGS.includes('el'))
  const s = lib.colorizedStream({ url: 'https://x.com/a.mp4' }, SERVER, { mode: 'soft', langs: lib.DEFAULT_LANGS })
  assert.strictEqual(s.subtitles.length, 21)
  assert.ok(s.subtitles.every(x => /^[a-z]{3}$/.test(x.lang)), 'every language has a 3 letter code')
  assert.strictEqual(new Set(s.subtitles.map(x => x.id)).size, 21)
})

test('the original stream is offered with the same subtitles, untouched URL', () => {
  const sub = { mode: 'soft', langs: ['el', 'en'] }
  const out = lib.wrapStreams([{ url: 'https://x.com/a.mp4', name: 'Orig' }], SERVER, { keepOriginals: false, subtitles: sub, withOriginals: true })
  assert.strictEqual(out.length, 2)
  assert.strictEqual(out[1].name, 'AI subtitles')
  assert.strictEqual(out[1].url, 'https://x.com/a.mp4')
  assert.deepStrictEqual(out[1].subtitles.map(x => x.lang), ['ell', 'eng'])
  assert.strictEqual(lib.subtitledOriginal({ url: 'https://x.com/a.mp4' }, SERVER, { mode: 'off', langs: ['el'] }), null)
})

// ---- resolution ranking and restoration

test('resolution tags are read from name, title, filename and url', () => {
  assert.strictEqual(lib.detectResolution({ url: 'https://x.com/a.mp4', name: 'Source 1080p' }), 1080)
  assert.strictEqual(lib.detectResolution({ url: 'https://x.com/My.Film.2160p.mkv' }), 2160)
  assert.strictEqual(lib.detectResolution({ url: 'https://x.com/a.mp4', title: 'UHD HDR' }), 2160)
  assert.strictEqual(lib.detectResolution({ url: 'https://x.com/a.mp4', behaviorHints: { filename: 'film.720p.mkv' } }), 720)
  assert.strictEqual(lib.detectResolution({ url: 'https://x.com/a.mp4', name: 'no hint here' }), null)
})

test('real pixel size maps to a class, widescreen included', () => {
  assert.strictEqual(lib.bucketFromSize(1920, 800), 1080)
  assert.strictEqual(lib.bucketFromSize(1280, 536), 720)
  assert.strictEqual(lib.bucketFromSize(3840, 1600), 2160)
  assert.strictEqual(lib.bucketFromSize(720, 404), 480)
})

test('preferred resolutions first, then whatever else exists, nothing dropped', () => {
  const mk = (n) => ({ url: `https://x.com/${n}`, name: n })
  const s = [mk('a 480p'), mk('b 2160p'), mk('c untagged'), mk('d 1080p'), mk('e 720p'), mk('f 576p')]
  const ranked = lib.rankStreams(s).map(x => x.name[0])
  assert.deepStrictEqual(ranked, ['e', 'd', 'b', 'f', 'a', 'c'])  // 720, 1080, 4K, then 576, 480, then unknown
  assert.deepStrictEqual(lib.rankStreams(s, [2160, 1080, 720]).map(x => x.name[0]).slice(0, 3), ['b', 'd', 'e'])
  const noPreferred = [mk('x 480p'), mk('y untagged'), mk('z 576p')]
  assert.deepStrictEqual(lib.rankStreams(noPreferred).map(x => x.name[0]), ['z', 'x', 'y'])  // falls back to what exists
  assert.strictEqual(lib.rankStreams(noPreferred).length, 3)
})

test('untagged streams are measured and then ranked by their real size', async () => {
  const a = { url: 'https://x.com/a', name: 'plain A' }
  const b = { url: 'https://x.com/b', name: 'plain B' }
  const c = { url: 'https://x.com/c', name: 'broken' }
  const tagged = { url: 'https://x.com/t', name: 'tag 1080p' }
  const probe = async (url) => {
    if (url.endsWith('/a')) return { width: 1280, height: 536 }
    if (url.endsWith('/b')) return { width: 3840, height: 1600 }
    throw new Error('unreachable')
  }
  const measured = await lib.enrichResolutions([a, b, c, tagged], probe)
  assert.strictEqual(measured.get(a.url), 720)
  assert.strictEqual(measured.get(b.url), 2160)
  assert.ok(!measured.has(c.url) && !measured.has(tagged.url))  // tagged ones are not probed
  assert.deepStrictEqual(lib.rankStreams([c, b, tagged, a], [720, 1080, 2160], measured).map(x => x.name),
    ['plain A', 'tag 1080p', 'plain B', 'broken'])
})

test('a slow probe does not hold the answer back', async () => {
  const slow = () => new Promise(() => {})
  const t0 = Date.now()
  const m = await lib.enrichResolutions([{ url: 'https://x.com/a' }], slow, { timeoutMs: 100 })
  assert.ok(Date.now() - t0 < 1000 && m.size === 0)
})

test('restoration options become server parameters', () => {
  assert.deepStrictEqual(lib.restoreParams({ mode: 'medium', deflicker: true, stabilize: true, removeText: 'bottom' }),
    { restore: 'medium', deflicker: 1, stabilize: 1, rmtext: 'bottom' })
  assert.deepStrictEqual(lib.restoreParams({ mode: 'off', sharpen: 0, denoise: 0 }), {})
  assert.strictEqual(lib.restoreActive({ mode: 'off' }), false)
  assert.strictEqual(lib.restoreActive({ sharpen: 0.4 }), true)
})

test('colorized stream carries the restoration options, enhanced stream does not colorize', () => {
  const rest = { mode: 'light', removeText: '0,0.8,1,0.2' }
  const c = new URL(lib.colorizedStream({ url: 'https://x.com/a.mp4' }, SERVER, null, rest).url)
  assert.strictEqual(c.searchParams.get('restore'), 'light')
  assert.strictEqual(c.searchParams.get('rmtext'), '0,0.8,1,0.2')
  assert.ok(!c.searchParams.has('colorize'))
  const e = lib.enhancedStream({ url: 'https://x.com/a.mp4', name: 'N' }, SERVER, null, rest)
  assert.strictEqual(e.name, 'Enhanced (AI)')
  assert.strictEqual(new URL(e.url).searchParams.get('colorize'), '0')
  assert.strictEqual(lib.enhancedStream({ url: 'https://x.com/a.mp4' }, SERVER, null, { mode: 'off' }), null)
})

test('wrapStreams ranks first, adds enhanced streams only when restoration is on', () => {
  const s = [{ url: 'https://x.com/a', name: 'A 1080p' }, { url: 'https://x.com/b', name: 'B 720p' }]
  const off = lib.wrapStreams(s, SERVER, { restore: { mode: 'off' } })
  assert.deepStrictEqual(off.map(x => x.name), ['Colorized (AI)', 'Colorized (AI)'])
  assert.ok(off[0].title.includes('B 720p'))  // 720p is looked for first
  const on = lib.wrapStreams(s, SERVER, { restore: { mode: 'light' } })
  assert.deepStrictEqual(on.map(x => x.name), ['Colorized (AI)', 'Colorized (AI)', 'Enhanced (AI)', 'Enhanced (AI)'])
})
