'use strict'
const fs = require('fs')
const path = require('path')
const { addonBuilder, serveHTTP } = require('stremio-addon-sdk')
const lib = require('./lib')

// ---- configuration: addon.config.json, overridden by environment variables ----
function loadConfig () {
  const file = process.env.CS_ADDON_CONFIG || path.join(__dirname, 'addon.config.json')
  let cfg = {}
  if (fs.existsSync(file)) {
    try { cfg = JSON.parse(fs.readFileSync(file, 'utf8')) } catch (e) {
      console.error(`Cannot read ${file}: ${e.message}`)
    }
  }
  return {
    port: parseInt(process.env.CS_ADDON_PORT || cfg.port || 7000, 10),
    // Address the PLAYER uses to reach the colorization server.
    serverUrl: process.env.CS_SERVER_URL || cfg.serverUrl || 'http://127.0.0.1:7002',
    // Manifest URLs of addons YOU choose to use as stream sources (a Stremio addon can not see other addons' streams).
    upstreams: (process.env.CS_UPSTREAMS ? process.env.CS_UPSTREAMS.split(',') : cfg.upstreams || []).map(s => s.trim()).filter(Boolean),
    keepOriginals: process.env.CS_KEEP_ORIGINALS ? process.env.CS_KEEP_ORIGINALS === '1' : cfg.keepOriginals !== false,
    library: process.env.CS_LIBRARY || cfg.library || path.join(__dirname, 'library.json'),
    timeoutMs: cfg.timeoutMs || 8000,
    // By default EVERY direct stream is offered colorized, in the order the source gave them. Set rankByResolution
    // to true to put the preferred resolutions first, and maxStreams to a number to limit how many are wrapped.
    rankByResolution: process.env.CS_RANK ? process.env.CS_RANK === '1' : !!cfg.rankByResolution,
    maxStreams: parseInt(process.env.CS_MAX_STREAMS || cfg.maxStreams || 0, 10),
    // Which resolutions to look for first (only used when rankByResolution is true). Nothing is dropped: if none of them exists, what is available is used.
    resolutionOrder: process.env.CS_RESOLUTION_ORDER ? process.env.CS_RESOLUTION_ORDER.split(',').map(Number)
      : cfg.resolutionOrder || [720, 1080, 2160],
    // Picture restoration for any file: mode off|light|medium|strong, or sharpen/denoise 0..1, deflicker, stabilize,
    // removeText ("bottom", "top" or "x,y,w,h"). enhancedStream adds an "Enhanced (AI)" stream without colorizing.
    restore: {
      mode: process.env.CS_RESTORE || (cfg.restore && cfg.restore.mode) || 'off',
      sharpen: cfg.restore && cfg.restore.sharpen,
      denoise: cfg.restore && cfg.restore.denoise,
      deflicker: !!(cfg.restore && cfg.restore.deflicker),
      stabilize: !!(cfg.restore && cfg.restore.stabilize),
      removeText: (cfg.restore && cfg.restore.removeText) || '',
      enhancedStream: !(cfg.restore && cfg.restore.enhancedStream === false)
    },
    // subtitles: mode soft (a track per language) | burn (first language drawn on the picture) | off
    // langs: default = the 20 most spoken languages plus Greek. prepare: start recognition when a title is opened.
    subtitles: {
      mode: process.env.CS_SUBS_MODE || (cfg.subtitles && cfg.subtitles.mode) || 'soft',
      langs: (process.env.CS_SUBS_LANGS ? process.env.CS_SUBS_LANGS.split(',') : (cfg.subtitles && cfg.subtitles.langs) || lib.DEFAULT_LANGS)
        .map(s => s.trim()).filter(Boolean),
      source: (cfg.subtitles && cfg.subtitles.source) || undefined,
      withOriginals: !(cfg.subtitles && cfg.subtitles.withOriginals === false),
      prepare: (cfg.subtitles && cfg.subtitles.prepare) || 'first', // first | none
      prepareLangs: (cfg.subtitles && cfg.subtitles.prepareLangs) || ['el', 'en']
    }
  }
}
const config = loadConfig()

function loadLibrary () {
  try { return JSON.parse(fs.readFileSync(config.library, 'utf8')) } catch (e) { return [] }
}

const manifest = {
  id: 'org.colorstream.addon',
  version: '0.1.0',
  name: 'ColorStream',
  description: 'Adds a "Colorized (AI)" stream for black and white films. Runs locally, works only with ' +
    'direct HTTP(S) streams and files you own or that are public domain.',
  resources: ['stream', 'catalog', 'meta'],
  types: ['movie', 'series', 'other'],
  idPrefixes: ['tt', 'cs:'],
  catalogs: [{ type: 'movie', id: 'colorstream-library', name: 'ColorStream Library' }],
  behaviorHints: { configurable: false }
}

const builder = new addonBuilder(manifest)

async function fetchJson (url) {
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort(), config.timeoutMs)
  try {
    const res = await fetch(url, { signal: ctl.signal, headers: { accept: 'application/json' } })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } finally { clearTimeout(timer) }
}

/** Real size of a stream, measured by the colorization server (null if it cannot tell). */
async function probeSource (url, headers) {
  try {
    const q = `src=${encodeURIComponent(url)}` + (headers ? `&hdr=${encodeURIComponent(JSON.stringify(headers))}` : '')
    const res = await fetch(`${config.serverUrl.replace(/\/+$/, '')}/api/probe?${q}`)
    return res.ok ? await res.json() : null
  } catch (e) { return null }
}

/** Fire and forget: ask the server to start recognizing (and pre-translating) a film so subtitles are ready sooner. */
function prepareSubtitles (url, headers) {
  const sub = config.subtitles
  if (sub.mode === 'off' || sub.prepare === 'none') return
  fetch(`${config.serverUrl.replace(/\/+$/, '')}/api/subs/prepare`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ src: url, hdr: headers || undefined, langs: sub.prepareLangs })
  }).catch(() => {}) // the server may not be running, streams still work
}

builder.defineCatalogHandler(async ({ type, id }) => {
  if (id !== 'colorstream-library') return { metas: [] }
  const metas = loadLibrary().map(item => ({
    id: 'cs:' + item.id, type: 'movie', name: item.name, poster: item.poster, description: item.description,
    releaseInfo: item.year ? String(item.year) : undefined
  }))
  return { metas }
})

builder.defineMetaHandler(async ({ id }) => {
  const item = loadLibrary().find(i => 'cs:' + i.id === id)
  if (!item) return { meta: null }
  return { meta: { id, type: 'movie', name: item.name, poster: item.poster, description: item.description } }
})

builder.defineStreamHandler(async ({ type, id }) => {
  // 1) Your own library (local files or URLs you own / public domain)
  if (id.startsWith('cs:')) {
    const item = loadLibrary().find(i => 'cs:' + i.id === id)
    if (item) prepareSubtitles(item.url)
    return { streams: item ? [lib.libraryStream(item, config.serverUrl, config.subtitles, config.restore)] : [] }
  }
  // 2) Wrap direct streams from upstream addons you configured
  if (!config.upstreams.length) {
    console.log(`No upstream addons configured, nothing to wrap for ${id}. See README.`)
    return { streams: [] }
  }
  const results = await Promise.allSettled(config.upstreams.map(m =>
    fetchJson(`${lib.addonBase(m)}/stream/${type}/${encodeURIComponent(id)}.json`)))
  const all = []
  results.forEach((r, i) => {
    if (r.status === 'fulfilled') all.push(...(r.value.streams || []))
    else console.warn(`Upstream ${config.upstreams[i]} failed: ${r.reason && r.reason.message}`)
  })
  const direct = all.filter(lib.isDirectHttpStream)
  // streams that carry no 720p/1080p/4K tag are measured by the server, so they can still be ranked
  const measured = config.rankByResolution ? await lib.enrichResolutions(direct, probeSource) : new Map()
  const ranked = config.rankByResolution ? lib.rankStreams(direct, config.resolutionOrder, measured) : direct
  if (ranked[0]) prepareSubtitles(ranked[0].url, lib.requestHeaders(ranked[0]))
  const streams = lib.wrapStreams(all, config.serverUrl, {
    keepOriginals: config.keepOriginals, subtitles: config.subtitles, withOriginals: config.subtitles.withOriginals,
    restore: config.restore, enhanced: config.restore.enhancedStream, rank: config.rankByResolution,
    max: config.maxStreams, order: config.resolutionOrder, measured })
  const kinds = { direct: 0, infoHash: 0, youtube: 0, external: 0, other: 0 }
  for (const st of all) {
    if (lib.isDirectHttpStream(st)) kinds.direct++
    else if (st.infoHash) kinds.infoHash++
    else if (st.ytId) kinds.youtube++
    else if (st.externalUrl) kinds.external++
    else kinds.other++
  }
  console.log(`${id}: ${all.length} upstream streams (direct http(s): ${kinds.direct}, torrent infoHash: ${kinds.infoHash}, ` +
    `youtube: ${kinds.youtube}, external link: ${kinds.external}, other: ${kinds.other}), ${streams.length} returned`)
  if (all.length && !kinds.direct) {
    console.warn('None of the upstream streams has a direct http(s) url, so there is nothing to wrap. ColorStream only handles ' +
      'direct http(s) streams (your own files or sources that give a plain video URL), never torrent streams.')
  }
  return { streams, cacheMaxAge: 60 }
})

if (require.main === module) {
  const lib_items = loadLibrary().length
  console.log(`Upstream addons: ${config.upstreams.length}, library items: ${lib_items}, subtitles: ${config.subtitles.mode}, restore: ${config.restore.mode}`)
  if (!config.upstreams.length && !lib_items) {
    console.warn('WARNING: no upstream addons and no library items are configured, so Stremio will show NO streams. ' +
      'Copy library.example.json to library.json (and edit it), and/or put upstream manifest URLs in addon.config.json.')
  }
  serveHTTP(builder.getInterface(), { port: config.port })
  console.log(`ColorStream addon: http://127.0.0.1:${config.port}/manifest.json`)
  console.log(`Colorization server expected at ${config.serverUrl}`)
}

module.exports = { manifest, builder, config }
