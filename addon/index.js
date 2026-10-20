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
    // subtitles: mode off | soft | burn, langs e.g. ["el"] or ["el","en"], source auto | existing | asr
    subtitles: {
      mode: process.env.CS_SUBS_MODE || (cfg.subtitles && cfg.subtitles.mode) || 'off',
      langs: (process.env.CS_SUBS_LANGS ? process.env.CS_SUBS_LANGS.split(',') : (cfg.subtitles && cfg.subtitles.langs) || [])
        .map(s => s.trim()).filter(Boolean),
      source: (cfg.subtitles && cfg.subtitles.source) || undefined
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
    return { streams: item ? [lib.libraryStream(item, config.serverUrl, config.subtitles)] : [] }
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
  const streams = lib.wrapStreams(all, config.serverUrl, { keepOriginals: config.keepOriginals, subtitles: config.subtitles })
  console.log(`${id}: ${all.length} upstream streams, ${streams.length} returned`)
  return { streams, cacheMaxAge: 60 }
})

if (require.main === module) {
  serveHTTP(builder.getInterface(), { port: config.port })
  console.log(`ColorStream addon: http://127.0.0.1:${config.port}/manifest.json`)
  console.log(`Colorization server expected at ${config.serverUrl}`)
}

module.exports = { manifest, builder, config }
