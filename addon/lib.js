'use strict'
// Pure helpers (no network, no SDK) so they can be unit tested.

/** Only direct http(s) URLs are accepted. Torrents, magnets, YouTube ids etc. are ignored. */
function isDirectHttpStream (stream) {
  return !!stream && typeof stream.url === 'string' && /^https?:\/\//i.test(stream.url)
}

/** Build the URL the player will open: the local colorization server's HLS endpoint. */
function buildServerUrl (serverUrl, sourceUrl, headers, extra) {
  const base = serverUrl.replace(/\/+$/, '')
  let url = `${base}/stream?src=${encodeURIComponent(sourceUrl)}`
  if (headers && Object.keys(headers).length) {
    url += `&hdr=${encodeURIComponent(JSON.stringify(headers))}`
  }
  for (const [k, v] of Object.entries(extra || {})) {
    if (v !== undefined && v !== null && v !== '') url += `&${k}=${encodeURIComponent(v)}`
  }
  return url
}

// ISO 639-1 (what the server uses) -> ISO 639-2 (what Stremio expects in subtitle objects)
const ISO3 = {
  en: 'eng', el: 'ell', es: 'spa', fr: 'fra', de: 'deu', it: 'ita', pt: 'por', nl: 'nld', sv: 'swe', da: 'dan',
  no: 'nor', fi: 'fin', pl: 'pol', cs: 'ces', hu: 'hun', ro: 'ron', tr: 'tur', ru: 'rus', uk: 'ukr', bg: 'bul',
  sr: 'srp', ar: 'ara', fa: 'fas', he: 'heb', hi: 'hin', th: 'tha', zh: 'zho', ja: 'jpn', ko: 'kor', id: 'ind',
  vi: 'vie', bn: 'ben', ur: 'urd', sw: 'swa', mr: 'mar', te: 'tel', ta: 'tam'
}

// 20 most spoken languages (approximate ranking) plus Greek
const DEFAULT_LANGS = ['en', 'zh', 'hi', 'es', 'fr', 'ar', 'bn', 'pt', 'ru', 'ur', 'id', 'de', 'ja', 'sw', 'mr', 'te',
  'tr', 'ta', 'vi', 'ko', 'el']

/**
 * Subtitle options -> pieces for the stream.
 *  mode 'burn': the server draws the first language onto every frame (frame exact, any player, any speed)
 *  mode 'soft': Stremio gets one subtitle track per language, created by the server
 */
function subtitleParts (serverUrl, sourceUrl, headers, sub) {
  const none = { extra: {}, subtitles: undefined }
  if (!sub || !sub.mode || sub.mode === 'off' || !(sub.langs && sub.langs.length)) return none
  const base = serverUrl.replace(/\/+$/, '')
  const extra = {}
  if (sub.mode === 'burn') {
    extra.subs = 'burn'
    extra.lang = sub.langs[0]
    if (sub.source) extra.subsrc = sub.source
    return { extra, subtitles: undefined }
  }
  const hdr = headers && Object.keys(headers).length ? `&hdr=${encodeURIComponent(JSON.stringify(headers))}` : ''
  const subtitles = sub.langs.map(l => ({
    id: `colorstream-${l}`,
    lang: ISO3[l] || l,
    url: `${base}/subs.vtt?src=${encodeURIComponent(sourceUrl)}&lang=${encodeURIComponent(l)}` +
      (sub.source ? `&subsrc=${encodeURIComponent(sub.source)}` : '') + hdr
  }))
  return { extra, subtitles }
}

/** Request headers the original stream needs, as declared by Stremio's behaviorHints. */
function requestHeaders (stream) {
  const h = stream && stream.behaviorHints && stream.behaviorHints.proxyHeaders
  return h && h.request ? h.request : null
}

// ---- picture restoration options -> query parameters of the server
function restoreParams (rest) {
  const x = {}
  if (!rest) return x
  if (rest.mode && rest.mode !== 'off') x.restore = rest.mode
  if (rest.sharpen > 0) x.sharpen = rest.sharpen
  if (rest.denoise > 0) x.denoise = rest.denoise
  if (rest.deflicker) x.deflicker = 1
  if (rest.stabilize) x.stabilize = 1
  if (rest.removeText) x.rmtext = rest.removeText
  return x
}

function restoreActive (rest) {
  return Object.keys(restoreParams(rest)).length > 0
}

// ---- ranking by resolution. Tags are optional: untagged streams are measured by the server (see enrichResolutions)
const RES_TAGS = [
  [2160, /\b(2160p?|4k|uhd)\b/i], [1440, /\b1440p?\b/i], [1080, /\b(1080[pi]?|full[ -]?hd|fhd)\b/i],
  [720, /\b720p?\b/i], [576, /\b576p?\b/i], [480, /\b(480p?|sd)\b/i]
]

function textOf (stream) {
  const h = stream.behaviorHints || {}
  let file = ''
  try { file = decodeURIComponent(new URL(stream.url).pathname) } catch (e) { file = String(stream.url || '') }
  return [stream.name, stream.title, stream.description, h.filename, file].filter(Boolean).join(' ')
}

/** 2160 / 1440 / 1080 / 720 / 576 / 480, or null when the stream says nothing about its size. */
function detectResolution (stream) {
  const t = textOf(stream)
  for (const [res, re] of RES_TAGS) if (re.test(t)) return res
  return null
}

/** Resolution class from real pixel size (a 1920x800 widescreen film is 1080p). */
function bucketFromSize (width, height) {
  if (width >= 3200 || height >= 1800) return 2160
  if (width >= 2400 || height >= 1300) return 1440
  if (width >= 1800 || height >= 1000) return 1080
  if (width >= 1200 || height >= 700) return 720
  if (width >= 900 || height >= 520) return 576
  return 480
}

/**
 * Best first: the resolutions in `order` (default 720, 1080, 4K) come first in that order, then any other known
 * resolution (higher first), then streams whose size could not be found. Nothing is dropped: if none of the
 * preferred ones exists, whatever is available is used. Stable for equal ranks.
 */
function rankStreams (streams, order = [720, 1080, 2160], measured = new Map()) {
  const key = (s) => {
    const r = detectResolution(s) || measured.get(s.url) || null
    if (r === null) return [2, 0]
    const i = order.indexOf(r)
    return i >= 0 ? [0, i] : [1, -r]
  }
  return streams.map((s, i) => ({ s, i, k: key(s) }))
    .sort((a, b) => a.k[0] - b.k[0] || a.k[1] - b.k[1] || a.i - b.i)
    .map(x => x.s)
}

/** Find the real size of untagged streams through `probe(url) -> {width, height}`. Failures are ignored. */
async function enrichResolutions (streams, probe, { concurrency = 4, timeoutMs = 5000 } = {}) {
  const measured = new Map()
  const todo = streams.filter(s => s.url && !detectResolution(s))
  let next = 0
  const worker = async () => {
    while (next < todo.length) {
      const s = todo[next++]
      try {
        const info = await Promise.race([
          probe(s.url, requestHeaders(s)),
          new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs))
        ])
        if (info && info.width && info.height) measured.set(s.url, bucketFromSize(info.width, info.height))
      } catch (e) { /* unknown size: ranked last, still playable */ }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, todo.length) }, worker))
  return measured
}

/** The ORIGINAL stream, untouched and played directly by Stremio, plus AI subtitles in every language. */
function subtitledOriginal (stream, serverUrl, sub) {
  if (!sub || !sub.mode || sub.mode === 'off') return null
  const parts = subtitleParts(serverUrl, stream.url, requestHeaders(stream), { ...sub, mode: 'soft' })
  if (!parts.subtitles) return null
  const original = [stream.name, stream.title || stream.description].filter(Boolean).join(' | ')
  return {
    ...stream,
    name: 'AI subtitles',
    title: original ? `AI subtitles (${parts.subtitles.length} languages)\n${original}` : `AI subtitles (${parts.subtitles.length} languages)`,
    subtitles: parts.subtitles
  }
}

/** Turn one direct stream into its "Colorized (AI)" twin. */
function colorizedStream (stream, serverUrl, sub, rest, label = 'Colorized (AI)', extraParams = {}) {
  const headers = requestHeaders(stream)
  const original = [stream.name, stream.title || stream.description].filter(Boolean).join(' | ')
  const parts = subtitleParts(serverUrl, stream.url, headers, sub)
  const out = {
    name: label,
    title: original ? `${label}\n${original}` : label,
    url: buildServerUrl(serverUrl, stream.url, headers, { ...parts.extra, ...restoreParams(rest), ...extraParams }),
    behaviorHints: {
      bingeGroup: 'colorstream-' + (stream.behaviorHints && stream.behaviorHints.bingeGroup || 'default')
    }
  }
  if (parts.subtitles) out.subtitles = parts.subtitles
  return out
}

/** Restoration only (sharper, steadier, text removed), colors untouched. Works for color and black and white. */
function enhancedStream (stream, serverUrl, sub, rest) {
  if (!restoreActive(rest)) return null
  return colorizedStream(stream, serverUrl, sub, rest, 'Enhanced (AI)', { colorize: 0 })
}

/**
 * Given streams from upstream sources, return what the addon should reply with.
 * keepOriginals=true returns originals first, then the colorized twins.
 */
function wrapStreams (streams, serverUrl, { keepOriginals = false, max = 0, subtitles = null, withOriginals = false,
  restore = null, enhanced = true, rank = false, order = [720, 1080, 2160], measured = new Map() } = {}) {
  // Default: EVERY direct stream is wrapped, in the order the source gave them (max 0 = no limit, rank false = no ranking).
  let direct = (streams || []).filter(isDirectHttpStream)
  if (rank) direct = rankStreams(direct, order, measured)
  if (max > 0) direct = direct.slice(0, max)
  const wrapped = direct.map(s => colorizedStream(s, serverUrl, subtitles, restore))
  const sharper = enhanced ? direct.map(s => enhancedStream(s, serverUrl, subtitles, restore)).filter(Boolean) : []
  const subbed = withOriginals ? direct.map(s => subtitledOriginal(s, serverUrl, subtitles)).filter(Boolean) : []
  return keepOriginals ? [...direct, ...wrapped, ...sharper, ...subbed] : [...wrapped, ...sharper, ...subbed]
}

/** "https://host/path/manifest.json" -> "https://host/path" */
function addonBase (manifestUrl) {
  return manifestUrl.replace(/\/manifest\.json(\?.*)?$/i, '').replace(/\/+$/, '')
}

/** Library entries (your own files) -> stream for a cs: id. */
function libraryStream (item, serverUrl, sub, rest) {
  return colorizedStream({ url: item.url, name: item.name, title: item.description || '' }, serverUrl, sub, rest)
}

module.exports = { restoreParams, restoreActive, detectResolution, bucketFromSize, rankStreams, enrichResolutions, enhancedStream, DEFAULT_LANGS, subtitledOriginal, subtitleParts, isDirectHttpStream, buildServerUrl, requestHeaders, colorizedStream, wrapStreams, addonBase, libraryStream }
