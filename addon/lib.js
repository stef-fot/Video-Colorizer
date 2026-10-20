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
  vi: 'vie'
}

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

/** Turn one direct stream into its "Colorized (AI)" twin. */
function colorizedStream (stream, serverUrl, sub) {
  const headers = requestHeaders(stream)
  const original = [stream.name, stream.title || stream.description].filter(Boolean).join(' | ')
  const parts = subtitleParts(serverUrl, stream.url, headers, sub)
  const out = {
    name: 'Colorized (AI)',
    title: original ? `Colorized (AI)\n${original}` : 'Colorized (AI)',
    url: buildServerUrl(serverUrl, stream.url, headers, parts.extra),
    behaviorHints: {
      bingeGroup: 'colorstream-' + (stream.behaviorHints && stream.behaviorHints.bingeGroup || 'default')
    }
  }
  if (parts.subtitles) out.subtitles = parts.subtitles
  return out
}

/**
 * Given streams from upstream sources, return what the addon should reply with.
 * keepOriginals=true returns originals first, then the colorized twins.
 */
function wrapStreams (streams, serverUrl, { keepOriginals = false, max = 5, subtitles = null } = {}) {
  const direct = (streams || []).filter(isDirectHttpStream).slice(0, max)
  const wrapped = direct.map(s => colorizedStream(s, serverUrl, subtitles))
  return keepOriginals ? [...direct, ...wrapped] : wrapped
}

/** "https://host/path/manifest.json" -> "https://host/path" */
function addonBase (manifestUrl) {
  return manifestUrl.replace(/\/manifest\.json(\?.*)?$/i, '').replace(/\/+$/, '')
}

/** Library entries (your own files) -> stream for a cs: id. */
function libraryStream (item, serverUrl, sub) {
  return colorizedStream({ url: item.url, name: item.name, title: item.description || '' }, serverUrl, sub)
}

module.exports = { subtitleParts, isDirectHttpStream, buildServerUrl, requestHeaders, colorizedStream, wrapStreams, addonBase, libraryStream }
