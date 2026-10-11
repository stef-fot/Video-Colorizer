'use strict'
// Pure helpers (no network, no SDK) so they can be unit tested.

/** Only direct http(s) URLs are accepted. Torrents, magnets, YouTube ids etc. are ignored. */
function isDirectHttpStream (stream) {
  return !!stream && typeof stream.url === 'string' && /^https?:\/\//i.test(stream.url)
}

/** Build the URL the player will open: the local colorization server's HLS endpoint. */
function buildServerUrl (serverUrl, sourceUrl, headers) {
  const base = serverUrl.replace(/\/+$/, '')
  let url = `${base}/stream?src=${encodeURIComponent(sourceUrl)}`
  if (headers && Object.keys(headers).length) {
    url += `&hdr=${encodeURIComponent(JSON.stringify(headers))}`
  }
  return url
}

/** Request headers the original stream needs, as declared by Stremio's behaviorHints. */
function requestHeaders (stream) {
  const h = stream && stream.behaviorHints && stream.behaviorHints.proxyHeaders
  return h && h.request ? h.request : null
}

/** Turn one direct stream into its "Colorized (AI)" twin. */
function colorizedStream (stream, serverUrl) {
  const headers = requestHeaders(stream)
  const original = [stream.name, stream.title || stream.description].filter(Boolean).join(' | ')
  return {
    name: 'Colorized (AI)',
    title: original ? `Colorized (AI)\n${original}` : 'Colorized (AI)',
    url: buildServerUrl(serverUrl, stream.url, headers),
    behaviorHints: {
      bingeGroup: 'colorstream-' + (stream.behaviorHints && stream.behaviorHints.bingeGroup || 'default')
    }
  }
}

/**
 * Given streams from upstream sources, return what the addon should reply with.
 * keepOriginals=true returns originals first, then the colorized twins.
 */
function wrapStreams (streams, serverUrl, { keepOriginals = false, max = 5 } = {}) {
  const direct = (streams || []).filter(isDirectHttpStream).slice(0, max)
  const wrapped = direct.map(s => colorizedStream(s, serverUrl))
  return keepOriginals ? [...direct, ...wrapped] : wrapped
}

/** "https://host/path/manifest.json" -> "https://host/path" */
function addonBase (manifestUrl) {
  return manifestUrl.replace(/\/manifest\.json(\?.*)?$/i, '').replace(/\/+$/, '')
}

/** Library entries (your own files) -> stream for a cs: id. */
function libraryStream (item, serverUrl) {
  return colorizedStream({ url: item.url, name: item.name, title: item.description || '' }, serverUrl)
}

module.exports = { isDirectHttpStream, buildServerUrl, requestHeaders, colorizedStream, wrapStreams, addonBase, libraryStream }
