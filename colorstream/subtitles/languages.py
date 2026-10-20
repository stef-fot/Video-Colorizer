"""Language table: names, ISO 639-2 codes (for container tags), script class and legacy encodings."""
from __future__ import annotations

# code: (name, iso639-2, script, legacy encodings tried for old subtitle files)
LANGUAGES = {
    "en": ("English", "eng", "latin", ["cp1252"]),
    "el": ("Greek", "ell", "greek", ["cp1253", "iso8859-7"]),
    "es": ("Spanish", "spa", "latin", ["cp1252"]),
    "fr": ("French", "fra", "latin", ["cp1252"]),
    "de": ("German", "deu", "latin", ["cp1252"]),
    "it": ("Italian", "ita", "latin", ["cp1252"]),
    "pt": ("Portuguese", "por", "latin", ["cp1252"]),
    "nl": ("Dutch", "nld", "latin", ["cp1252"]),
    "sv": ("Swedish", "swe", "latin", ["cp1252"]),
    "da": ("Danish", "dan", "latin", ["cp1252"]),
    "no": ("Norwegian", "nor", "latin", ["cp1252"]),
    "fi": ("Finnish", "fin", "latin", ["cp1252"]),
    "pl": ("Polish", "pol", "latin", ["cp1250"]),
    "cs": ("Czech", "ces", "latin", ["cp1250"]),
    "hu": ("Hungarian", "hun", "latin", ["cp1250"]),
    "ro": ("Romanian", "ron", "latin", ["cp1250"]),
    "tr": ("Turkish", "tur", "latin", ["cp1254"]),
    "ru": ("Russian", "rus", "cyrillic", ["cp1251", "koi8-r"]),
    "uk": ("Ukrainian", "ukr", "cyrillic", ["cp1251", "koi8-u"]),
    "bg": ("Bulgarian", "bul", "cyrillic", ["cp1251"]),
    "sr": ("Serbian", "srp", "cyrillic", ["cp1251"]),
    "ar": ("Arabic", "ara", "arabic", ["cp1256"]),
    "fa": ("Persian", "fas", "arabic", ["cp1256"]),
    "he": ("Hebrew", "heb", "hebrew", ["cp1255"]),
    "hi": ("Hindi", "hin", "indic", []),
    "th": ("Thai", "tha", "thai", ["cp874"]),
    "zh": ("Chinese", "zho", "cjk", ["gb18030", "big5"]),
    "ja": ("Japanese", "jpn", "cjk", ["shift_jis", "cp932"]),
    "ko": ("Korean", "kor", "cjk", ["cp949"]),
    "id": ("Indonesian", "ind", "latin", ["cp1252"]),
    "vi": ("Vietnamese", "vie", "latin", ["cp1258"]),
}

# Scripts that have no spaces between words: wrap anywhere.
NO_SPACE_SCRIPTS = {"cjk", "thai"}
RTL_SCRIPTS = {"arabic", "hebrew"}


def normalize_lang(code: str | None) -> str:
    """'el-GR', 'ell', 'Greek' -> 'el'. Unknown or empty -> 'auto'."""
    if not code:
        return "auto"
    c = code.strip().lower().replace("_", "-")
    if c == "auto":
        return "auto"
    base = c.split("-")[0]
    if base in LANGUAGES:
        return base
    for k, (name, iso3, _, _) in LANGUAGES.items():
        if base == iso3 or base == name.lower():
            return k
    if base in ("gre", "ell", "gr"):  # 'gr' is what many subtitle file names use for Greek
        return "el"
    return base


def iso3(code: str) -> str:
    return LANGUAGES.get(code, ("", code, "", []))[1]


def script_of(code: str) -> str:
    return LANGUAGES.get(code, ("", "", "latin", []))[2]


def script_of_text(text: str) -> str:
    """Rough script detection from the characters, used when the language is unknown."""
    counts: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        if 0x0370 <= o <= 0x03FF or 0x1F00 <= o <= 0x1FFF:
            s = "greek"
        elif 0x0400 <= o <= 0x04FF:
            s = "cyrillic"
        elif 0x0590 <= o <= 0x05FF:
            s = "hebrew"
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
            s = "arabic"
        elif 0x0900 <= o <= 0x097F:
            s = "indic"
        elif 0x0E00 <= o <= 0x0E7F:
            s = "thai"
        elif 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
            s = "cjk"
        elif ch.isalpha():
            s = "latin"
        else:
            continue
        counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else "latin"
