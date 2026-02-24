"""Message strings in the language of the user's question (for warnings and labels)."""

from typing import Dict

# LangChain BraveSearch returns list of {title, link, snippet}; we use "link" for URL
try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except Exception:
    detect = None


def detect_language(question: str) -> str:
    """Return ISO 639-1 code (e.g. 'en', 'de', 'da') from question text. Falls back to 'en'."""
    if not question or not question.strip():
        return "en"
    if detect is None:
        return "en"
    text = question.strip()
    if len(text) < 3:
        return "en"
    try:
        return detect(text) or "en"
    except Exception:
        return "en"


# Web search could not find good sources; answer may be based on insufficient information.
WARNING_WEB_SEARCH_POOR: Dict[str, str] = {
    "en": "The web search could not find sufficiently good sources. The answer may be based on insufficient information.",
    "de": "Die Websuche konnte keine ausreichend guten Quellen liefern. Die Antwort basiert möglicherweise auf unzureichenden Informationen.",
    "da": "Websøgningen kunne ikke finde tilstrækkeligt gode kilder. Svaret kan være baseret på utilstrækkelige oplysninger.",
}

# Label shown when Brave Search was used this turn (sources include web).
LABEL_SOURCES_INCL_WEB: Dict[str, str] = {
    "en": "Sources incl. web search",
    "de": "Quellen inkl. Websuche",
    "da": "Kilder inkl. websøgning",
}

# No trusted sources available; answer could not be verified against IAEA or Danish sources.
WARNING_NO_TRUSTED_SOURCES: Dict[str, str] = {
    "en": "No trusted sources available. The answer could not be verified against IAEA or Danish sources.",
    "de": "Keine vertrauenswürdigen Quellen verfügbar. Die Antwort konnte nicht gegen IAEA- oder dänische Quellen bestätigt werden.",
    "da": "Ingen pålidelige kilder tilgængelige. Svaret kunne ikke verificeres mod IAEA- eller danske kilder.",
}

# Answer not verified after also checking trusted web sources.
WARNING_NOT_VERIFIED_AFTER_WEB: Dict[str, str] = {
    "en": "The answer could not be verified against IAEA or Danish official sources. Trusted web sources were also checked.",
    "de": "Die Antwort konnte nicht gegen IAEA- oder dänische offizielle Quellen bestätigt werden. Es wurden auch vertrauenswürdige Webquellen geprüft.",
    "da": "Svaret kunne ikke verificeres mod IAEA eller danske officielle kilder. Pålidelige webkilder blev også tjekket.",
}

# Answer not fully verified against the provided trusted sources (no web search involved).
WARNING_NOT_VERIFIED_TRUSTED_ONLY: Dict[str, str] = {
    "en": "The answer could not be fully verified against the provided trusted sources.",
    "de": "Die Antwort konnte nicht vollständig gegen die bereitgestellten vertrauenswürdigen Quellen bestätigt werden.",
    "da": "Svaret kunne ikke fuldt ud verificeres mod de angivne pålidelige kilder.",
}


def get_warning_web_search_poor(lang: str) -> str:
    return WARNING_WEB_SEARCH_POOR.get(lang) or WARNING_WEB_SEARCH_POOR["en"]


def get_label_sources_incl_web(lang: str) -> str:
    return LABEL_SOURCES_INCL_WEB.get(lang) or LABEL_SOURCES_INCL_WEB["en"]


def get_warning_no_trusted_sources(lang: str) -> str:
    return WARNING_NO_TRUSTED_SOURCES.get(lang) or WARNING_NO_TRUSTED_SOURCES["en"]


def get_warning_not_verified_after_web(lang: str) -> str:
    return WARNING_NOT_VERIFIED_AFTER_WEB.get(lang) or WARNING_NOT_VERIFIED_AFTER_WEB["en"]


def get_warning_not_verified_trusted_only(lang: str) -> str:
    return WARNING_NOT_VERIFIED_TRUSTED_ONLY.get(lang) or WARNING_NOT_VERIFIED_TRUSTED_ONLY["en"]
