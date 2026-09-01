#!/usr/bin/env python3
from __future__ import annotations

import ipaddress
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, unquote, urlparse

import httpx
from selectolax.parser import HTMLParser

# ------------------------------- Configuration --------------------------------

FETCH_TIMEOUT = float(os.environ.get("WEB_FETCH_TIMEOUT", "20"))
FETCH_MAX_BYTES = int(os.environ.get("WEB_FETCH_MAX_BYTES", str(3_000_000)))
FETCH_WORKERS = int(os.environ.get("WEB_FETCH_WORKERS", "4"))
# Budget for text injected into the prompt (chars, not tokens)
PER_PAGE_CHARS = int(os.environ.get("WEB_PER_PAGE_CHARS", "6000"))
TOTAL_CHARS = int(os.environ.get("WEB_TOTAL_CHARS", "16000"))

USER_AGENT = os.environ.get(
    "WEB_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ------------------------------ SSRF-safe fetching -----------------------------

_PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8", "2001:db8::/32",
)]


@lru_cache(maxsize=2048)
def _resolve_ips(host: str) -> Tuple[str, ...]:
    try:
        info = socket.getaddrinfo(host, None)
        return tuple(sorted({sockaddr[0] for *_, sockaddr in info}))
    except Exception:
        return tuple()


def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        host = (parsed.hostname or "").strip().strip(".")
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
            return not any(ip in net for net in _PRIVATE_NETS)
        except ValueError:
            pass
        ips = _resolve_ips(host)
        if not ips:
            return False
        return not any(
            any(ipaddress.ip_address(ip) in net for net in _PRIVATE_NETS) for ip in ips
        )
    except Exception:
        return False


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=httpx.Timeout(FETCH_TIMEOUT, connect=10.0),
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


def fetch_page(url: str, client: Optional[httpx.Client] = None) -> Dict[str, Any]:
    if not is_safe_url(url):
        return {"ok": False, "url": url, "error": "URL not allowed"}
    own = client is None
    client = client or _make_client()
    try:
        current = url
        for _hop in range(6):
            with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        return {"ok": False, "url": current, "error": "Empty redirect"}
                    nxt = str(httpx.URL(current).join(loc))
                    if not is_safe_url(nxt):
                        return {"ok": False, "url": current, "error": "Redirect to disallowed URL"}
                    current = nxt
                    continue
                if resp.status_code != 200:
                    return {"ok": False, "url": current, "status": resp.status_code,
                            "error": f"HTTP {resp.status_code}"}
                ctype = (resp.headers.get("content-type") or "").lower()
                if not any(t in ctype for t in ("text/", "html", "xml", "json")):
                    return {"ok": False, "url": current, "error": f"Unsupported content-type: {ctype}"}
                cl = resp.headers.get("content-length")
                if cl and cl.isdigit() and int(cl) > FETCH_MAX_BYTES:
                    return {"ok": False, "url": current, "error": "Too large"}
                buf, total = [], 0
                for chunk in resp.iter_bytes():
                    buf.append(chunk)
                    total += len(chunk)
                    if total > FETCH_MAX_BYTES:
                        return {"ok": False, "url": current, "error": "Too large"}
                html = b"".join(buf).decode("utf-8", errors="ignore")

            title, text = _extract_text(html, current, ctype)
            if not text:
                return {"ok": False, "url": current, "error": "No extractable text"}
            return {"ok": True, "url": current, "title": title, "text": text}
        return {"ok": False, "url": url, "error": "Too many redirects"}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}
    finally:
        if own:
            client.close()


def _extract_text(html: str, url: str, ctype: str) -> Tuple[str, str]:
    if "json" in ctype and "html" not in ctype:
        return "", _clean(html)
    try:
        import trafilatura
        meta = trafilatura.bare_extraction(
            html, url=url, include_comments=False, include_tables=True,
            favor_recall=True, with_metadata=True,
        )
        if meta is not None:
            # trafilatura >=2 returns a Document object; older returns a dict
            title = getattr(meta, "title", None) or (meta.get("title") if isinstance(meta, dict) else "") or ""
            text = getattr(meta, "text", None) or (meta.get("text") if isinstance(meta, dict) else "") or ""
            if not title:
                node = HTMLParser(html).css_first("title")
                title = _clean(node.text()) if node else ""
            if text and len(text) > 200:
                return _clean(title), text.strip()
    except Exception:
        pass
    # Fallback: strip DOM
    try:
        tree = HTMLParser(html)
        for sel in ("script", "style", "noscript", "nav", "footer", "header"):
            for node in tree.css(sel):
                node.decompose()
        title_node = tree.css_first("title")
        title = _clean(title_node.text()) if title_node else ""
        body = tree.body.text(separator="\n") if tree.body else ""
        return title, _clean_paragraphs(body)
    except Exception:
        return "", ""


def _clean_paragraphs(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ------------------------------- Web search (SERP) -----------------------------

class EngineBlocked(Exception):
    pass


_GOOGLE_SKIP_HOSTS = ("google.", "youtube.com/redirect", "webcache.googleusercontent")


def _google_search(query: str, limit: int, client: httpx.Client) -> List[Dict[str, str]]:
    url = f"https://www.google.com/search?q={quote_plus(query)}&num={min(limit * 2, 20)}&hl=en&gbv=1"
    resp = client.get(url, follow_redirects=True)
    final = str(resp.url)
    if resp.status_code == 429 or "/sorry/" in final or "consent.google" in final:
        raise EngineBlocked(f"google blocked (status={resp.status_code})")
    if resp.status_code != 200:
        raise EngineBlocked(f"google HTTP {resp.status_code}")
    if "unusual traffic" in resp.text[:5000].lower():
        raise EngineBlocked("google captcha page")

    out: List[Dict[str, str]] = []
    seen = set()
    tree = HTMLParser(resp.text)
    for a in tree.css("a"):
        href = a.attributes.get("href") or ""
        if href.startswith("/url?q="):
            real = unquote(href[len("/url?q="):].split("&")[0])
        elif href.startswith("http"):
            real = href
        else:
            continue
        if any(h in real for h in _GOOGLE_SKIP_HOSTS) or not is_safe_url(real):
            continue
        title = _clean(a.text())
        if not title or len(title) < 5 or real in seen:
            continue
        seen.add(real)
        out.append({"title": title[:200], "url": real, "engine": "google"})
        if len(out) >= limit:
            break
    return out


def _ddg_unwrap(href: str) -> str:
    m = re.search(r"[?&]uddg=([^&]+)", href or "")
    return unquote(m.group(1)) if m else (href or "")


def _ddg_search(query: str, limit: int, client: httpx.Client) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()

    def parse(html: str, selector: str) -> None:
        tree = HTMLParser(html)
        for a in tree.css(selector):
            real = _ddg_unwrap(a.attributes.get("href") or "")
            title = _clean(a.text())
            if not title or not is_safe_url(real) or real in seen:
                continue
            seen.add(real)
            out.append({"title": title[:200], "url": real, "engine": "ddg"})
            if len(out) >= limit:
                return

    try:
        r = client.get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                       follow_redirects=True)
        if r.status_code == 200:
            parse(r.text, "a.result__a")
    except Exception:
        pass
    if len(out) < limit:
        try:
            r = client.get(f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}",
                           follow_redirects=True)
            if r.status_code == 200:
                parse(r.text, "a.result-link")
        except Exception:
            pass
    return out


def _mojeek_search(query: str, limit: int, client: httpx.Client) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    r = client.get(f"https://www.mojeek.com/search?q={quote_plus(query)}",
                   follow_redirects=True)
    if r.status_code != 200:
        return out
    tree = HTMLParser(r.text)
    seen = set()
    for a in tree.css("ul.results-standard a.title, a.title"):
        href = a.attributes.get("href") or ""
        title = _clean(a.text())
        if not title or not is_safe_url(href) or href in seen:
            continue
        seen.add(href)
        out.append({"title": title[:200], "url": href, "engine": "mojeek"})
        if len(out) >= limit:
            break
    return out


def web_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    limit = max(1, min(int(limit), 10))
    results: List[Dict[str, str]] = []
    seen: set = set()

    def take(batch: List[Dict[str, str]]) -> None:
        for r in batch:
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
            if len(results) >= limit:
                return

    with _make_client() as client:
        for engine in (_google_search, _ddg_search, _mojeek_search):
            if len(results) >= limit:
                break
            try:
                take(engine(query, limit, client))
            except Exception:
                continue
    return results[:limit]


# --------------------------- Search + fetch for chat ---------------------------

def search_and_fetch(query: str, limit: int = 3,
                     per_page_chars: int = PER_PAGE_CHARS,
                     total_chars: int = TOTAL_CHARS) -> Dict[str, Any]:
    hits = web_search(query, limit=limit)
    if not hits:
        return {"context": "", "sources": [], "errors": []}

    fetched: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(hits))) as ex:
        futs = {ex.submit(fetch_page, h["url"]): h["url"] for h in hits}
        for fut in as_completed(futs):
            fetched[futs[fut]] = fut.result()

    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    used = 0
    for h in hits:  # preserve engine ranking order
        page = fetched.get(h["url"]) or {}
        if not page.get("ok"):
            errors.append({"url": h["url"], "error": page.get("error", "fetch failed")})
            continue
        budget = min(per_page_chars, total_chars - used)
        if budget <= 0:
            break
        text = page["text"][:budget]
        used += len(text)
        n = len(sources) + 1
        title = page.get("title") or h["title"]
        blocks.append(f"[{n}] {title} — {page['url']}\n{text}")
        sources.append({"n": n, "title": title, "url": page["url"]})

    return {"context": "\n\n".join(blocks), "sources": sources, "errors": errors}
