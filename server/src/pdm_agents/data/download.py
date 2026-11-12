"""Download Backblaze Drive Stats quarterly zips for the configured years.

Links are discovered from the official Drive Stats page at run time (no guessed
URLs). The Hugging Face mirror (``backblaze/Drive_Stats``) is an alternative
source; it lists its files through the Hub API.

Backblaze asks users to cite Backblaze as the source of the data; see README.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from pdm_agents.config import AppConfig
from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

BB_ZIP_RE = re.compile(r"data_Q([1-4])_(\d{4})\.zip$")
HF_ZIP_RE = re.compile(r"(\d{4})_Q([1-4])\.zip$")
CHUNK = 1 << 20
USER_AGENT = "pdm-agents/0.1 (research; drive failure prediction)"


@dataclass(frozen=True)
class QuarterFile:
    year: int
    quarter: int
    url: str

    @property
    def filename(self) -> str:
        return f"data_Q{self.quarter}_{self.year}.zip"


def discover_backblaze(page_url: str, years: list[int], timeout: float = 60) -> list[QuarterFile]:
    """Parse the Drive Stats page and return quarterly zip links for ``years``."""
    resp = requests.get(page_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    found: dict[tuple[int, int], QuarterFile] = {}
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        m = BB_ZIP_RE.search(href)
        if not m:
            continue
        q, y = int(m.group(1)), int(m.group(2))
        if y in years:
            found[(y, q)] = QuarterFile(y, q, href)
    return [found[k] for k in sorted(found)]


def discover_huggingface(dataset: str, years: list[int], timeout: float = 60) -> list[QuarterFile]:
    """List quarterly zips of the Hugging Face mirror via the Hub tree API."""
    api = f"https://huggingface.co/api/datasets/{dataset}/tree/main/zip_csv"
    resp = requests.get(api, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    out = []
    for entry in resp.json():
        m = HF_ZIP_RE.search(entry.get("path", ""))
        if m and int(m.group(1)) in years:
            url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{entry['path']}"
            out.append(QuarterFile(int(m.group(1)), int(m.group(2)), url))
    return sorted(out, key=lambda f: (f.year, f.quarter))


def _remote_size(session: requests.Session, url: str, timeout: float) -> int | None:
    resp = session.head(url, allow_redirects=True, timeout=timeout)
    if resp.ok and resp.headers.get("Content-Length"):
        return int(resp.headers["Content-Length"])
    return None


def download_file(
    url: str, dest: Path, timeout: float = 60, max_stalled_attempts: int = 10
) -> Path:
    """Download ``url`` to ``dest`` with HTTP range resume and size verification.

    Retries indefinitely while bytes keep arriving; gives up after
    ``max_stalled_attempts`` consecutive attempts that made no progress.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    total = _remote_size(session, url, timeout)
    if dest.exists() and total is not None and dest.stat().st_size == total:
        log.info("already downloaded %s (%.1f MB)", dest.name, total / 1e6)
        return dest

    stalled = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have == total:
            break
        if stalled >= max_stalled_attempts:
            raise RuntimeError(f"could not download {url}: no progress in {stalled} attempts")
        start = have
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
                if have and resp.status_code != 206:
                    have = 0  # server ignored the range: restart
                resp.raise_for_status()
                mode = "ab" if have else "wb"
                last_log = time.monotonic()
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(CHUNK):
                        fh.write(chunk)
                        have += len(chunk)
                        if time.monotonic() - last_log > 15:
                            pct = f"{100 * have / total:.1f}%" if total else f"{have / 1e6:.0f} MB"
                            log.info("%s: %s", dest.name, pct)
                            last_log = time.monotonic()
            if total is None or have == total:
                break
        except (requests.RequestException, OSError) as exc:
            have = part.stat().st_size if part.exists() else 0
            stalled = 0 if have > start else stalled + 1
            wait = min(60, 2 ** max(stalled, 1))
            log.warning("%s interrupted at %.1f MB (%s); retrying in %ds",
                        dest.name, have / 1e6, type(exc).__name__, wait)
            time.sleep(wait)

    size = part.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(f"{dest.name}: size {size} != expected {total}")
    part.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, size / 1e6)
    return dest


def discover(cfg: AppConfig) -> list[QuarterFile]:
    """Quarterly files available for the configured source and years."""
    if cfg.data.source == "huggingface":
        files = discover_huggingface(cfg.data.huggingface_dataset, cfg.data.years)
    else:
        files = discover_backblaze(cfg.data.drive_stats_page, cfg.data.years)
    missing = sorted(
        {(y, q) for y in cfg.data.years for q in range(1, 5)} - {(f.year, f.quarter) for f in files}
    )
    if missing:
        log.warning("quarters not published by %s: %s", cfg.data.source, missing)
    return files


def download_all(cfg: AppConfig, workers: int | None = None) -> list[Path]:
    """Download every configured quarter into ``paths.raw`` (in parallel); returns zip paths."""
    files = discover(cfg)
    if not files:
        raise RuntimeError(f"no Drive Stats zips found for years {cfg.data.years}")
    workers = workers or cfg.data.download_workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_file, f.url, cfg.paths.raw / f.filename) for f in files]
        return [fut.result() for fut in futures]
