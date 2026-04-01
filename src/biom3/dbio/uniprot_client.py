"""UniProt REST API client with batch fetching, caching, and rate limiting."""

import json
import os
import time
from pathlib import Path

import requests

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb"
DEFAULT_CACHE_DIR = ".uniprot_cache"
BATCH_SIZE = 25
REQUEST_DELAY = 0.5
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0


class UniProtCache:
    """Simple disk-based JSON cache keyed by accession ID."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, accession):
        return self.cache_dir / f"{accession}.json"

    def get(self, accession):
        path = self._path(accession)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache entry for %s, deleting", accession)
            path.unlink(missing_ok=True)
            return None

    def put(self, accession, data):
        self._path(accession).write_text(json.dumps(data))

    def has(self, accession):
        return self._path(accession).exists()


class UniProtClient:
    """Fetches protein annotations from UniProt REST API."""

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, use_cache=True):
        self.cache = UniProtCache(cache_dir) if use_cache else None
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _request_with_retry(self, method, url, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Rate limited, waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Server error %d, retrying in %.1fs",
                                   resp.status_code, wait)
                    time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.warning("Request failed (%s), retrying in %.1fs", e, wait)
                    time.sleep(wait)
                else:
                    raise
        return None

    def fetch_batch(self, accessions):
        """Fetch annotations for a batch of accessions via search endpoint."""
        query = " OR ".join(f"accession:{a}" for a in accessions)
        resp = self._request_with_retry(
            "GET",
            UNIPROT_SEARCH_URL,
            params={"query": query, "format": "json", "size": 500},
        )
        if resp is None:
            logger.warning("Batch fetch failed for %d accessions (no response)",
                           len(accessions))
            return {}
        if resp.status_code != 200:
            logger.warning("Batch fetch failed for %d accessions (HTTP %d: %s)",
                           len(accessions), resp.status_code,
                           resp.text[:200] if resp.text else "")
            return {}

        data = resp.json()
        results = {}
        for entry in data.get("results", []):
            acc = entry.get("primaryAccession")
            if acc:
                results[acc] = entry
        return results

    def fetch_single(self, accession):
        """Fetch a single accession as fallback."""
        resp = self._request_with_retry(
            "GET", f"{UNIPROT_ENTRY_URL}/{accession}.json"
        )
        if resp is None or resp.status_code != 200:
            return None
        return resp.json()

    def fetch_all(self, accessions, batch_size=BATCH_SIZE):
        """Fetch all accessions with batching, caching, and progress logging."""
        accessions = [a for a in accessions if a and str(a) != "nan"]

        results = {}
        uncached = []

        if self.cache:
            for acc in accessions:
                cached = self.cache.get(acc)
                if cached is not None:
                    results[acc] = cached
                else:
                    uncached.append(acc)
            logger.info("Cache: %s hits, %s to fetch",
                        f"{len(results):,}", f"{len(uncached):,}")
        else:
            uncached = list(accessions)

        if not uncached:
            return results

        total_batches = (len(uncached) + batch_size - 1) // batch_size
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            batch_num = i // batch_size + 1
            if batch_num % 20 == 1 or batch_num == total_batches:
                logger.info("Batch %d/%d (%s accessions fetched so far)",
                            batch_num, total_batches, f"{len(results):,}")

            batch_results = self.fetch_batch(batch)

            for acc, entry in batch_results.items():
                results[acc] = entry
                if self.cache:
                    self.cache.put(acc, entry)

            missing = set(batch) - set(batch_results.keys())
            for acc in missing:
                entry = self.fetch_single(acc)
                if entry:
                    results[acc] = entry
                    if self.cache:
                        self.cache.put(acc, entry)

            if i + batch_size < len(uncached):
                time.sleep(REQUEST_DELAY)

        fetched = len(results) - (len(accessions) - len(uncached))
        not_found = len(uncached) - fetched
        if not_found > 0:
            logger.warning("%d accessions not found in UniProt", not_found)

        return results
