"""Submit identified protein isoforms to EMBL-EBI InterProScan asynchronously.

The sequence-identification pipeline is reused directly; no protein FASTA
needs to be copied or reparsed:

    python 5a_protein_sequence/submitting_to_InterPro.py
    python 5a_protein_sequence/submitting_to_InterPro.py submit MEF2C 1 2 --email you@example.org
    python 5a_protein_sequence/submitting_to_InterPro.py check
    python 5a_protein_sequence/submitting_to_InterPro.py check --watch --poll-interval 300
    python 5a_protein_sequence/submitting_to_InterPro.py export --gene MEF2C
    python 5a_protein_sequence/submitting_to_InterPro.py export

Running without a subcommand starts an interactive prompt for submit, check,
and export operations. Exports are saved automatically in
5a_protein_sequence/CSV with a gene-specific or all-genes filename.

The API job registry, raw responses, and normalized annotations are stored in
5a_protein_sequence/interproscan_jobs/results.sqlite3. InterProScan 6's
official REST API is documented at
https://www.ebi.ac.uk/Tools/services/rest/iprscan6?json .
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Mapping
from urllib.parse import quote

import pandas as pd
import requests

ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOLDER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from identify_protein_seq import get_ranked_isoform_proteins

LOGGER = logging.getLogger(__name__)
DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent / "interproscan_jobs" / "results.sqlite3"
)
EXPORT_FOLDER = Path(__file__).resolve().parent / "CSV"
API_BASE_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan6"
API_MAX_JOBS = 30

ACTIVE_STATUSES = {
    "RUNNING",
    "SUBMITTING",
    "SUBMISSION_UNKNOWN",
    "RESULT_RETRIEVAL_FAILED",
    "UNKNOWN_STATUS",
}
RETRYABLE_STATUSES = {"FAILED", "EXPIRED"}


@dataclass(frozen=True)
class InterProConfig:
    """Network, batch, and polling limits for the InterProScan service."""

    base_url: str = API_BASE_URL
    max_active_jobs: int = API_MAX_JOBS
    submission_delay_seconds: float = 1.0
    poll_interval_seconds: float = 300.0
    request_timeout_seconds: float = 60.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    stale_submission_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_active_jobs <= API_MAX_JOBS:
            raise ValueError(f"max_active_jobs must be between 1 and {API_MAX_JOBS}")
        if self.submission_delay_seconds < 0:
            raise ValueError("submission_delay_seconds must not be negative")
        if self.poll_interval_seconds < 30:
            raise ValueError("poll_interval_seconds must be at least 30 seconds")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")
        if self.retry_backoff_seconds < 0 or self.stale_submission_seconds <= 0:
            raise ValueError("retry_backoff_seconds must be non-negative and stale timeout positive")


class InterProAPIError(RuntimeError):
    """An API request error, including whether a submission may have reached EBI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        submission_uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.submission_uncertain = submission_uncertain


class InterProScanClient:
    """Small client for the documented Job Dispatcher REST endpoints."""

    def __init__(
        self,
        config: InterProConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or InterProConfig()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Sapal-Speke-InterProScan/1.0"})

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _get(self, path: str) -> requests.Response:
        url = self._url(path)
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.request_timeout_seconds,
                )
                status_code = response.status_code
                if status_code == 429 or status_code >= 500:
                    if attempt + 1 < self.config.retry_attempts:
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else self.config.retry_backoff_seconds * (2**attempt)
                        )
                        time.sleep(delay)
                        continue
                response.raise_for_status()
                return response
            except requests.HTTPError as error:
                response = error.response
                status_code = response.status_code if response is not None else None
                if (
                    status_code in (429, 500, 502, 503, 504)
                    and attempt + 1 < self.config.retry_attempts
                ):
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                    continue
                detail = response.text[:500] if response is not None else str(error)
                raise InterProAPIError(
                    f"GET {url} failed ({status_code}): {detail}",
                    status_code=status_code,
                ) from error
            except requests.RequestException as error:
                if attempt + 1 == self.config.retry_attempts:
                    raise InterProAPIError(f"GET {url} failed: {error}") from error
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise InterProAPIError(f"GET {url} failed after retries")

    def submit(
        self,
        sequence: str,
        sequence_hash: str,
        email: str,
        title: str | None = None,
        sequence_name: str | None = None,
    ) -> str:
        """Submit one protein using the API's form-encoded /run endpoint."""
        url = self._url("/run")
        data = {
            "email": email,
            "title": title or f"sapal-speke-{sequence_hash[:16]}",
            "sequence": f">{sequence_name or sequence_hash}\n{sequence}\n",
            "stype": "p",
            "goterms": "true",
            "pathways": "true",
        }
        try:
            response = self.session.post(
                url,
                data=data,
                timeout=self.config.request_timeout_seconds,
            )
        except requests.RequestException as error:
            raise InterProAPIError(
                f"POST {url} failed before a job ID was received: {error}",
                submission_uncertain=True,
            ) from error

        if response.status_code >= 400:
            detail = response.text[:500]
            raise InterProAPIError(
                f"POST {url} failed ({response.status_code}): {detail}",
                status_code=response.status_code,
                submission_uncertain=response.status_code >= 500,
            )

        job_id = response.text.strip()
        if not job_id:
            raise InterProAPIError(
                "InterProScan accepted the request but returned an empty job ID",
                submission_uncertain=True,
            )
        return job_id

    def get_status(self, job_id: str) -> str:
        response = self._get(f"/status/{quote(job_id, safe='')}")
        return response.text.strip().upper()

    def get_result_types(self, job_id: str) -> list[str]:
        response = self._get(f"/resulttypes/{quote(job_id, safe='')}")
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as error:
            raise InterProAPIError(f"Invalid result-types XML for job {job_id}") from error
        return [
            element.text.strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "identifier"
            and element.text
            and element.text.strip()
        ]

    def get_result(self, job_id: str, result_type: str) -> str:
        response = self._get(
            f"/result/{quote(job_id, safe='')}/{quote(result_type, safe='')}"
        )
        return response.text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def _database(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the local persistent job and annotation tables if needed."""
    connection = _connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_hash TEXT PRIMARY KEY,
                protein_sequence TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS isoforms (
                sequence_hash TEXT NOT NULL REFERENCES sequences(sequence_hash),
                isoform_id TEXT NOT NULL,
                gene_name TEXT NOT NULL,
                isoform_rank INTEGER NOT NULL,
                transcript_id TEXT NOT NULL,
                PRIMARY KEY (sequence_hash, isoform_id)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_hash TEXT NOT NULL REFERENCES sequences(sequence_hash),
                job_id TEXT UNIQUE,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                submitted_at TEXT,
                last_checked_at TEXT,
                completed_at TEXT,
                error TEXT,
                result_type TEXT,
                raw_result TEXT
            );
            CREATE INDEX IF NOT EXISTS jobs_by_status
                ON jobs(status, id);
            CREATE INDEX IF NOT EXISTS jobs_by_sequence
                ON jobs(sequence_hash, id);
            CREATE TABLE IF NOT EXISTS annotations (
                job_pk INTEGER NOT NULL REFERENCES jobs(id),
                sequence_hash TEXT NOT NULL REFERENCES sequences(sequence_hash),
                match_index INTEGER NOT NULL,
                location_index INTEGER NOT NULL,
                sequence_identifier TEXT,
                interpro_accession TEXT,
                interpro_description TEXT,
                entry_type TEXT,
                member_database TEXT,
                signature_accession TEXT,
                signature_description TEXT,
                start INTEGER,
                end INTEGER,
                score TEXT,
                evalue TEXT,
                go_terms_json TEXT NOT NULL,
                pathways_json TEXT NOT NULL,
                raw_match_json TEXT NOT NULL,
                raw_location_json TEXT NOT NULL,
                PRIMARY KEY (job_pk, match_index, location_index)
            );
            """
        )
    finally:
        connection.close()


def _normalise_protein_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("Protein sequences must not be empty")
    if re.fullmatch(r"[A-Z]+", normalized) is None:
        raise ValueError("Protein sequences must contain only amino-acid letters")
    return normalized


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _latest_job(connection: sqlite3.Connection, sequence_hash: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM jobs WHERE sequence_hash = ? ORDER BY id DESC LIMIT 1",
        (sequence_hash,),
    ).fetchone()


def _active_job_count(connection: sqlite3.Connection) -> int:
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    row = connection.execute(
        f"SELECT COUNT(*) AS count FROM jobs WHERE status IN ({placeholders})",
        tuple(ACTIVE_STATUSES),
    ).fetchone()
    return int(row["count"])


def _interpro_job_names(
    isoforms: list[sqlite3.Row], sequence_hash: str
) -> tuple[str, str]:
    ranks_by_gene: dict[str, list[int]] = {}
    for isoform in isoforms:
        ranks_by_gene.setdefault(str(isoform["gene_name"]), []).append(
            int(isoform["isoform_rank"])
        )

    labels = []
    for gene_name, ranks in sorted(ranks_by_gene.items()):
        gene_token = re.sub(r"[^A-Za-z0-9._-]+", "-", gene_name).strip("-._")
        gene_token = gene_token or "UnknownGene"
        rank_text = "-".join(str(rank) for rank in sorted(set(ranks)))
        labels.append(f"{gene_token}-Iso-{rank_text}")

    identifier = "-".join(labels) or f"UnknownSeq-{sequence_hash[:12]}"
    return f"protseq-{identifier}", f"iprscan6-{identifier}"


def _sequence_names_by_hash(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, str]]:
    isoforms_by_hash: dict[str, list[sqlite3.Row]] = {}
    rows = connection.execute(
        """
        SELECT sequence_hash, gene_name, isoform_rank
        FROM isoforms
        ORDER BY gene_name, isoform_rank
        """
    ).fetchall()
    for row in rows:
        isoforms_by_hash.setdefault(str(row["sequence_hash"]), []).append(row)
    return {
        sequence_hash: _interpro_job_names(isoforms, sequence_hash)
        for sequence_hash, isoforms in isoforms_by_hash.items()
    }


def submit_sequences(
    gene_name: str,
    proteins_by_rank: Mapping[int, tuple[str, str]],
    *,
    email: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    config: InterProConfig | None = None,
    client: InterProScanClient | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Persist isoforms, deduplicate sequences, and submit one allowed job batch.

    ``proteins_by_rank`` is returned by ``identify_protein_seq.get_ranked_isoform_proteins``:
    rank -> (transcript ID, amino-acid sequence). Identical sequences share one
    remote job but retain every gene/rank/transcript mapping locally.
    """
    if not email or not email.strip():
        raise ValueError("A valid contact email is required by the InterProScan REST API")
    if not gene_name:
        raise ValueError("gene_name must not be empty")
    if not proteins_by_rank:
        raise ValueError("At least one protein sequence is required")

    cfg = config or InterProConfig()
    api = client or InterProScanClient(cfg)
    initialize_database(db_path)
    now = _utc_now()
    unique_sequences: dict[str, str] = {}
    reused = 0
    queued = 0

    with _database(db_path) as connection:
        for rank, (transcript_id, protein) in proteins_by_rank.items():
            if not isinstance(rank, int) or rank < 1:
                raise ValueError("Isoform ranks must be positive integers")
            sequence = _normalise_protein_sequence(protein)
            sequence_hash = _sequence_hash(sequence)
            unique_sequences[sequence_hash] = sequence
            isoform_id = f"{gene_name}|rank={rank}|transcript={transcript_id}"

            connection.execute(
                """
                INSERT OR IGNORE INTO sequences(sequence_hash, protein_sequence, created_at)
                VALUES (?, ?, ?)
                """,
                (sequence_hash, sequence, now),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO isoforms(
                    sequence_hash, isoform_id, gene_name, isoform_rank, transcript_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (sequence_hash, isoform_id, gene_name, rank, transcript_id),
            )

            success = connection.execute(
                "SELECT 1 FROM jobs WHERE sequence_hash = ? AND status = 'FINISHED' LIMIT 1",
                (sequence_hash,),
            ).fetchone()
            if success:
                reused += 1
                continue

            latest = _latest_job(connection, sequence_hash)
            if latest and latest["status"] in ACTIVE_STATUSES:
                if latest["status"] == "QUEUED":
                    connection.execute(
                        "UPDATE jobs SET email = ?, updated_at = ? WHERE id = ?",
                        (email.strip(), now, latest["id"]),
                    )
                    queued += 1
                else:
                    reused += 1
                continue
            if latest and latest["status"] not in RETRYABLE_STATUSES:
                reused += 1
                continue
            if latest and not retry_failed:
                reused += 1
                continue

            attempt = int(latest["attempt"]) + 1 if latest else 1
            connection.execute(
                """
                INSERT INTO jobs(
                    sequence_hash, status, attempt, email, created_at, updated_at
                ) VALUES (?, 'QUEUED', ?, ?, ?, ?)
                """,
                (sequence_hash, attempt, email.strip(), now, now),
            )
            queued += 1

    submitted_job_ids = _dispatch_queued(db_path, cfg, api)
    with _database(db_path) as connection:
        pending_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'QUEUED'"
            ).fetchone()[0]
        )
        unknown_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'SUBMISSION_UNKNOWN'"
            ).fetchone()[0]
        )

    return {
        "unique_sequences_in_request": len(unique_sequences),
        "cached_or_already_recorded": reused,
        "queued": pending_count,
        "submitted_now": len(submitted_job_ids),
        "job_ids": submitted_job_ids,
        "submission_unknown": unknown_count,
    }


def _dispatch_queued(
    db_path: str | Path,
    config: InterProConfig,
    client: InterProScanClient,
) -> list[str]:
    """Submit at most one batch, and only when no previous batch is outstanding."""
    initialize_database(db_path)
    with _database(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if _active_job_count(connection):
            return []
        rows = connection.execute(
            """
            SELECT j.id, j.sequence_hash, j.email, s.protein_sequence
            FROM jobs AS j
            JOIN sequences AS s USING (sequence_hash)
            WHERE j.status = 'QUEUED'
            ORDER BY j.id
            LIMIT ?
            """,
            (config.max_active_jobs,),
        ).fetchall()
        for row in rows:
            connection.execute(
                """
                UPDATE jobs SET status = 'SUBMITTING', updated_at = ?, error = NULL
                WHERE id = ? AND status = 'QUEUED'
                """,
                (_utc_now(), row["id"]),
            )

    queued_jobs = []
    with _database(db_path) as connection:
        for row in rows:
            isoforms = connection.execute(
                """
                SELECT gene_name, isoform_rank
                FROM isoforms
                WHERE sequence_hash = ?
                ORDER BY gene_name, isoform_rank
                """,
                (row["sequence_hash"],),
            ).fetchall()
            sequence_name, job_name = _interpro_job_names(
                isoforms, row["sequence_hash"]
            )
            queued_jobs.append((row, sequence_name, job_name))

    submitted: list[str] = []
    for index, (row, sequence_name, job_name) in enumerate(queued_jobs):
        try:
            job_id = client.submit(
                row["protein_sequence"],
                row["sequence_hash"],
                row["email"],
                title=job_name,
                sequence_name=sequence_name,
            )
        except InterProAPIError as error:
            if error.submission_uncertain:
                status = "SUBMISSION_UNKNOWN"
            else:
                status = "FAILED"
            with _database(db_path) as connection:
                connection.execute(
                    """
                    UPDATE jobs SET status = ?, updated_at = ?, completed_at = ?,
                        error = ? WHERE id = ?
                    """,
                    (
                        status,
                        _utc_now(),
                        _utc_now() if status == "FAILED" else None,
                        str(error),
                        row["id"],
                    ),
                )
            LOGGER.error(
                "Sequence %s (%s) submission status is %s: %s",
                sequence_name,
                job_name,
                status,
                error,
            )
            if error.submission_uncertain:
                break
            continue

        submitted_at = _utc_now()
        with _database(db_path) as connection:
            connection.execute(
                """
                UPDATE jobs SET job_id = ?, status = 'RUNNING',
                    submitted_at = ?, updated_at = ?, error = NULL
                WHERE id = ?
                """,
                (job_id, submitted_at, submitted_at, row["id"]),
            )
        submitted.append(job_id)
        LOGGER.info(
            "Submitted sequence %s as job %s (InterProScan ID: %s)",
            sequence_name,
            job_name,
            job_id,
        )
        if config.submission_delay_seconds and index + 1 < len(rows):
            time.sleep(config.submission_delay_seconds)

    return submitted


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _result_sequence_identifier(result: Mapping[str, Any]) -> str | None:
    xrefs = result.get("xref") or result.get("xrefs") or []
    if isinstance(xrefs, dict):
        xrefs = [xrefs]
    if isinstance(xrefs, list):
        for xref in xrefs:
            if not isinstance(xref, dict):
                continue
            name = str(xref.get("name", "")).casefold()
            if name in {"sequenceid", "sequence identifier", "id"} and xref.get("id"):
                return str(xref["id"])
        for xref in xrefs:
            if isinstance(xref, dict) and xref.get("id"):
                return str(xref["id"])
    sequence = result.get("sequence")
    if isinstance(sequence, dict):
        return sequence.get("id") or sequence.get("identifier")
    return None


def parse_interproscan_json(raw_result: str) -> list[dict[str, Any]]:
    """Flatten InterProScan JSON matches while retaining each raw match/location."""
    payload = json.loads(raw_result)
    result_items = payload.get("results", []) if isinstance(payload, dict) else payload
    if isinstance(result_items, dict):
        result_items = [result_items]
    if not isinstance(result_items, list):
        raise ValueError("InterProScan JSON has no results list")

    annotations: list[dict[str, Any]] = []
    for result in result_items:
        if not isinstance(result, dict):
            continue
        sequence_identifier = _result_sequence_identifier(result)
        matches = result.get("matches") or []
        if isinstance(matches, dict):
            matches = [matches]
        for match_index, match in enumerate(matches):
            if not isinstance(match, dict):
                continue
            signature = match.get("signature") or {}
            entry = match.get("entry") or {}
            library_release = signature.get("signatureLibraryRelease") or {}
            go_terms = match.get("goXRefs") or entry.get("goXRefs") or match.get("goTerms") or []
            pathways = (
                match.get("pathwayXRefs")
                or entry.get("pathwayXRefs")
                or match.get("pathways")
                or []
            )
            locations = match.get("locations") or []
            if isinstance(locations, dict):
                locations = [locations]
            if not locations:
                locations = [{}]

            for location_index, location in enumerate(locations):
                if not isinstance(location, dict):
                    location = {}
                fragments = location.get("fragments") or []
                start = _as_int(location.get("start"))
                end = _as_int(location.get("end"))
                if fragments:
                    fragment_starts = [
                        _as_int(fragment.get("start"))
                        for fragment in fragments
                        if isinstance(fragment, dict)
                    ]
                    fragment_ends = [
                        _as_int(fragment.get("end"))
                        for fragment in fragments
                        if isinstance(fragment, dict)
                    ]
                    valid_starts = [value for value in fragment_starts if value is not None]
                    valid_ends = [value for value in fragment_ends if value is not None]
                    if start is None and valid_starts:
                        start = min(valid_starts)
                    if end is None and valid_ends:
                        end = max(valid_ends)

                annotations.append(
                    {
                        "match_index": match_index,
                        "location_index": location_index,
                        "sequence_identifier": sequence_identifier,
                        "interpro_accession": entry.get("accession"),
                        "interpro_description": entry.get("description") or entry.get("name"),
                        "entry_type": entry.get("type"),
                        "member_database": (
                            library_release.get("library")
                            or signature.get("library")
                            or signature.get("database")
                        ),
                        "signature_accession": signature.get("accession"),
                        "signature_description": (
                            signature.get("description") or signature.get("name")
                        ),
                        "start": start,
                        "end": end,
                        "score": _string_or_none(location.get("score") or match.get("score")),
                        "evalue": _string_or_none(location.get("evalue") or match.get("evalue")),
                        "go_terms_json": _json_text(go_terms),
                        "pathways_json": _json_text(pathways),
                        "raw_match_json": _json_text(match),
                        "raw_location_json": _json_text(location),
                    }
                )
    return annotations


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _parse_interproscan_tsv(raw_result: str) -> list[dict[str, Any]]:
    """Parse the documented tabular result as a fallback; the raw file is kept too."""
    annotations: list[dict[str, Any]] = []
    for row_number, fields in enumerate(csv.reader(raw_result.splitlines(), delimiter="\t")):
        if not fields or not any(fields):
            continue
        if len(fields) < 8:
            raise ValueError(f"Malformed InterProScan TSV row {row_number + 1}")
        fields += [""] * max(0, 15 - len(fields))
        go_terms = [term for term in fields[13].split("|") if term]
        pathways = [term for term in fields[14].split("|") if term]
        annotations.append(
            {
                "match_index": row_number,
                "location_index": 0,
                "sequence_identifier": fields[0] or None,
                "interpro_accession": fields[11] or None,
                "interpro_description": fields[12] or None,
                "entry_type": None,
                "member_database": fields[3] or None,
                "signature_accession": fields[4] or None,
                "signature_description": fields[5] or None,
                "start": _as_int(fields[6]),
                "end": _as_int(fields[7]),
                "score": fields[8] or None,
                "evalue": None,
                "go_terms_json": _json_text(go_terms),
                "pathways_json": _json_text(pathways),
                "raw_match_json": _json_text({"tsv_row": fields}),
                "raw_location_json": _json_text({"start": fields[6], "end": fields[7]}),
            }
        )
    return annotations


def _parse_interproscan_xml(raw_result: str) -> list[dict[str, Any]]:
    """Extract signature, InterPro, location, GO, and pathway fields from XML."""
    root = ET.fromstring(raw_result)

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1].casefold()

    annotations: list[dict[str, Any]] = []
    proteins = [element for element in root.iter() if local_name(element) == "protein"]
    if not proteins:
        proteins = [root]
    for protein in proteins:
        xref = next(
            (element for element in protein.iter() if local_name(element) == "xref"),
            None,
        )
        sequence_identifier = xref.get("id") if xref is not None else None
        matches = [
            element
            for element in protein.iter()
            if (
                local_name(element) == "match"
                or local_name(element).endswith("-match")
            )
            and any(local_name(child) == "signature" for child in element)
        ]
        for match_index, match in enumerate(matches):
            signature = next(
                child for child in match if local_name(child) == "signature"
            )
            entry = next(
                (child for child in match if local_name(child) == "entry"),
                None,
            )
            release = next(
                (
                    element
                    for element in signature.iter()
                    if local_name(element) == "signature-library-release"
                ),
                None,
            )
            locations = [
                element
                for element in match.iter()
                if local_name(element).endswith("location")
                and "start" in element.attrib
                and "end" in element.attrib
            ] or [match]
            go_terms = [
                element.attrib
                for element in (entry.iter() if entry is not None else [])
                if "go" in local_name(element) and "xref" in local_name(element)
            ]
            pathways = [
                element.attrib
                for element in (entry.iter() if entry is not None else [])
                if "pathway" in local_name(element) and "xref" in local_name(element)
            ]

            for location_index, location in enumerate(locations):
                annotations.append(
                    {
                        "match_index": match_index,
                        "location_index": location_index,
                        "sequence_identifier": sequence_identifier,
                        "interpro_accession": entry.get("ac") if entry is not None else None,
                        "interpro_description": (
                            entry.get("desc") or entry.get("name")
                            if entry is not None
                            else None
                        ),
                        "entry_type": entry.get("type") if entry is not None else None,
                        "member_database": (
                            release.get("library") if release is not None else None
                        ),
                        "signature_accession": signature.get("ac"),
                        "signature_description": (
                            signature.get("desc") or signature.get("name")
                        ),
                        "start": _as_int(location.get("start")),
                        "end": _as_int(location.get("end")),
                        "score": location.get("score"),
                        "evalue": location.get("evalue"),
                        "go_terms_json": _json_text(go_terms),
                        "pathways_json": _json_text(pathways),
                        "raw_match_json": ET.tostring(match, encoding="unicode"),
                        "raw_location_json": ET.tostring(location, encoding="unicode"),
                    }
                )
    return annotations


def _parse_result(raw_result: str, result_type: str) -> list[dict[str, Any]]:
    normalized_type = result_type.casefold()
    if normalized_type == "json":
        return parse_interproscan_json(raw_result)
    if normalized_type in {"xml", "interproscan-xml"}:
        return _parse_interproscan_xml(raw_result)
    if normalized_type == "tsv":
        return _parse_interproscan_tsv(raw_result)
    raise ValueError(f"Unsupported InterProScan result type: {result_type}")


def _store_result(
    db_path: str | Path,
    job: sqlite3.Row,
    result_type: str,
    raw_result: str,
    annotations: list[dict[str, Any]],
) -> None:
    now = _utc_now()
    with _database(db_path) as connection:
        connection.execute("DELETE FROM annotations WHERE job_pk = ?", (job["id"],))
        for annotation in annotations:
            connection.execute(
                """
                INSERT INTO annotations(
                    job_pk, sequence_hash, match_index, location_index,
                    sequence_identifier, interpro_accession, interpro_description,
                    entry_type, member_database, signature_accession,
                    signature_description, start, end, score, evalue,
                    go_terms_json, pathways_json, raw_match_json, raw_location_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    job["id"],
                    job["sequence_hash"],
                    annotation["match_index"],
                    annotation["location_index"],
                    annotation["sequence_identifier"],
                    annotation["interpro_accession"],
                    annotation["interpro_description"],
                    annotation["entry_type"],
                    annotation["member_database"],
                    annotation["signature_accession"],
                    annotation["signature_description"],
                    annotation["start"],
                    annotation["end"],
                    annotation["score"],
                    annotation["evalue"],
                    annotation["go_terms_json"],
                    annotation["pathways_json"],
                    annotation["raw_match_json"],
                    annotation["raw_location_json"],
                ),
            )
        connection.execute(
            """
            UPDATE jobs SET status = 'FINISHED', result_type = ?, raw_result = ?,
                updated_at = ?, completed_at = ?, last_checked_at = ?, error = NULL
            WHERE id = ?
            """,
            (result_type, raw_result, now, now, now, job["id"]),
        )


def retrieve_results(
    job_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    config: InterProConfig | None = None,
    client: InterProScanClient | None = None,
    _status_verified: bool = False,
) -> int:
    """Download, parse, and persist a completed job's best available result.

    JSON is preferred because it retains nested InterPro, member-database,
    location, GO, and pathway annotations. XML and TSV are supported fallbacks;
    the complete downloaded response is always saved in the SQLite job row.
    """
    initialize_database(db_path)
    cfg = config or InterProConfig()
    api = client or InterProScanClient(cfg)
    with _database(db_path) as connection:
        job = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ? ORDER BY id DESC LIMIT 1",
            (job_id,),
        ).fetchone()
    if job is None:
        raise ValueError(f"Job ID {job_id!r} is not recorded in {db_path}")
    if job["status"] == "FINISHED" and job["raw_result"] is not None:
        with _database(db_path) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM annotations WHERE job_pk = ?",
                    (job["id"],),
                ).fetchone()[0]
            )

    try:
        if not _status_verified and api.get_status(job_id) != "FINISHED":
            raise ValueError(f"InterProScan job {job_id} is not finished yet")
        result_types = api.get_result_types(job_id)
        available = {result_type.casefold(): result_type for result_type in result_types}
        result_type = next(
            (available[name] for name in ("json", "xml", "interproscan-xml", "tsv") if name in available),
            None,
        )
        if result_type is None:
            raise ValueError(
                f"InterProScan job {job_id} has no supported JSON, XML, or TSV result type "
                f"(available: {', '.join(result_types) or 'none'})"
            )
        raw_result = api.get_result(job_id, result_type)
        annotations = _parse_result(raw_result, result_type)
    except (InterProAPIError, ValueError, ET.ParseError, json.JSONDecodeError) as error:
        with _database(db_path) as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'RESULT_RETRIEVAL_FAILED',
                    updated_at = ?, last_checked_at = ?, error = ?
                WHERE id = ?
                """,
                (_utc_now(), _utc_now(), str(error), job["id"]),
            )
        raise

    _store_result(db_path, job, result_type, raw_result, annotations)
    LOGGER.info("Stored %d annotations for InterProScan job %s", len(annotations), job_id)
    return len(annotations)


def _is_due(last_checked_at: str | None, interval_seconds: float) -> bool:
    if not last_checked_at:
        return True
    try:
        checked_at = datetime.fromisoformat(last_checked_at)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - checked_at >= timedelta(seconds=interval_seconds)


def _status_counts(db_path: str | Path) -> dict[str, int]:
    with _database(db_path) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
        ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def _print_job_counts(counts: Mapping[str, int]) -> None:
    if counts:
        print(
            "Job counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
    else:
        print("No InterProScan jobs are recorded.")


def _watch_for_stop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            response = input(
                "Press Enter on a blank line to stop watching; "
                "type anything to keep waiting: "
            )
        except EOFError:
            stop_event.set()
            return
        if not response.strip():
            stop_event.set()
            return


def check_jobs(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    config: InterProConfig | None = None,
    client: InterProScanClient | None = None,
    watch: bool = False,
) -> dict[str, int]:
    """Check recorded jobs, retrieve completed results, and advance queued batches.

    With ``watch=False`` this makes one status pass and returns. ``watch=True``
    prints counts on each pass and checks until jobs finish or, in a terminal,
    a blank line is entered at the stop prompt.
    """
    cfg = config or InterProConfig()
    api = client or InterProScanClient(cfg)
    initialize_database(db_path)
    stop_event = threading.Event() if watch else None
    stop_prompt_started = False

    while True:
        stale_before = datetime.now(timezone.utc) - timedelta(
            seconds=cfg.stale_submission_seconds
        )
        with _database(db_path) as connection:
            stale = connection.execute(
                """
                SELECT id FROM jobs
                WHERE status = 'SUBMITTING' AND updated_at < ?
                """,
                (stale_before.isoformat(timespec="seconds"),),
            ).fetchall()
            for row in stale:
                connection.execute(
                    """
                    UPDATE jobs SET status = 'SUBMISSION_UNKNOWN',
                        updated_at = ?, error = ?
                    WHERE id = ?
                    """,
                    (
                        _utc_now(),
                        "Process ended before the API response was saved; submission may have succeeded.",
                        row["id"],
                    ),
                )
            jobs = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('RUNNING', 'UNKNOWN_STATUS', 'RESULT_RETRIEVAL_FAILED')
                    AND job_id IS NOT NULL
                ORDER BY id
                """
            ).fetchall()

        for job in jobs:
            if not _is_due(job["last_checked_at"], cfg.poll_interval_seconds):
                continue
            job_id = str(job["job_id"])
            if job["status"] == "RESULT_RETRIEVAL_FAILED":
                try:
                    retrieve_results(
                        job_id,
                        db_path=db_path,
                        config=cfg,
                        client=api,
                        _status_verified=True,
                    )
                except (InterProAPIError, ValueError, ET.ParseError, json.JSONDecodeError) as error:
                    LOGGER.warning("Could not retrieve results for job %s: %s", job_id, error)
                continue

            try:
                remote_status = api.get_status(job_id)
            except InterProAPIError as error:
                now = _utc_now()
                status = "EXPIRED" if error.status_code == 404 else job["status"]
                with _database(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET status = ?, last_checked_at = ?, updated_at = ?,
                            completed_at = ?, error = ? WHERE id = ?
                        """,
                        (
                            status,
                            now,
                            now,
                            now if status == "EXPIRED" else None,
                            str(error),
                            job["id"],
                        ),
                    )
                LOGGER.warning("Could not check InterProScan job %s: %s", job_id, error)
                continue

            now = _utc_now()
            if remote_status in {"QUEUED", "RUNNING"}:
                with _database(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'RUNNING', last_checked_at = ?,
                            updated_at = ?, error = NULL WHERE id = ?
                        """,
                        (now, now, job["id"]),
                    )
            elif remote_status == "FINISHED":
                with _database(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET last_checked_at = ?, updated_at = ?, error = NULL
                        WHERE id = ?
                        """,
                        (now, now, job["id"]),
                    )
                try:
                    retrieve_results(
                        job_id,
                        db_path=db_path,
                        config=cfg,
                        client=api,
                        _status_verified=True,
                    )
                except (InterProAPIError, ValueError, ET.ParseError, json.JSONDecodeError) as error:
                    LOGGER.warning("Could not retrieve results for job %s: %s", job_id, error)
            elif remote_status in {"FAILURE", "ERROR", "FAILED"}:
                with _database(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'FAILED', last_checked_at = ?,
                            completed_at = ?, updated_at = ?, error = ?
                        WHERE id = ?
                        """,
                        (now, now, now, f"InterProScan status: {remote_status}", job["id"]),
                    )
                LOGGER.error("InterProScan job %s ended with status %s", job_id, remote_status)
            elif remote_status in {"NOT_FOUND", "NOTFOUND"}:
                with _database(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'EXPIRED', last_checked_at = ?,
                            completed_at = ?, updated_at = ?, error = ?
                        WHERE id = ?
                        """,
                        (
                            now,
                            now,
                            now,
                            "Job is no longer available from the remote service.",
                            job["id"],
                        ),
                    )
                LOGGER.error("InterProScan job %s is no longer available", job_id)
            else:
                with _database(db_path) as connection:
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'UNKNOWN_STATUS', last_checked_at = ?,
                            updated_at = ?, error = ? WHERE id = ?
                        """,
                        (now, now, f"Unrecognized API status: {remote_status}", job["id"]),
                    )

        _dispatch_queued(db_path, cfg, api)
        counts = _status_counts(db_path)
        if not watch:
            return counts
        _print_job_counts(counts)

        with _database(db_path) as connection:
            checkable = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                    WHERE status IN ('RUNNING', 'UNKNOWN_STATUS', 'RESULT_RETRIEVAL_FAILED')
                        AND job_id IS NOT NULL
                    """
                ).fetchone()[0]
            )
            queued = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'QUEUED'"
                ).fetchone()[0]
            )
            unknown_submissions = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('SUBMISSION_UNKNOWN', 'SUBMITTING')"
                ).fetchone()[0]
            )
        if checkable == 0 and queued == 0:
            if unknown_submissions:
                LOGGER.error(
                    "%d submission(s) have no confirmed job ID; inspect the database before retrying.",
                    unknown_submissions,
                )
            stop_event.set()
            return counts
        if not stop_prompt_started:
            if sys.stdin.isatty():
                threading.Thread(
                    target=_watch_for_stop,
                    args=(stop_event,),
                    daemon=True,
                ).start()
            stop_prompt_started = True
        LOGGER.info(
            "Waiting %d seconds before the next InterProScan status check",
            int(cfg.poll_interval_seconds),
        )
        if stop_event.wait(cfg.poll_interval_seconds):
            return counts


def load_annotations(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    gene_name: str | None = None,
) -> pd.DataFrame:
    """Load the latest successful annotations per unique sequence as a DataFrame.

    Duplicate sequences are expanded back to every mapped gene/rank/transcript.
    The sequence and job names reflect all isoforms currently mapped to each
    unique protein sequence. GO and pathway values are returned as Python lists;
    raw JSON columns remain available for complete match and location details.
    """
    initialize_database(db_path)
    query = """
        SELECT
            i.gene_name, i.isoform_rank, i.transcript_id, i.isoform_id,
            s.sequence_hash, s.protein_sequence, j.job_id, j.result_type,
            a.sequence_identifier, a.interpro_accession, a.interpro_description,
            a.entry_type, a.member_database, a.signature_accession,
            a.signature_description, a.start, a.end, a.score, a.evalue,
            a.go_terms_json, a.pathways_json, a.raw_match_json, a.raw_location_json
        FROM annotations AS a
        JOIN jobs AS j ON j.id = a.job_pk
        JOIN sequences AS s ON s.sequence_hash = a.sequence_hash
        JOIN isoforms AS i ON i.sequence_hash = a.sequence_hash
        JOIN (
            SELECT sequence_hash, MAX(id) AS job_pk
            FROM jobs
            WHERE status = 'FINISHED'
            GROUP BY sequence_hash
        ) AS latest ON latest.job_pk = j.id
        ORDER BY i.gene_name, i.isoform_rank, a.match_index, a.location_index
    """
    with _database(db_path) as connection:
        annotations = pd.read_sql_query(query, connection)
        names_by_hash = _sequence_names_by_hash(connection)
    names = annotations["sequence_hash"].map(
        lambda sequence_hash: names_by_hash.get(
            sequence_hash, _interpro_job_names([], str(sequence_hash))
        )
    )
    annotations["sequence_name"] = names.map(lambda pair: pair[0])
    annotations["job_name"] = names.map(lambda pair: pair[1])
    if gene_name is not None:
        annotations = annotations[
            annotations["gene_name"].astype(str).str.casefold()
            == str(gene_name).casefold()
        ].reset_index(drop=True)
    for json_column, list_column in (
        ("go_terms_json", "go_terms"),
        ("pathways_json", "pathways"),
    ):
        if json_column in annotations.columns:
            annotations[list_column] = annotations[json_column].map(
                lambda value: json.loads(value) if value else []
            )
    return annotations


def load_completed_isoforms(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    gene_name: str | None = None,
) -> pd.DataFrame:
    """Load isoform mappings whose unique protein sequence has a FINISHED job.

    Unlike :func:`load_annotations`, this also returns completed sequences with
    no annotation rows.
    """
    initialize_database(db_path)
    gene_filter = ""
    parameters: tuple[str, ...] = ()
    if gene_name is not None:
        gene_filter = "WHERE LOWER(i.gene_name) = LOWER(?)"
        parameters = (gene_name,)

    with _database(db_path) as connection:
        return pd.read_sql_query(
            f"""
            SELECT
                i.gene_name, i.isoform_rank, i.transcript_id, i.isoform_id,
                s.sequence_hash, s.protein_sequence, j.job_id, j.completed_at
            FROM isoforms AS i
            JOIN sequences AS s ON s.sequence_hash = i.sequence_hash
            JOIN (
                SELECT sequence_hash, MAX(id) AS job_pk
                FROM jobs
                WHERE status = 'FINISHED'
                GROUP BY sequence_hash
            ) AS latest ON latest.sequence_hash = i.sequence_hash
            JOIN jobs AS j ON j.id = latest.job_pk
            {gene_filter}
            ORDER BY i.gene_name, i.isoform_rank, i.transcript_id
            """,
            connection,
            params=parameters,
        )


def load_jobs(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    include_raw_results: bool = False,
) -> pd.DataFrame:
    """Load persistent job state, optionally including complete raw API results."""
    initialize_database(db_path)
    columns = """
        j.id, j.job_id, j.sequence_hash, s.protein_sequence, j.status, j.attempt,
        j.created_at, j.submitted_at, j.last_checked_at, j.completed_at, j.error,
        j.result_type
    """
    if include_raw_results:
        columns += ", j.raw_result"
    with _database(db_path) as connection:
        jobs = pd.read_sql_query(
            f"""
            SELECT {columns}
            FROM jobs AS j JOIN sequences AS s USING (sequence_hash)
            ORDER BY j.id
            """,
            connection,
        )
        names_by_hash = _sequence_names_by_hash(connection)
    names = jobs["sequence_hash"].map(
        lambda sequence_hash: names_by_hash.get(
            sequence_hash, _interpro_job_names([], str(sequence_hash))
        )
    )
    jobs["sequence_name"] = names.map(lambda pair: pair[0])
    jobs["job_name"] = names.map(lambda pair: pair[1])
    return jobs


def _pipeline_proteins(
    gene_name: str,
    isoform_numbers: list[int],
    matrix_path: Path,
    gtf_path: Path,
    cds_path: Path,
    genome_path: Path | None,
) -> dict[int, tuple[str, str]]:
    return get_ranked_isoform_proteins(
        gene_name,
        isoform_numbers,
        matrix_path=matrix_path,
        gtf_path=gtf_path,
        cds_fasta_path=cds_path,
        genome_fasta_path=genome_path,
    )


def _config_from_args(args: argparse.Namespace) -> InterProConfig:
    return InterProConfig(
        max_active_jobs=args.max_active_jobs,
        submission_delay_seconds=args.submission_delay,
        poll_interval_seconds=args.poll_interval,
        request_timeout_seconds=args.request_timeout,
    )


def _interactive_input(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return None


def _run_interactive() -> int:
    print("InterProScan workflow: submit sequences, check jobs, or export annotations.")
    while True:
        command = _interactive_input("Action [submit/check/export, or q to quit]: ")
        if command is None or command.casefold() in {"", "q", "quit", "exit"}:
            return 0
        command = command.casefold()
        if command not in {"submit", "check", "export"}:
            print("Choose submit, check, export, or q to quit.")
            continue

        command_args = [command]
        if command == "submit":
            gene_name = _interactive_input("Gene name (blank to cancel): ")
            if gene_name is None:
                return 0
            if not gene_name:
                continue

            ranks_text = _interactive_input("Isoform ranks (e.g. 1 2 3 or 1,2,3; blank to cancel): ")
            if ranks_text is None:
                return 0
            if not ranks_text:
                continue
            try:
                ranks = [
                    int(value)
                    for value in re.split(r"[,\s]+", ranks_text)
                    if value
                ]
            except ValueError:
                print("Enter positive integer ranks, separated by spaces or commas.")
                continue
            if not ranks or any(rank < 1 for rank in ranks):
                print("Isoform ranks must be positive integers.")
                continue
            if len(set(ranks)) != len(ranks):
                print("Do not enter duplicate isoform ranks.")
                continue

            email_default = os.environ.get("INTERPROSCAN_EMAIL", "")
            email_prompt = (
                f"Contact email [{email_default}]: "
                if email_default
                else "Contact email: "
            )
            email = _interactive_input(email_prompt)
            if email is None:
                return 0
            email = email or email_default
            if not email:
                print("A contact email is required to submit sequences.")
                continue

            retry = _interactive_input("Retry previously failed jobs? [y/N]: ")
            if retry is None:
                return 0
            if retry.casefold() not in {"", "y", "yes", "n", "no"}:
                print("Answer yes or no.")
                continue

            command_args.extend(
                [gene_name, *(str(rank) for rank in ranks), "--email", email]
            )
            if retry.casefold() in {"y", "yes"}:
                command_args.append("--retry-failed")
        elif command == "check":
            watch = _interactive_input("Wait and keep checking until jobs finish? [y/N]: ")
            if watch is None:
                return 0
            if watch.casefold() not in {"", "y", "yes", "n", "no"}:
                print("Answer yes or no.")
                continue
            if watch.casefold() in {"y", "yes"}:
                command_args.append("--watch")
        else:
            gene_name = _interactive_input("Gene filter (blank for all genes): ")
            if gene_name is None:
                return 0
            if gene_name:
                command_args.extend(["--gene", gene_name])

        result = main(command_args)
        if result:
            return result


def _export_output_path(gene_name: str | None) -> Path:
    if gene_name:
        gene_label = re.sub(r"[^A-Za-z0-9._-]+", "_", gene_name).strip("._-")
        gene_label = gene_label or "gene"
    else:
        gene_label = "all_genes"

    return EXPORT_FOLDER / f"interproscan_annotations_{gene_label}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit pipeline protein isoforms to InterProScan and retrieve results later."
    )
    subparsers = parser.add_subparsers(dest="command")

    submit_parser = subparsers.add_parser(
        "submit", help="Identify selected isoforms and submit new sequences"
    )
    submit_parser.add_argument("gene_name")
    submit_parser.add_argument("isoform_numbers", nargs="+", type=int)
    submit_parser.add_argument("--email", default=os.environ.get("INTERPROSCAN_EMAIL"))
    submit_parser.add_argument("--retry-failed", action="store_true")
    submit_parser.add_argument("--matrix", type=Path, default=ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms_matrix.tsv")
    submit_parser.add_argument("--gtf", type=Path, default=ROOT_FOLDER / "Long_Read_Data" / "expressed_isoforms.gtf")
    submit_parser.add_argument("--cds", type=Path, default=ROOT_FOLDER / "Long_Read_Data" / "cds_output.fa")
    submit_parser.add_argument("--genome", type=Path)
    submit_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    submit_parser.add_argument("--max-active-jobs", type=int, default=API_MAX_JOBS)
    submit_parser.add_argument("--submission-delay", type=float, default=1.0)
    submit_parser.add_argument("--poll-interval", type=float, default=300.0)
    submit_parser.add_argument("--request-timeout", type=float, default=60.0)

    check_parser = subparsers.add_parser(
        "check", help="Check saved job IDs and retrieve completed results"
    )
    check_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    check_parser.add_argument("--watch", action="store_true")
    check_parser.add_argument("--max-active-jobs", type=int, default=API_MAX_JOBS)
    check_parser.add_argument("--submission-delay", type=float, default=1.0)
    check_parser.add_argument("--poll-interval", type=float, default=300.0)
    check_parser.add_argument("--request-timeout", type=float, default=60.0)

    export_parser = subparsers.add_parser(
        "export", help="Export annotations to the project's CSV folder"
    )
    export_parser.add_argument("--gene")
    export_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command is None:
        return _run_interactive()

    if args.command == "submit":
        if not args.email:
            parser.error("Provide --email or set the INTERPROSCAN_EMAIL environment variable.")
        if not args.matrix.is_file():
            raise FileNotFoundError(f"Expression matrix not found: {args.matrix}")
        if not args.gtf.is_file():
            raise FileNotFoundError(f"GTF not found: {args.gtf}")
        config = _config_from_args(args)
        proteins = _pipeline_proteins(
            args.gene_name,
            args.isoform_numbers,
            args.matrix,
            args.gtf,
            args.cds,
            args.genome,
        )
        report = submit_sequences(
            args.gene_name,
            proteins,
            email=args.email,
            db_path=args.db,
            config=config,
            retry_failed=args.retry_failed,
        )
        print(
            f"Unique sequences: {report['unique_sequences_in_request']}; "
            f"submitted now: {report['submitted_now']}; queued: {report['queued']}; "
            f"cached/already recorded: {report['cached_or_already_recorded']}."
        )
        print(f"Job database: {args.db}")
        return 0

    if args.command == "check":
        counts = check_jobs(
            db_path=args.db,
            config=_config_from_args(args),
            watch=args.watch,
        )
        if not args.watch:
            _print_job_counts(counts)
        print(f"Job database: {args.db}")
        return 0

    gene_filter = args.gene.strip() if args.gene and args.gene.strip() else None
    annotations = load_annotations(args.db, gene_name=gene_filter)
    output = _export_output_path(gene_filter)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(output, index=False)
    print(f"Exported {len(annotations)} annotation rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())