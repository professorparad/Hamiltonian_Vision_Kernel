"""Cross-check every hardware job the manuscripts cite against the IBM Quantum account.

This closes the account-side half of item F2 in ``overleaf_docs/todo.md``: the job IDs,
backends, shot counts and PSNRs printed in the manuscripts are all readable from local
JSON, but "the artifact says so" is not provenance. This script asks the service.

    python IBM_Cloud/verify_hardware_jobs.py                # report only
    python IBM_Cloud/verify_hardware_jobs.py --write-map    # also fill in RESULTS_MAP.md

For every job it retrieves ``backend``, ``status``, ``creation_date`` and the shot count
recorded in the submitted primitive input, and compares backend + shots against what the
manuscript claims (read from the retained artifacts, not from the .tex). The account's
instance/CRN is reported once at the top so it can be recorded in the map.

Requires a configured account -- either a saved one::

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...")

or ``IBM_QUANTUM_TOKEN`` in the environment. Read-only: it retrieves jobs, never submits.

The IonQ replay jobs in the supplement's Table 15 ran on IonQ's cloud, not IBM's, so they
are listed as ``skipped (IonQ)`` -- they are not retrievable through this service and
must be checked in the IonQ console if provenance for them is wanted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_MAP = REPO_ROOT / "overleaf_docs" / "RESULTS_MAP.md"
SUPPLEMENT = REPO_ROOT / "overleaf_docs" / "supplementary_study.tex"

PILOT_MONALISA = (
    REPO_ROOT / "IBM_Cloud/outputs/hardware_reconstruction/hardware_reconstruction_report.json"
)
PILOT_CIFAR = (
    REPO_ROOT / "IBM_Cloud/outputs/hvk2d_cifar_hardware_reconstruction/summary.json"
)
ANCHORS = (
    REPO_ROOT / "IBM_Cloud/outputs/hardware_robustness_study/real_hardware_anchors.json"
)


def _walk(node):
    """Yield every dict nested anywhere inside ``node``."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def expected_jobs() -> list[dict]:
    """Job records the manuscripts rely on, read from the retained artifacts."""
    jobs: list[dict] = []
    seen: set[str] = set()

    def add(record: dict, campaign: str, default_shots: int | None = None) -> None:
        job_id = record.get("job_id")
        if not job_id or job_id in seen:
            return
        seen.add(job_id)
        jobs.append(
            {
                "job_id": job_id,
                "campaign": campaign,
                "backend": record.get("backend"),
                "shots": record.get("shots", default_shots),
                "psnr_db": record.get("psnr_hardware_db", record.get("psnr")),
                "image": record.get("image_name") or record.get("image"),
            }
        )

    for path, campaign, shots in (
        (PILOT_MONALISA, "reconstruction pilot", 256),
        (PILOT_CIFAR, "reconstruction pilot", 256),
        (ANCHORS, "repeated-execution anchor", None),
    ):
        if not path.exists():
            print(f"  ! missing artifact, skipped: {path.relative_to(REPO_ROOT)}")
            continue
        for record in _walk(json.loads(path.read_text(encoding="utf-8"))):
            add(record, campaign, shots)

    jobs.extend(_replay_ledger_jobs(seen))
    return jobs


def _replay_ledger_jobs(seen: set[str]) -> list[dict]:
    """The archived replay campaigns, read out of the supplement's ledger table."""
    if not SUPPLEMENT.exists():
        return []
    text = SUPPLEMENT.read_text(encoding="utf-8")
    marker = "label{tab:replay_job_ledger}"
    if marker not in text:
        return []
    table = text[text.index(marker) :]
    table = table[: table.index(r"\end{tabular}")]
    out = []
    for line in table.splitlines():
        if "&" not in line or "texttt" not in line:
            continue
        cells = [
            re.sub(r"\\texttt\{|\}|\\\\", "", cell).replace("\\_", "_").strip()
            for cell in line.split("&")
        ]
        if len(cells) < 5:
            continue
        backend, shots, job_id = cells[2], cells[3], cells[4].strip(" \\")
        if job_id in seen:
            continue
        seen.add(job_id)
        out.append(
            {
                "job_id": job_id,
                "campaign": "replay ledger",
                "backend": backend,
                "shots": int(shots) if shots.isdigit() else None,
                "psnr_db": None,
                "image": None,
            }
        )
    return out


def _shots_of(job) -> int | None:
    """Dig the shot count out of whatever shape this job's inputs happen to have."""
    try:
        inputs = job.inputs or {}
    except Exception:  # noqa: BLE001 - the service can refuse to return inputs
        return None
    options = inputs.get("options") or {}
    for key in ("default_shots", "shots"):
        if isinstance(options.get(key), int):
            return options[key]
    execution = options.get("execution") or {}
    if isinstance(execution.get("shots"), int):
        return execution["shots"]
    for pub in inputs.get("pubs") or []:
        if isinstance(pub, (list, tuple)) and len(pub) >= 3 and isinstance(pub[2], int):
            return pub[2]
    return None


def verify(jobs: list[dict]) -> list[dict]:
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    service = (
        QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        if token
        else QiskitRuntimeService()
    )

    account = service.active_account() or {}
    print("Account")
    print(f"  channel  : {account.get('channel')}")
    print(f"  instance : {account.get('instance')}   <- record this as the instance/CRN")
    print()

    results = []
    for job in jobs:
        row = dict(job, account="", notes=[])
        if "ionq" in (job["backend"] or "").lower():
            row["account"] = "skipped (IonQ)"
            row["notes"].append("not an IBM Quantum job; check the IonQ console")
            results.append(row)
            continue
        try:
            remote = service.job(job["job_id"])
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            row["account"] = "NOT RETRIEVABLE"
            row["notes"].append(f"{type(exc).__name__}: {exc}")
            results.append(row)
            continue

        try:
            backend = remote.backend().name
        except Exception:  # noqa: BLE001
            backend = None
        shots = _shots_of(remote)
        status = str(remote.status())
        created = getattr(remote, "creation_date", None)

        if job["backend"] and backend and backend != job["backend"]:
            row["notes"].append(f"backend mismatch: paper {job['backend']}, service {backend}")
        if job["shots"] and shots and shots != job["shots"]:
            row["notes"].append(f"shots mismatch: paper {job['shots']}, service {shots}")
        if status.upper().find("DONE") < 0:
            row["notes"].append(f"status is {status}, not DONE")

        row["account"] = "ok" if not row["notes"] else "MISMATCH"
        row["service_backend"] = backend
        row["service_shots"] = shots
        row["service_status"] = status
        row["created"] = str(created) if created else None
        results.append(row)
    return results


def report(results: list[dict]) -> int:
    width = max(len(r["job_id"]) for r in results)
    bad = 0
    for row in results:
        flag = row["account"]
        if flag == "MISMATCH" or flag == "NOT RETRIEVABLE":
            bad += 1
        detail = "; ".join(row["notes"]) if row["notes"] else ""
        print(
            f"{row['job_id']:<{width}}  {row['campaign']:<26} "
            f"{str(row['backend']):<16} {flag:<16} {detail}"
        )
    print()
    print(f"{len(results)} jobs, {bad} needing attention")
    return bad


def write_map(results: list[dict]) -> None:
    """Replace the `pending` cells in RESULTS_MAP.md with the verified outcome."""
    if not RESULTS_MAP.exists():
        print(f"! {RESULTS_MAP} not found; nothing rewritten")
        return
    text = RESULTS_MAP.read_text(encoding="utf-8")
    for row in results:
        pattern = re.compile(
            r"(`" + re.escape(row["job_id"]) + r"`[^\n|]*(?:\|[^\n|]*)*?\|\s*)pending(\s*\|)"
        )
        text, count = pattern.subn(rf"\g<1>{row['account']}\g<2>", text)
        if not count:
            print(f"  ! no `pending` cell found for {row['job_id']}")
    RESULTS_MAP.write_text(text, encoding="utf-8")
    try:
        shown = RESULTS_MAP.relative_to(REPO_ROOT)
    except ValueError:
        shown = RESULTS_MAP
    print(f"RESULTS_MAP.md updated: {shown}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-map",
        action="store_true",
        help="rewrite the Account column of overleaf_docs/RESULTS_MAP.md in place",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the jobs that would be checked and exit (no account needed)",
    )
    args = parser.parse_args()

    jobs = expected_jobs()
    if args.list:
        for job in jobs:
            print(f"{job['job_id']}  {job['campaign']:<26} {job['backend']}  shots={job['shots']}")
        print(f"\n{len(jobs)} jobs")
        return 0

    results = verify(jobs)
    bad = report(results)
    if args.write_map:
        write_map(results)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
