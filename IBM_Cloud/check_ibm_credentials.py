"""Preflight for the hardware-provenance check (``overleaf_docs/todo.md`` item H1).

``verify_hardware_jobs.py`` needs a working IBM Quantum credential, and when it has none
it fails with whatever the service happens to raise -- which does not say *which* thing
is wrong. This script asks the four questions separately, so the answer is unambiguous::

    python IBM_Cloud/check_ibm_credentials.py

  1. is a key present?          ``IBM_QUANTUM_TOKEN`` / ``IBM_CLOUD_API_KEY``, or a
                                saved account in ``~/.qiskit/qiskit-ibm.json``
  2. does IBM Cloud accept it?  the key is exchanged for an IAM token, which is where an
                                expired, deleted, or legacy quantum.ibm.com key fails
  3. what instances can it see? every CRN the key reaches -- one of these is the
                                instance/CRN line RESULTS_MAP.md's F2 ledger asks for
  4. does a job retrieve?       one IBM job id out of that ledger, end to end

Failing at 2 means the key itself is dead and no amount of instance fiddling will help:
make a new one at https://quantum.cloud.ibm.com. Passing 2 and 3 but failing 4 means the
account is fine and the *job* is not reachable -- that is a provenance finding, to be
flagged in RESULTS_MAP.md rather than papered over.

``--save`` writes the working key (and the instance, if exactly one was found) to the
qiskit account file so the verify run afterwards needs no environment variable.

Read-only against the service: it never submits a circuit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
ACCOUNT_FILE = Path.home() / ".qiskit" / "qiskit-ibm.json"
CHANNEL = "ibm_quantum_platform"

OK = "  ok   "
BAD = " FAIL  "
SKIP = " skip  "


def _mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]} ({len(key)} chars)" if len(key) > 12 else "(too short to be a key)"


def find_key() -> tuple[str | None, str]:
    """The key to test, plus a human-readable note on where it came from."""
    for name in ("IBM_QUANTUM_TOKEN", "IBM_CLOUD_API_KEY", "QISKIT_IBM_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip(), f"environment variable {name}"

    if ACCOUNT_FILE.exists():
        try:
            saved = json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"{ACCOUNT_FILE} is unreadable ({exc})"
        for name, account in saved.items():
            token = account.get("token")
            if token:
                return token.strip(), f"saved account '{name}' in {ACCOUNT_FILE}"
        return None, f"{ACCOUNT_FILE} exists but holds no token"

    return None, "nothing set, and no saved account file"


def check_iam(key: str) -> tuple[bool, str, str | None]:
    """Exchange the key for an IAM token. This is the step an expired key fails."""
    body = urllib.parse.urlencode(
        {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": key}
    ).encode()
    request = urllib.request.Request(
        IAM_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        return (
            True,
            f"IBM Cloud issued an access token (expires in {payload.get('expires_in')}s)",
            payload.get("access_token"),
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("errorMessage", detail)
        except json.JSONDecodeError:
            pass
        hint = ""
        if exc.code in (400, 401):
            hint = (
                "\n         This is the expired / revoked / wrong-platform case. A token copied"
                "\n         from the retired quantum.ibm.com dashboard also lands here: the new"
                "\n         platform wants an IBM Cloud API key. Make a fresh one at"
                "\n         https://quantum.cloud.ibm.com (or IBM Cloud > Manage > Access (IAM)"
                "\n         > API keys > Create) and re-run this script."
            )
        return False, f"HTTP {exc.code}: {detail}{hint}", None
    except urllib.error.URLError as exc:
        return (
            False,
            f"could not reach {IAM_TOKEN_URL} ({exc.reason}) -- network, not credentials",
            None,
        )


def account_context(access_token: str) -> str:
    """Which IBM Cloud account this key belongs to, and what else the IBMid can reach.

    An API key is bound to the account it was created in and cannot be switched to
    another one (IAM refuses with BXNIM0413E), so "key is fine but sees no instance"
    almost always means the key was made in the wrong account.
    """
    import base64

    lines = []
    try:
        claims_b64 = access_token.split(".")[1]
        claims_b64 += "=" * (-len(claims_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(claims_b64))
        lines.append(f"           key belongs to account {claims.get('account', {}).get('bss')}")
        lines.append(f"           IBMid {claims.get('sub')}")
    except Exception:  # noqa: BLE001 - decorative; never let it break the report
        pass

    request = urllib.request.Request(
        "https://accounts.cloud.ibm.com/v1/accounts",
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            accounts = json.loads(response.read()).get("resources", [])
    except Exception:  # noqa: BLE001
        return "\n".join(lines)

    if accounts:
        lines.append("           accounts this IBMid can reach:")
        for entry in accounts:
            guid = (entry.get("metadata") or {}).get("guid")
            body = entry.get("entity") or {}
            state = body.get("state", "?")
            mark = "" if state == "ACTIVE" else "   <- not usable"
            lines.append(f"             {body.get('name', '?')}  [{state}]  {guid}{mark}")
        lines.append(
            "           An API key only ever works in the account it was created in."
            "\n           If the jobs ran under a different account, make the key there."
        )
    return "\n".join(lines)


def check_instances(key: str, access_token: str | None = None) -> tuple[bool, str, list[str]]:
    """List every instance the key reaches; the CRN is what RESULTS_MAP.md wants."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return False, (
            "qiskit-ibm-runtime is not installed"
            " (pip install -r IBM_Cloud/requirements-ibm.txt)"
        ), []

    instances: list = []
    try:
        instances = list(QiskitRuntimeService(channel=CHANNEL, token=key).instances())
    except Exception as exc:  # noqa: BLE001 - any failure here is a report, not a crash
        # An account with no runtime instance raises rather than returning [], so the
        # "nothing to query" case arrives here as well as through an empty list.
        if "no matching instances" not in str(exc).lower():
            return False, f"{type(exc).__name__}: {exc}", []

    if not instances:
        detail = "the key authenticates but reaches no instance -- there is nothing to query.\n"
        if access_token:
            detail += account_context(access_token) + "\n"
        detail += (
            "           If that account is the right one, create an instance at"
            " https://quantum.cloud.ibm.com\n           (the Open plan is free). But note:"
            " a NEW instance cannot retrieve jobs submitted\n           under a previous"
            " one -- job history is scoped to the instance that ran them."
        )
        return False, detail, []

    crns = []
    lines = []
    for entry in instances:
        crn = entry.get("crn") or entry.get("instance") or ""
        crns.append(crn)
        lines.append(
            f"           {entry.get('name', '?')}  plan={entry.get('plan', '?')}\n           {crn}"
        )
    return True, f"{len(instances)} instance(s) visible:\n" + "\n".join(lines), crns


def check_one_job(key: str, instance: str | None) -> tuple[bool, str]:
    """Retrieve a single IBM job from the F2 ledger, end to end."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from verify_hardware_jobs import expected_jobs
    except ImportError as exc:
        return False, f"could not import the job ledger ({exc})"

    ibm_jobs = [j for j in expected_jobs() if "ionq" not in (j["backend"] or "").lower()]
    if not ibm_jobs:
        return False, "no IBM jobs found in the ledger -- check the artifacts are present"

    probe = ibm_jobs[0]
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        service = QiskitRuntimeService(channel=CHANNEL, token=key, instance=instance)
        job = service.job(probe["job_id"])
        backend = job.backend().name
        return True, (
            f"{probe['job_id']} retrieved: backend={backend}, status={job.status()}, "
            f"created={getattr(job, 'creation_date', None)}"
        )
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"{probe['job_id']} did not retrieve -- {type(exc).__name__}: {exc}\n"
            "         The account works, so this is about the job, not the key: wrong"
            " instance,\n         or the job is past the platform's retention window."
            " Record that in\n         RESULTS_MAP.md as a finding; do not overwrite the"
            " ledger values."
        )


def save_account(key: str, crns: list[str]) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    instance = crns[0] if len(crns) == 1 else None
    QiskitRuntimeService.save_account(
        channel=CHANNEL, token=key, instance=instance, overwrite=True, set_as_default=True
    )
    print(f"\nSaved to {ACCOUNT_FILE} (channel={CHANNEL}, instance={instance or 'auto'}).")
    if instance is None and len(crns) > 1:
        print("  ! several instances were visible, so none was pinned. If the verify run picks")
        print("    the wrong one, re-save with instance=<the CRN the jobs were run under>.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        action="store_true",
        help="if the key works, write it to the qiskit account file for later runs",
    )
    args = parser.parse_args()

    key, origin = find_key()
    print("1. key present")
    if not key:
        print(f"{BAD} {origin}")
        print("\n       Set one and re-run:")
        print("           export IBM_QUANTUM_TOKEN='<your IBM Cloud API key>'")
        print("           python IBM_Cloud/check_ibm_credentials.py --save")
        return 1
    print(f"{OK} {_mask(key)} from {origin}")

    print("\n2. IBM Cloud accepts the key")
    accepted, detail, access_token = check_iam(key)
    print(f"{OK if accepted else BAD} {detail}")
    if not accepted:
        return 1

    print("\n3. instances the key can see")
    reachable, detail, crns = check_instances(key, access_token)
    print(f"{OK if reachable else BAD} {detail}")
    if not reachable:
        return 1

    print("\n4. a ledger job retrieves")
    retrieved, detail = check_one_job(key, crns[0] if len(crns) == 1 else None)
    print(f"{OK if retrieved else BAD} {detail}")

    if args.save:
        save_account(key, crns)

    if retrieved:
        print("\nCredential is good. Now close item H1:")
        print("    python IBM_Cloud/verify_hardware_jobs.py --write-map")
        return 0
    print("\nThe credential is fine but the job is not reachable. Run the full sweep to see")
    print("how many of the ledger's IBM jobs are affected before deciding what to write:")
    print("    python IBM_Cloud/verify_hardware_jobs.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
