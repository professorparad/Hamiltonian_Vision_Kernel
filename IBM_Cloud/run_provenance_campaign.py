"""Re-submit the manuscripts' hardware campaign under a live IBM/IonQ account.

Why this exists. ``RESULTS_MAP.md`` §F2 records that every job identifier printed in
the two manuscripts was submitted through an IBM Cloud instance belonging to an
IIT Bhubaneswar trial account that is now ``CANCELED``. Job history is scoped to the
submitting instance, so none of those jobs can be retrieved from the service again --
not by making a new key, not by creating a new instance. The printed numbers stay
backed by retained local artifacts; what is gone, permanently, is the *second*
line of evidence, the service-side confirmation.

This runner rebuilds that second line from scratch. It re-executes the same circuits,
rebuilt from the same checkpoints, under whatever account is live now, and records
the one field the original campaign failed to serialize -- the instance CRN, which is
F2's own stated "lesson for future campaigns".

READ THIS BEFORE USING THE NUMBERS. A fresh run is not a verification of Table 3 or
Table 4. Different hardware, different calibration, different day: the PSNRs that come
back will not equal the printed ones and must never be substituted for them. What this
campaign establishes is provenance -- that these circuits, from these checkpoints, run
on real hardware under an account the student controls and can show a referee.

Nothing here writes into a retained artifact path. Every output goes under
``IBM_Cloud/outputs/provenance_campaign/``; the drivers that hardcode an output
directory are redirected by rebinding their module-level constant.

Usage::

    export IBM_QUANTUM_TOKEN='<IBM Cloud API key>'
    export IONQ_API_TOKEN='<IonQ key>'
    python IBM_Cloud/check_ibm_credentials.py --save     # once, writes the qiskit account
    python IBM_Cloud/run_provenance_campaign.py --stage preflight
    python IBM_Cloud/run_provenance_campaign.py --stage all

Stages run in their own subprocess so a failure in one does not lose the others; each
appends to ``ledger.jsonl`` as it goes, and ``--stage ledger`` compiles that into the
markdown table that RESULTS_MAP.md's F2 section wants.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_DIR = REPO_ROOT / "IBM_Cloud"
if str(IBM_DIR) not in sys.path:
    sys.path.insert(0, str(IBM_DIR))

DEFAULT_ROOT = IBM_DIR / "outputs" / "provenance_campaign"

# The backends the original campaign used, per RESULTS_MAP.md §F2. Kept as the default
# so a re-run is comparable where the account can still reach them; --ibm-backend
# overrides when it cannot.
PILOT_BACKEND = "ibm_fez"
ANCHOR_PRIMARY = "ibm_kingston"
ANCHOR_CROSS = "ibm_marrakesh"
SWEEP_BACKEND = "ibm_marrakesh"
SHOTS = [256, 512, 1024]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def account_context() -> dict:
    """Instance CRN and remaining quota -- recorded next to every job id."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    account = service.active_account() or {}
    context = {
        "channel": account.get("channel"),
        "instance": account.get("instance"),
        "url": account.get("url"),
    }
    try:
        context["usage_remaining_seconds"] = service.usage().get("usage_remaining_seconds")
    except Exception as exc:  # usage() is not on every runtime version / plan
        context["usage_error"] = f"{type(exc).__name__}: {exc}"
    return context


def append_ledger(root: Path, record: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def stage_preflight(root: Path, args: argparse.Namespace) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    context = account_context()
    print("IBM account context:")
    print(json.dumps(context, indent=2))
    if not context.get("instance"):
        print(
            "\n  WARNING: no instance on the active account. §F2 recorded exactly this for"
            "\n  account 059312687e3b4f8484a4d6cd7c311a3d (resource controller rows_count: 0)."
            "\n  Create a Qiskit Runtime instance at https://quantum.cloud.ibm.com first --"
            "\n  a valid key alone cannot submit."
        )

    service = QiskitRuntimeService()
    reachable = sorted(b.name for b in service.backends(operational=True, simulator=False))
    print(f"\nReachable QPUs ({len(reachable)}): {', '.join(reachable) or '(none)'}")
    for name in (PILOT_BACKEND, ANCHOR_PRIMARY, ANCHOR_CROSS):
        mark = "ok" if name in reachable else "NOT REACHABLE -- pass --ibm-backend to substitute"
        print(f"  {name:18} {mark}")

    if os.environ.get("IONQ_API_TOKEN"):
        try:
            from qiskit_ionq import IonQProvider

            targets = IonQProvider(token=os.environ["IONQ_API_TOKEN"]).backends()
            # .name is a plain attribute on some qiskit-ionq versions, a method on others
            names = sorted(str(b.name() if callable(b.name) else b.name) for b in targets)
            print(f"\nIonQ reachable: {', '.join(names)}")
        except Exception as exc:
            print(f"\nIonQ check failed: {type(exc).__name__}: {exc}")
    else:
        print("\nIONQ_API_TOKEN not set -- the 10 IonQ jobs will be skipped.")

    append_ledger(root, {"stage": "preflight", "utc": utc_now(), "account": context, "backends": reachable})


# ---------------------------------------------------------------------------
# pilot -- paper Table 3 / supplement §7.4 (5 IBM jobs, 176 circuits, 256 shots)
# ---------------------------------------------------------------------------


def stage_pilot_monalisa(root: Path, args: argparse.Namespace) -> None:
    module = importlib.import_module("run_hvk_hardware_reconstruction")
    out = root / "hardware_reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    module.OUTPUT_DIR = out

    backend = args.ibm_backend or PILOT_BACKEND
    context = account_context()
    sys.argv = [
        "run_hvk_hardware_reconstruction.py",
        "--submit",
        "--backend", backend,
        "--shots", str(args.shots),
        "--max-patches", "16",
        "--allow-large-job",
    ]
    module.main()

    report = json.loads((out / "hardware_reconstruction_report.json").read_text())
    append_ledger(root, {
        "stage": "pilot", "label": "Monalisa (HVK1D)", "provider": "ibm",
        "utc": utc_now(), "account": context, "n_circuits": 48,
        "backend": report["backend"], "shots": report["shots"], "job_id": report["job_id"],
        "psnr_db": report["psnr_hardware_db"],
        "paper_table": "paper Table 3", "paper_psnr_db": 25.896,
    })


def stage_pilot_cifar(root: Path, args: argparse.Namespace) -> None:
    module = importlib.import_module("run_hvk2d_cifar_hardware_reconstruction")
    out = root / "hvk2d_cifar_hardware_reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    module.OUTPUT_DIR = out

    backend = args.ibm_backend or PILOT_BACKEND
    context = account_context()
    sys.argv = [
        "run_hvk2d_cifar_hardware_reconstruction.py",
        "--submit",
        "--backend", backend,
        "--shots", str(args.shots),
        "--max-patches", "16",
        "--allow-large-job",
    ]
    module.main()

    # One job per image; the paper's four CIFAR PSNRs, in checkpoint-stem order.
    paper_psnr = {
        "0000_cat_domestic_cat_s_000907": 31.521,
        "0001_ship_hydrofoil_s_000078": 26.440,
        "0002_ship_sea_boat_s_001456": 31.196,
        "0003_airplane_jetliner_s_001705": 29.099,
    }
    for result in json.loads((out / "summary.json").read_text()):
        append_ledger(root, {
            "stage": "pilot", "label": result["stem"], "provider": "ibm",
            "utc": utc_now(), "account": context, "n_circuits": 32,
            "backend": result["backend"], "shots": args.shots, "job_id": result["job_id"],
            "psnr_db": result["psnr_hardware_db"],
            "paper_table": "paper Table 3", "paper_psnr_db": paper_psnr.get(result["stem"]),
        })


# ---------------------------------------------------------------------------
# anchors -- paper Table 4 (4 IBM jobs)
# ---------------------------------------------------------------------------


def stage_anchors(root: Path, args: argparse.Namespace) -> None:
    module = importlib.import_module("run_hardware_robustness_real_anchors")
    out = root / "hardware_robustness_study"
    out.mkdir(parents=True, exist_ok=True)
    module.OUT_DIR = out

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    context = account_context()

    monalisa = module.load_hvk1d()
    cat = module.load_hvk2d("0000_cat_domestic_cat_s_000907")

    # All four rows of Table 4. The shipped driver carries only three: its cross-backend
    # marrakesh point was already complete when it was written, so it was left out of the
    # job list. A from-scratch campaign has to submit it too.
    plan = [
        (monalisa, args.ibm_backend or ANCHOR_CROSS, 256, 25.942),
        (monalisa, args.ibm_backend or ANCHOR_PRIMARY, 1024, 26.103),
        (cat, args.ibm_backend or ANCHOR_PRIMARY, 256, 28.926),
        (cat, args.ibm_backend or ANCHOR_PRIMARY, 1024, 31.237),
    ]

    results = []
    for checkpoint, backend_name, shots, paper_psnr in plan:
        backend = service.backend(backend_name)
        print(f"\n=== {checkpoint['topology']} / {checkpoint['image_name']} "
              f"on {backend.name} @ {shots} shots ===", flush=True)
        record = module.run_on_hardware(checkpoint, backend, shots)
        results.append(record)
        (out / "real_hardware_anchors.json").write_text(json.dumps(results, indent=2))
        append_ledger(root, {
            "stage": "anchors", "label": f"{checkpoint['image_name']} ({checkpoint['topology']})",
            "provider": "ibm", "utc": utc_now(), "account": context,
            "n_circuits": record["n_circuits"], "backend": record["backend"],
            "shots": record["shots"], "job_id": record["job_id"], "psnr_db": record["psnr"],
            "paper_table": "paper Table 4", "paper_psnr_db": paper_psnr,
        })


# ---------------------------------------------------------------------------
# replay sweeps -- supplement Table 15. These drivers already take --output-dir,
# so they run unmodified as subprocesses; job ids are harvested afterwards.
# ---------------------------------------------------------------------------

SWEEPS = {
    "order": ("run_checkpoint_hardware_sweep.py", "checkpoint_hardware_sweep",
              ["ionq"], [256, 512, 1024], "Order parameter, cross-platform"),
    "temperature": ("run_checkpoint_temperature_hardware_sweep.py", "checkpoint_temperature_hardware_sweep",
                    ["ionq"], [1024], "R_ES, three-basis"),
    "bond-dim": ("run_checkpoint_bond_dimension_hardware_sweep.py", "checkpoint_bond_dimension_hardware_sweep",
                 ["ibm", "ionq"], [256, 512, 1024], "Bond-dim. order parameter"),
    "bond-dim-temp": ("run_checkpoint_bond_dimension_temperature_hardware_sweep.py",
                      "checkpoint_bond_dimension_temperature_hardware_sweep",
                      ["ibm", "ionq"], [256, 512, 1024], "Bond-dim. R_ES"),
}


def stage_sweep(root: Path, args: argparse.Namespace, name: str) -> None:
    script, out_name, providers, shots_list, campaign = SWEEPS[name]
    if args.providers:
        providers = [p for p in providers if p in args.providers]
    out = root / out_name
    out.mkdir(parents=True, exist_ok=True)

    if "ionq" in providers and not os.environ.get("IONQ_API_TOKEN"):
        providers = [p for p in providers if p != "ionq"]
        print(f"[{name}] IONQ_API_TOKEN not set -- IonQ jobs skipped")
    if not providers:
        print(f"[{name}] nothing to submit")
        return

    context = account_context() if "ibm" in providers else {}
    command = [sys.executable, str(IBM_DIR / script), "--providers", *providers,
               "--ibm-backend", args.ibm_backend or SWEEP_BACKEND, "--output-dir", str(out)]
    for shots in shots_list:
        command += ["--shots", str(shots)]
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))

    for path in sorted(out.glob("*.json")):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        job_ids = {row.get("job_id") for row in rows if row.get("job_id")}
        if not job_ids:
            continue
        provider = "ionq" if path.name.startswith("ionq") else "ibm"
        append_ledger(root, {
            "stage": name, "label": campaign, "provider": provider, "utc": utc_now(),
            "account": context if provider == "ibm" else {}, "n_circuits": len(rows),
            "backend": rows[0].get("backend") or path.stem, "shots": rows[0].get("shots"),
            "job_id": sorted(job_ids)[0], "artifact": str(path.relative_to(REPO_ROOT)),
            "paper_table": "supplement Table 15",
        })


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def stage_ledger(root: Path, args: argparse.Namespace) -> None:
    lines = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines() if line.strip()]
    jobs, seen = [], set()
    for record in lines:
        job_id = record.get("job_id")
        if job_id and job_id not in seen:
            seen.add(job_id)
            jobs.append(record)

    crns = {record["account"].get("instance") for record in jobs
            if record.get("account", {}).get("instance")}
    (root / "ledger.json").write_text(json.dumps({
        "campaign": "provenance re-submission",
        "compiled_utc": utc_now(),
        "instance_crn": sorted(crns),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }, indent=2))

    rows = ["| Campaign | Platform | Backend | Shots | Job ID | New PSNR (dB) | Paper PSNR (dB) |",
            "|---|---|---|---|---|---|---|"]
    for record in jobs:
        psnr = f"{record['psnr_db']:.3f}" if record.get("psnr_db") is not None else "--"
        paper = f"{record['paper_psnr_db']:.3f}" if record.get("paper_psnr_db") is not None else "--"
        platform = "IonQ" if record["provider"] == "ionq" else "IBM hardware"
        rows.append(f"| {record['label']} | {platform} | `{record['backend']}` | "
                    f"{record.get('shots') or '--'} | `{record['job_id']}` | {psnr} | {paper} |")
    rows += ["", f"Instance / CRN: {', '.join(sorted(crns)) or '(none recorded)'}", ""]
    rows.append("These are NEW jobs on a live account, not a retrieval of the jobs printed in")
    rows.append("the manuscripts -- those are unrecoverable (§F2). The PSNR columns are expected")
    rows.append("to differ: different hardware, calibration and day. Nothing in the manuscripts")
    rows.append("should be changed to match this table.")
    (root / "ledger.md").write_text("\n".join(rows) + "\n")

    write_latex(root, jobs, sorted(crns))

    print("\n".join(rows))
    print(f"\nWrote {root / 'ledger.json'}, {root / 'ledger.md'} and {root / 'ledger.tex'}")


# Friendly names for the checkpoint stems, matching how the supplement already writes them.
STEM_NAMES = {
    "0000_cat_domestic_cat_s_000907": "CIFAR cat",
    "0001_ship_hydrofoil_s_000078": "CIFAR ship (hydrofoil)",
    "0002_ship_sea_boat_s_001456": "CIFAR ship (sea boat)",
    "0003_airplane_jetliner_s_001705": "CIFAR airplane",
    "monalisa": "Monalisa",
}


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def pretty(record: dict) -> str:
    label = record["label"]
    for stem, name in STEM_NAMES.items():
        if stem in label:
            return name if "HVK" not in label else f"{name} ({label.split('(')[-1].rstrip(')')})"
    return label


def write_latex(root: Path, jobs: list[dict], crns: list[str]) -> None:
    """Two tables: reconstruction jobs carry a PSNR, replay jobs do not."""
    recon = [j for j in jobs if j["stage"] in ("pilot", "anchors")]
    replay = [j for j in jobs if j["stage"] not in ("pilot", "anchors")]
    out = []

    if recon:
        out += [
            r"\begin{table}[H]", r"\centering",
            r"\caption{Re-execution of the reconstruction pilot and repeated-execution anchors "
            r"under a live IBM Quantum instance (2026-09-05). These are new jobs on new hardware "
            r"calibration, not a retrieval of the identifiers in Tables~3 and~4: each PSNR is the "
            r"value its own job returned, and the originals are unrecoverable (the submitting "
            r"account is closed).}",
            r"\label{tab:reexecution_ledger}", r"\scriptsize",
            r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lllccc}", r"\toprule",
            r"Image & Backend & Job ID & Shots & PSNR (dB) & Originally printed (dB) \\", r"\midrule",
        ]
        for record in recon:
            psnr = f"{record['psnr_db']:.3f}" if record.get("psnr_db") is not None else "--"
            paper = f"{record['paper_psnr_db']:.3f}" if record.get("paper_psnr_db") is not None else "--"
            out.append(f"{pretty(record)} & \\texttt{{{tex_escape(record['backend'])}}} & "
                       f"\\texttt{{{record['job_id']}}} & {record.get('shots')} & {psnr} & {paper} \\\\")
        out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]

    if replay:
        out += [
            r"\begin{table}[H]", r"\centering",
            r"\caption{Re-execution of the replay ledger (2026-09-05), same campaigns as "
            r"Table~\ref{tab:replay_job_ledger}. Auditability only; no claim rests on these.}",
            r"\label{tab:reexecution_replay_ledger}", r"\scriptsize",
            r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lllll}", r"\toprule",
            r"Campaign & Platform & Backend & Shots & Job ID \\", r"\midrule",
        ]
        for record in replay:
            platform = "IonQ ideal sim." if record["provider"] == "ionq" else "IBM hardware"
            out.append(f"{tex_escape(record['label'])} & {platform} & "
                       f"\\texttt{{{tex_escape(record['backend'])}}} & {record.get('shots') or '--'} & "
                       f"\\texttt{{{record['job_id']}}} \\\\")
        out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]

    for crn in crns:
        out.append(f"% instance CRN: {crn}")
    (root / "ledger.tex").write_text("\n".join(out) + "\n")


STAGES = {
    "preflight": stage_preflight,
    "pilot-monalisa": stage_pilot_monalisa,
    "pilot-cifar": stage_pilot_cifar,
    "anchors": stage_anchors,
    "order": lambda root, args: stage_sweep(root, args, "order"),
    "temperature": lambda root, args: stage_sweep(root, args, "temperature"),
    "bond-dim": lambda root, args: stage_sweep(root, args, "bond-dim"),
    "bond-dim-temp": lambda root, args: stage_sweep(root, args, "bond-dim-temp"),
    "ledger": stage_ledger,
}

ALL_ORDER = ["preflight", "pilot-monalisa", "pilot-cifar", "anchors",
             "order", "temperature", "bond-dim", "bond-dim-temp", "ledger"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="preflight", choices=[*STAGES, "all"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shots", type=int, default=256, help="pilot shot count (Table 3 used 256)")
    parser.add_argument("--ibm-backend", help="override every IBM backend (when the originals are unreachable)")
    parser.add_argument("--providers", nargs="+", choices=["ibm", "ionq"],
                        help="narrow a sweep stage to these providers (e.g. when the IBM half already ran)")
    args = parser.parse_args()

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    if args.stage != "all":
        STAGES[args.stage](root, args)
        return

    # Each stage in its own process: a queue timeout or a dead backend in one stage
    # must not cost the stages that already completed.
    for name in ALL_ORDER:
        command = [sys.executable, __file__, "--stage", name, "--output-root", str(root),
                   "--shots", str(args.shots)]
        if args.ibm_backend:
            command += ["--ibm-backend", args.ibm_backend]
        if args.providers:
            command += ["--providers", *args.providers]
        print(f"\n{'=' * 70}\n== stage: {name}\n{'=' * 70}", flush=True)
        completed = subprocess.run(command, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            print(f"\n!! stage {name} failed (exit {completed.returncode}); continuing.", flush=True)


if __name__ == "__main__":
    main()
