"""Compile a LaTeX source to PDF, robustly, on MiKTeX.

    from compile_tex import compile_tex
    pdf = compile_tex("paper_latex/paper_hvk.tex")   # -> Path to the .pdf

Why this exists (the three sharp edges of MiKTeX we kept hitting):

  1. On-the-fly package install prompts hang under ``-interaction=nonstopmode``:
     the installer asks "enter new name", never gets input, and the build dies
     with a bare "file `foo.sty' not found".
  2. After installing a package with ``mpm``, MiKTeX's filename database is
     stale, so LaTeX still can't find the new ``.sty`` until ``initexmf
     --update-fndb`` runs.
  3. ``latexmk`` caches a failed run and then reports "Nothing to do /
     up-to-date", refusing to rebuild until its state is cleaned (``-C``) or the
     build is forced (``-g``).

``compile_tex`` handles all three: it runs latexmk forced+clean, and on a
"file not found" error it installs the providing package, refreshes the fndb,
and retries once.

Requires MiKTeX (``latexmk``, ``mpm``, ``initexmf``, ``kpsewhich`` on PATH).
Bibliography is assumed inline (``thebibliography``); latexmk still runs the
extra pass needed to settle cross-references.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

__all__ = ["compile_tex", "LatexCompileError"]


class LatexCompileError(RuntimeError):
    """Raised when the document cannot be compiled to a PDF."""


# "! LaTeX Error: File `flushend.sty' not found."  ->  captures "flushend.sty"
_MISSING_STY = re.compile(r"File `([^']+\.sty)' not found")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise LatexCompileError(
            f"'{tool}' not found on PATH. This module needs a MiKTeX install "
            f"(latexmk, mpm, initexmf, kpsewhich)."
        )


def _package_for_sty(sty: str) -> str | None:
    """Return the mpm package name that provides ``sty`` (e.g. flushend.sty ->
    sttools), or None if it can't be determined. Uses ``mpm --list`` and matches
    on the basename minus extension appearing in a package's file set is not
    available cheaply, so we lean on the well-known CTAN bundle map plus a
    ``kpsewhich`` recheck after install."""
    # Known multi-file bundles where the .sty name != package name.
    known = {
        "flushend.sty": "sttools",
        "stfloats.sty": "sttools",
        "cases.sty": "sttools",
        "algorithm.sty": "algorithms",
        "algorithmic.sty": "algorithms",
        "algpseudocode.sty": "algorithmicx",
    }
    if sty in known:
        return known[sty]
    # Fall back to the common convention: package name == .sty basename.
    return Path(sty).stem


def _install_package(pkg: str, cwd: Path) -> bool:
    """Install a MiKTeX package and refresh the filename database. Returns True
    on apparent success."""
    proc = _run(["mpm", "--install", pkg], cwd)
    ok = proc.returncode == 0 and "unknown" not in (proc.stdout + proc.stderr).lower()
    # Refresh the fndb regardless — a partially-present bundle still needs it.
    _run(["initexmf", "--update-fndb"], cwd)
    return ok


def _latexmk(tex: str, cwd: Path) -> subprocess.CompletedProcess:
    # -g force, -pdf, halt on first error, non-interactive.
    return _run(
        ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex],
        cwd,
    )


def compile_tex(tex_path: str | Path, clean_aux: bool = True) -> Path:
    """Compile ``tex_path`` to a PDF beside it and return the PDF path.

    Runs latexmk forced (clearing any cached failure). If the build fails on a
    missing ``.sty``, installs the providing MiKTeX package, refreshes the
    filename database, and retries **once**. Raises ``LatexCompileError`` with
    the relevant LaTeX error on unrecoverable failure.

    Parameters
    ----------
    tex_path : path to the ``.tex`` file.
    clean_aux : if True (default), remove latexmk's auxiliary files first so a
        previously cached failure cannot suppress the rebuild.
    """
    for tool in ("latexmk", "mpm", "initexmf", "kpsewhich"):
        _require(tool)

    tex_path = Path(tex_path).resolve()
    if not tex_path.is_file():
        raise LatexCompileError(f"No such .tex file: {tex_path}")

    cwd = tex_path.parent
    name = tex_path.name
    pdf = tex_path.with_suffix(".pdf")

    if clean_aux:
        _run(["latexmk", "-C", name], cwd)  # clear aux + cached state

    proc = _latexmk(name, cwd)

    if proc.returncode != 0:
        # Did we fail because a package is missing? If so, install + retry once.
        log = _read_log(tex_path)
        m = _MISSING_STY.search(log) or _MISSING_STY.search(proc.stdout)
        if m:
            sty = m.group(1)
            pkg = _package_for_sty(sty)
            installed = _install_package(pkg, cwd) if pkg else False
            found = _run(["kpsewhich", sty], cwd).stdout.strip()
            if found:  # the .sty is now locatable regardless of mpm's chatter
                _run(["latexmk", "-C", name], cwd)  # clear the cached failure
                proc = _latexmk(name, cwd)
            else:
                raise LatexCompileError(
                    f"Missing LaTeX package file '{sty}'. Tried to install "
                    f"MiKTeX package '{pkg}' (installed={installed}) but the "
                    f"file is still not found. Install it manually with "
                    f"`mpm --install <package>` then rerun."
                )

    if proc.returncode != 0 or not pdf.is_file():
        raise LatexCompileError(_first_error(tex_path, proc))

    return pdf


def _read_log(tex_path: Path) -> str:
    log = tex_path.with_suffix(".log")
    if log.is_file():
        return log.read_text(encoding="utf-8", errors="replace")
    return ""


def _first_error(tex_path: Path, proc: subprocess.CompletedProcess) -> str:
    """Best-effort extraction of the first real LaTeX error for the exception
    message."""
    log = _read_log(tex_path)
    for line in log.splitlines():
        if line.startswith("!"):
            return f"LaTeX failed: {line.lstrip('! ').rstrip()}"
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-5:]
    return "LaTeX failed (no '!' line in log). Tail:\n" + "\n".join(tail)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python compile_tex.py <file.tex>", file=sys.stderr)
        raise SystemExit(2)
    out = compile_tex(sys.argv[1])
    print(out)
