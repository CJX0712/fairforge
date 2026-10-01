"""Idempotent GitHub push via the Git Data API (gh CLI), bypassing git push (502-prone).

Strategy (proven SOP):
  - empty repo  -> bootstrap with one Contents API PUT (creates initial commit)
  - then blobs -> tree -> commit (parents=[head]) -> PATCH/POST refs/heads/main
  - idempotent: re-running re-creates a new full commit on top (or force-updates)

Run: python scripts/push_to_github.py
Author: 晨星
"""

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "CJX0712/fairforge"
BRANCH = "main"
ROOT = Path(__file__).resolve().parent.parent
AUTHOR = {"name": "晨星", "email": "CJX0712@users.noreply.github.com"}
DESC = (
    "FairForge - reproducible fairness audit & mitigation toolkit "
    "with backend fallback (作者: 晨星)"
)

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "fairforge.egg-info",
    "build",
    "dist",
}
EXCLUDE_FILES = {".gitignore.bak", "push.log", "_gh_body.json", "_bootstrap.json"}
EXCLUDE_EXT = {".pyc", ".pyo", ".egg-info"}


def gh(args: list[str], input_bytes: bytes | None = None, max_retry: int = 12):
    """gh api call with backoff on transient/502 errors.

    On Windows the proxy drops stdin bodies for POST/PUT, so when a body is
    present we write it to a temp file and use --input (file), which works.
    """
    cmd = ["gh", "api"] + args
    args_str = " ".join(args)
    delay = 3
    last_err = ""
    tmp_path = None
    try:
        stdin = None
        if input_bytes is not None:
            tmp_path = ROOT / "_gh_body.json"
            tmp_path.write_bytes(input_bytes)
            cmd += ["--input", str(tmp_path)]
        else:
            stdin = b""
        for attempt in range(1, max_retry + 1):
            proc = subprocess.run(cmd, input=stdin, capture_output=True)
            out, err, code = proc.stdout, proc.stderr, proc.returncode
            err_s = err.decode("utf-8", "replace")
            last_err = err_s
            transient = code != 0 and any(
                k in err_s
                for k in (
                    "502",
                    "Bad Gateway",
                    "unexpected end of JSON",
                    "Connection reset",
                    "timed out",
                )
            )
            if code == 0:
                return json.loads(out) if out.strip() else None
            if not transient:
                # Non-transient: allow None only for missing ref/commit lookups
                if any(x in args_str for x in ("/git/ref/heads/", "/git/commits/")):
                    return None
                print(f"gh api FATAL: {cmd} -> {err_s[:500]}", flush=True)
                raise SystemExit(1)
            print(f"gh api retry {attempt}/{max_retry}: {err_s[:180]}", flush=True)
            time.sleep(delay)
            delay = min(delay + 2, 12)
        raise SystemExit(f"gh api retries exhausted: {cmd}\nlast={last_err[:300]}")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def ensure_repo():
    proc = subprocess.run(
        ["gh", "repo", "view", REPO, "--json", "name,url"], capture_output=True
    )
    if proc.returncode == 0:
        print("repo exists:", proc.stdout.decode()[:200], flush=True)
        return
    subprocess.run(
        ["gh", "repo", "create", REPO, "--public", "--description", DESC],
        check=True,
        capture_output=True,
    )
    print("repo created", flush=True)


def collect_files():
    files = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if rel.name in EXCLUDE_FILES or p.suffix in EXCLUDE_EXT:
            continue
        files.append((rel.as_posix(), p))
    return files


def bootstrap_empty():
    """Create an initial commit via Contents API so the repo has no git objects yet."""
    print("bootstrapping empty repo via Contents API...", flush=True)
    body = {
        "message": "bootstrap (作者: 晨星)",
        "content": base64.b64encode(b"# FairForge\n").decode(),
        "author": AUTHOR,
        "committer": AUTHOR,
    }
    # write to a temp file body
    tmp = ROOT / "_bootstrap.json"
    tmp.write_text(json.dumps(body), encoding="utf-8")
    res = subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "PUT",
            f"repos/{REPO}/contents/.gitkeep",
            "--input",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    try:
        tmp.unlink()
    except OSError:
        pass
    if res.returncode != 0:
        print("bootstrap FATAL:", res.stderr[:400], flush=True)
        raise SystemExit(1)
    print("bootstrap OK", flush=True)


def main():
    ensure_repo()

    # detect existing head
    ref = gh([f"repos/{REPO}/git/ref/heads/{BRANCH}"])
    if ref is None:
        # also check 409 empty via commits endpoint
        head_sha = None
        # try to bootstrap only if repo truly has no commits
        head_test = subprocess.run(
            ["gh", "api", f"repos/{REPO}/git/commits/main"], capture_output=True
        )
        if head_test.returncode != 0:
            bootstrap_empty()
        # re-query ref after bootstrap
        ref = gh([f"repos/{REPO}/git/ref/heads/{BRANCH}"])
        if ref is not None:
            head_sha = ref["object"]["sha"]
            print(f"head after bootstrap: {head_sha}", flush=True)
        else:
            # still none: use the bootstrap commit sha
            head_sha = None
    else:
        head_sha = ref["object"]["sha"]
        print(f"current head: {head_sha}", flush=True)

    # if head_sha still None (repo had objects but no main ref), find default branch
    if head_sha is None:
        view = subprocess.run(
            ["gh", "repo", "view", REPO, "--json", "defaultBranchRef"],
            capture_output=True,
            text=True,
        )
        if view.returncode == 0:
            dbr = json.loads(view.stdout).get("defaultBranchRef")
            if dbr and dbr.get("name"):
                dref = gh([f"repos/{REPO}/git/ref/heads/{dbr['name']}"])
                if dref:
                    head_sha = dref["object"]["sha"]
                    print(f"head via default branch: {head_sha}", flush=True)

    base_tree = None
    if head_sha:
        head_commit = gh([f"repos/{REPO}/git/commits/{head_sha}"])
        if head_commit:
            base_tree = head_commit["tree"]["sha"]

    # blobs
    tree_items = []
    for rel, path in collect_files():
        data = path.read_bytes()
        blob = gh(
            [f"repos/{REPO}/git/blobs"],
            input_bytes=json.dumps(
                {
                    "content": base64.b64encode(data).decode(),
                    "encoding": "base64",
                }
            ).encode(),
        )
        tree_items.append(
            {"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]}
        )
        if len(tree_items) % 15 == 0:
            print(f"  blobs {len(tree_items)}/{len(collect_files())}", flush=True)
    print(f"uploaded {len(tree_items)} blobs", flush=True)

    # tree
    tree_payload = {"tree": tree_items}
    if base_tree:
        tree_payload["base_tree"] = base_tree
    tree = gh([f"repos/{REPO}/git/trees"], input_bytes=json.dumps(tree_payload).encode())
    print(f"tree: {tree['sha']}", flush=True)

    # commit
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit_payload = {
        "message": (
            "FairForge v0.1.0: fairness audit & mitigation toolkit\n\n"
            "Unified fairness metrics, 8 mitigators (pre/in/post) with native fallback,\n"
            "FairPareto auto-selector flagship, 75 tests green, ruff clean.\n\n"
            "Author: 晨星"
        ),
        "tree": tree["sha"],
        "parents": [head_sha] if head_sha else [],
        "author": {**AUTHOR, "date": now},
        "committer": {**AUTHOR, "date": now},
    }
    commit = gh(
        [f"repos/{REPO}/git/commits"],
        input_bytes=json.dumps(commit_payload, ensure_ascii=False).encode("utf-8"),
    )
    print(f"commit: {commit['sha']}", flush=True)

    # ref
    if head_sha:
        gh(
            ["-X", "PATCH", f"repos/{REPO}/git/refs/heads/{BRANCH}"],
            input_bytes=json.dumps({"sha": commit["sha"], "force": True}).encode(),
        )
    else:
        gh(
            ["-X", "POST", f"repos/{REPO}/git/refs"],
            input_bytes=json.dumps(
                {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]}
            ).encode(),
        )
    print(f"ref {BRANCH} -> {commit['sha']}", flush=True)

    # verify
    ver = subprocess.run(
        ["gh", "repo", "view", REPO, "--json", "name,url,isEmpty,defaultBranchRef"],
        capture_output=True,
        text=True,
    )
    print(ver.stdout, flush=True)


if __name__ == "__main__":
    sys.exit(main())
