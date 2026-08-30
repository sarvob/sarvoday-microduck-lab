"""End-to-end test: every subject in workflow.json through the real
`WorkflowExecutor` — the same code path the canvas and the REST API use.

    python apps/09_microduck_lab/test_pipelines.py

Outputs land in ./_test_output for eyeballing. The choreographer hits an HF
Inference Provider when a token is present and otherwise falls back to the
keyword planner, so this runs offline too (after the first asset download).
"""

import json
import logging
import os
import shutil
import sys
import time
import types
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gradio.workflow as W  # noqa: E402
from gradio.helpers import special_args  # noqa: E402
from gradio.workflow_api import (  # noqa: E402
    WorkflowExecutor,
    WorkflowGraph,
    group_free_inputs,
)

import nodes as N  # noqa: E402

try:
    from huggingface_hub import get_token
    _tok = get_token()
except Exception:
    _tok = None
TOKEN = types.SimpleNamespace(token=_tok or os.environ.get("HF_TOKEN"))

OUTDIR = os.path.join(HERE, "_test_output")
os.makedirs(OUTDIR, exist_ok=True)


def call_fn(data, request=None, token=None):
    """Mirror of gradio's bound-function server fn (workflow.py `call_fn`)."""
    name = data[0] if data else ""
    fn = N.BIND.get(name)
    if fn is None:
        return json.dumps({"error": "No function '" + str(name) + "' bound"})
    try:
        args = json.loads(data[1] if len(data) > 1 else "[]")
        if not isinstance(args, list):
            args = [args]
        args, *_ = special_args(fn, args, request, None, token=token)
        result = fn(*args)
        return json.dumps(list(result) if isinstance(result, (list, tuple)) else [result])
    except Exception as e:
        return json.dumps({"error": type(e).__name__ + ": " + str(e)})


CALLERS = {"fn": call_fn, "model": W.call_model,
           "space": W.call_space, "dataset": W.fetch_dataset}

with open(os.path.join(HERE, "workflow.json"), encoding="utf-8") as f:
    GRAPH = WorkflowGraph.from_json(f.read())


# Training is the slow part, so the suite runs a short lesson by default.
OVERRIDES = {}   # exercise the real defaults


def seed_for(subject_id):
    node = GRAPH.node_by_id[subject_id]
    inputs = {}
    for free in group_free_inputs(GRAPH, [node]):
        ref = free["node"]
        rid = ref["id"]
        inputs[rid] = (OVERRIDES[rid] if rid in OVERRIDES
                       else (ref.get("data") or {}).get("out"))
    return inputs


def save(subject_id, value):
    """Persist an output so it can be looked at; return a one-line summary."""
    if isinstance(value, dict) and value.get("path"):
        src = value["path"]
        ext = os.path.splitext(src)[1] or ".bin"
        dst = os.path.join(OUTDIR, subject_id + ext)
        shutil.copyfile(src, dst)
        kb = os.path.getsize(dst) // 1024
        url = str(value.get("url") or "")
        shape = "path+url" if url.startswith("data:") else "path only (CANVAS WILL BREAK)"
        return ext[1:].upper() + ", " + str(kb) + " KB, " + shape + " -> " + os.path.basename(dst)

    if isinstance(value, str) and os.path.isfile(value):
        dst = os.path.join(OUTDIR, subject_id + os.path.splitext(value)[1])
        shutil.copyfile(value, dst)
        return "file -> " + os.path.basename(dst)

    text = str(value)
    with open(os.path.join(OUTDIR, subject_id + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    first = text.strip().splitlines()[0][:70] if text.strip() else "(empty)"
    return "text (" + str(len(text)) + " chars): " + first


def main():
    executor = WorkflowExecutor(GRAPH, CALLERS)
    targets = [s["id"] for s in GRAPH.subjects]
    if len(sys.argv) > 1:
        f = [a.lower() for a in sys.argv[1:]]
        targets = [t for t in targets if any(a in t.lower() for a in f)]

    print("\nRunning " + str(len(targets)) + " output(s) through WorkflowExecutor")
    print("outputs -> " + os.path.relpath(OUTDIR, os.getcwd()) + "\n")

    passed, failed = [], []
    for sid in targets:
        label = GRAPH.node_by_id[sid]["label"]
        t0 = time.time()
        try:
            value = executor.run(sid, seed_for(sid), request=None, token=TOKEN)
            if value is None or value == "":
                raise AssertionError("output was empty")
            print("  PASS  %-14s %-16s %6.1fs  %s"
                  % (sid, label, time.time() - t0, save(sid, value)))
            passed.append(sid)
        except Exception as e:
            print("  FAIL  %-14s %-16s %6.1fs  %s: %s"
                  % (sid, label, time.time() - t0, type(e).__name__, e))
            failed.append(sid)

    print("\n" + str(len(passed)) + " passed, " + str(len(failed)) + " failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
