"""Generate workflow.json. Run this, don't hand-edit the JSON.

    python build_workflow.py

The app AUTOSAVES workflow.json whenever it runs — a browser is not required,
a gradio_client call is enough. Kill every local server and re-run this before
committing or uploading.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LESSON = "teach it to turn around and walk to a marker behind it, without falling over"


def port(pid, label, ptype, required=False, output_index=None):
    p = {"id": pid, "label": label, "type": ptype}
    if required:
        p["required"] = True
    if output_index is not None:
        p["output_index"] = output_index
    return p


def edge(eid, src, src_port, tgt, tgt_port, etype):
    return {"id": eid, "from_node_id": src, "from_port_id": src_port,
            "to_node_id": tgt, "to_port_id": tgt_port, "type": etype}


GRAPH = {
    "schema_version": "2",
    "name": "Microduck School",
    "references": [
        {
            "id": "ref_lesson", "role": "reference", "label": "Lesson",
            "asset_type": "text",
            "inputs": [port("in", "Lesson", "text")],
            "outputs": [port("out", "Lesson", "text")],
            "data": {"out": DEFAULT_LESSON},
            "x": 40, "y": 60, "width": 260, "height": 160,
        },
        {
            "id": "ref_gens", "role": "reference", "label": "Generations",
            "asset_type": "number",
            "inputs": [port("in", "Generations", "number")],
            "outputs": [port("out", "Generations", "number")],
            "data": {"out": 8},
            "x": 40, "y": 260, "width": 260, "height": 110,
        },
        {
            "id": "ref_seed", "role": "reference", "label": "Seed",
            "asset_type": "number",
            "inputs": [port("in", "Seed", "number")],
            "outputs": [port("out", "Seed", "number")],
            "data": {"out": 0},
            "x": 40, "y": 410, "width": 260, "height": 110,
        },
    ],
    "operators": [
        {
            "id": "op_design", "role": "operator", "kind": "fn",
            "fn": "design_lesson", "label": "design_lesson (LLM)",
            "inputs": [port("in_lesson", "lesson", "text")],
            "outputs": [port("out_0", "spec", "text", output_index=0)],
            "data": {}, "x": 360, "y": 70, "width": 250, "height": 130,
        },
        {
            "id": "op_train", "role": "operator", "kind": "fn",
            "fn": "train_policy", "label": "train_policy (CEM)",
            "inputs": [port("in_spec", "spec", "text", required=True),
                       port("in_gens", "generations", "number"),
                       port("in_seed", "seed", "number")],
            "outputs": [port("out_0", "controller", "text", output_index=0),
                        port("out_1", "curve", "image", output_index=1)],
            "data": {}, "x": 670, "y": 220, "width": 260, "height": 170,
        },
        {
            "id": "op_show", "role": "operator", "kind": "fn",
            "fn": "demonstrate", "label": "demonstrate (before / after)",
            "inputs": [port("in_spec", "spec", "text", required=True),
                       port("in_policy", "controller", "text", required=True)],
            "outputs": [port("out_0", "video", "video", output_index=0),
                        port("out_1", "tracks", "image", output_index=1),
                        port("out_2", "report", "text", output_index=2)],
            "data": {}, "x": 990, "y": 70, "width": 270, "height": 180,
        },
    ],
    "subjects": [
        {
            "id": "sub_video", "role": "subject", "label": "The duck, after school",
            "asset_type": "video",
            "inputs": [port("in", "Video", "video")],
            "outputs": [], "data": {},
            "x": 1320, "y": 40, "width": 280, "height": 230,
        },
        {
            "id": "sub_curve", "role": "subject", "label": "Learning curve",
            "asset_type": "image",
            "inputs": [port("in", "Curve", "image")],
            "outputs": [], "data": {},
            "x": 670, "y": 430, "width": 280, "height": 210,
        },
        {
            "id": "sub_tracks", "role": "subject", "label": "Before / after",
            "asset_type": "image",
            "inputs": [port("in", "Tracks", "image")],
            "outputs": [], "data": {},
            "x": 1320, "y": 300, "width": 280, "height": 230,
        },
        {
            "id": "sub_report", "role": "subject", "label": "Report card",
            "asset_type": "text",
            "inputs": [port("in", "Report", "text")],
            "outputs": [], "data": {},
            "x": 1320, "y": 560, "width": 280, "height": 190,
        },
        {
            "id": "sub_spec", "role": "subject", "label": "The lesson, as a goal",
            "asset_type": "text",
            "inputs": [port("in", "Spec", "text")],
            "outputs": [], "data": {},
            "x": 360, "y": 240, "width": 250, "height": 190,
        },
    ],
    "edges": [
        edge("e1", "ref_lesson", "out", "op_design", "in_lesson", "text"),
        edge("e2", "op_design", "out_0", "sub_spec", "in", "text"),
        edge("e3", "op_design", "out_0", "op_train", "in_spec", "text"),
        edge("e4", "ref_gens", "out", "op_train", "in_gens", "number"),
        edge("e5", "ref_seed", "out", "op_train", "in_seed", "number"),
        edge("e6", "op_train", "out_1", "sub_curve", "in", "image"),
        edge("e7", "op_design", "out_0", "op_show", "in_spec", "text"),
        edge("e8", "op_train", "out_0", "op_show", "in_policy", "text"),
        edge("e9", "op_show", "out_0", "sub_video", "in", "video"),
        edge("e10", "op_show", "out_1", "sub_tracks", "in", "image"),
        edge("e11", "op_show", "out_2", "sub_report", "in", "text"),
    ],
}


def verify(graph):
    """No dangling edges, and no port types the canvas destroys."""
    nodes = {n["id"]: n for n in
             graph["references"] + graph["operators"] + graph["subjects"]}
    problems = []
    for e in graph["edges"]:
        src, tgt = nodes.get(e["from_node_id"]), nodes.get(e["to_node_id"])
        if src is None or tgt is None:
            problems.append("edge " + e["id"] + " points at a missing node")
            continue
        if e["from_port_id"] not in {p["id"] for p in src["outputs"]}:
            problems.append("edge " + e["id"] + ": no output port "
                            + e["from_port_id"] + " on " + e["from_node_id"])
        if e["to_port_id"] not in {p["id"] for p in tgt["inputs"]}:
            problems.append("edge " + e["id"] + ": no input port "
                            + e["to_port_id"] + " on " + e["to_node_id"])
    for n in nodes.values():
        for p in n["inputs"] + n["outputs"]:
            # The canvas serializes these with String(obj) -> "[object Object]".
            if p["type"] in ("json", "dataframe"):
                problems.append(n["id"] + "." + p["id"]
                                + " uses canvas-broken type " + p["type"])
    return problems


if __name__ == "__main__":
    issues = verify(GRAPH)
    if issues:
        raise SystemExit("refusing to write workflow.json:\n  "
                         + "\n  ".join(issues))
    out = os.path.join(HERE, "workflow.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(GRAPH, f, indent=2)
        f.write("\n")
    print("wrote " + out)
    print("  %d references, %d operators, %d subjects, %d edges" % (
        len(GRAPH["references"]), len(GRAPH["operators"]),
        len(GRAPH["subjects"]), len(GRAPH["edges"])))
