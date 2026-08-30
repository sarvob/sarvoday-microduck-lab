import os
import sys

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nodes import demonstrate, design_lesson, train_policy  # noqa: E402

# Pull the MJCF, meshes and policies during boot (~10 MB, ~45 s cold) so the
# first visitor's lesson doesn't pay for it. Non-fatal: the sim refetches lazily.
try:
    import duck
    duck.ensure_assets()
    print("microduck assets ready")
except Exception as e:
    print("microduck asset prefetch failed (" + type(e).__name__ + ": "
          + str(e) + "); will retry on first run")

demo = gr.Workflow(
    os.path.join(HERE, "workflow.json"),
    bind={"design_lesson": design_lesson, "train_policy": train_policy,
          "demonstrate": demonstrate},
)

if __name__ == "__main__":
    demo.launch()
