# Local narration

This folder uses `kokoro-onnx` with the Apache-2.0 Kokoro model and a neutral,
stock voice. It does not clone or impersonate a real person.

The local runtime lives in `.tts-venv/`. Model files live in
`stream/tts/models/`; both are intentionally excluded from Git because they
are generated or downloaded assets.

Generate narration from the repository root:

```bash
.tts-venv/bin/python stream/tts/generate_narration.py \
  "This is Sarvoday Robotics." \
  --output stream/tts/output/sample.wav
```

Sources:

- https://github.com/thewh1teagle/kokoro-onnx
- https://huggingface.co/hexgrad/Kokoro-82M
