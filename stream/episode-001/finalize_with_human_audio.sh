#!/usr/bin/env bash
set -euo pipefail

episode_dir="$(cd "$(dirname "$0")" && pwd)"
human_audio="${1:-$episode_dir/human-narration.wav}"
visual_master="$episode_dir/visual-master.mp4"
final_video="$episode_dir/final-with-human-narration.mp4"

if [[ ! -f "$human_audio" ]]; then
  echo "Missing genuine human narration: $human_audio" >&2
  echo "Record the approved narration script; synthetic or cloned speech is not accepted." >&2
  exit 2
fi

duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$human_audio")"
python3 - "$duration" <<'PY'
import sys

duration = float(sys.argv[1])
if not 35.0 <= duration <= 60.0:
    raise SystemExit(
        f"Human narration is {duration:.1f}s; re-record naturally within the approved 35–60s range."
    )
PY

ffmpeg -y -hide_banner -loglevel error \
  -i "$visual_master" -i "$human_audio" \
  -filter_complex "[1:a]highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=f=150:g=15,loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur=50,atrim=0:50[a]" \
  -map 0:v:0 -map "[a]" -c:v copy -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart -shortest "$final_video"

ffprobe -v error -show_entries stream=codec_type,codec_name -show_entries format=duration,size \
  -of json "$final_video"

echo "Created $final_video"
echo "Required next step: watch and listen end-to-end, then align captions-draft.srt to the real delivery."
