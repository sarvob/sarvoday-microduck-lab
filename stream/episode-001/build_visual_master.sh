#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
episode_dir="$repo_root/stream/episode-001"

python3 "$episode_dir/generate_graphics.py"

ffmpeg -y -hide_banner -loglevel error \
  -i "$repo_root/artifacts/001-spin-in-place/demonstration.mp4" \
  -i "$repo_root/artifacts/002-two-marker-sprint/demonstration.mp4" \
  -i "$repo_root/artifacts/003-ball-push/demonstration.mp4" \
  -loop 1 -i "$episode_dir/intro.png" \
  -loop 1 -i "$episode_dir/overlay-1.png" \
  -loop 1 -i "$episode_dir/overlay-2.png" \
  -loop 1 -i "$episode_dir/overlay-3.png" \
  -loop 1 -i "$episode_dir/outro.png" \
  -filter_complex "
    [3:v]fps=30,trim=duration=6,setpts=PTS-STARTPTS,format=yuv420p[intro];
    [0:v]scale=1920:1080:flags=lanczos,fps=30,tpad=stop_mode=clone:stop_duration=2,setpts=PTS-STARTPTS[v0];
    [4:v]fps=30,format=rgba[ov0];[v0][ov0]overlay=shortest=1[v0w];
    [1:v]scale=1920:1080:flags=lanczos,fps=30,tpad=stop_mode=clone:stop_duration=3.36,setpts=PTS-STARTPTS[v1];
    [5:v]fps=30,format=rgba[ov1];[v1][ov1]overlay=shortest=1[v1w];
    [2:v]scale=1920:1080:flags=lanczos,fps=30,tpad=stop_mode=clone:stop_duration=2,setpts=PTS-STARTPTS[v2];
    [6:v]fps=30,format=rgba[ov2];[v2][ov2]overlay=shortest=1[v2w];
    [7:v]fps=30,trim=duration=12,setpts=PTS-STARTPTS,format=yuv420p[outro];
    [intro][v0w][v1w][v2w][outro]concat=n=5:v=1:a=0[video]
  " \
  -map "[video]" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart \
  "$episode_dir/visual-master.mp4"

ffprobe -v error -show_entries format=duration,size -of json "$episode_dir/visual-master.mp4"
