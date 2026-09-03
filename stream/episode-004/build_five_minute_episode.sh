#!/usr/bin/env bash
set -euo pipefail

episode_dir="$(cd "$(dirname "$0")" && pwd)"
analysis="$episode_dir/handoff-analysis.mp4"
output="$episode_dir/episode-004-controlled-roll-5min.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -i "$episode_dir/controller-diagram.mp4" \
  -i "$episode_dir/roll-evidence-3-seeds.mp4" \
  -i "$episode_dir/roll-failure-0.84s-seed-23.mp4" \
  -i "$episode_dir/timing-sweep.mp4" \
  -i "$episode_dir/success-gates.mp4" \
  -i "$episode_dir/result-table.mp4" \
  -i "$episode_dir/lesson-outro.mp4" \
  -i "$episode_dir/analysis-narration.wav" \
  -filter_complex "
    [0:v]trim=0:8,setpts=PTS-STARTPTS[v0];
    [1:v]trim=0:4.5,setpts=2.0*(PTS-STARTPTS)[v1];
    [2:v]setpts=2.5*(PTS-STARTPTS)[v2];
    [3:v]trim=0:8,setpts=PTS-STARTPTS[v3];
    [4:v]trim=0:12,setpts=PTS-STARTPTS[v4];
    [5:v]trim=0:10,setpts=PTS-STARTPTS[v5];
    [6:v]setpts=PTS-STARTPTS[v6];
    [v0][v1][v2][v3][v4][v5][v6]concat=n=7:v=1:a=0[v];
    [7:a]highpass=f=70,lowpass=f=14500,afftdn=nf=-30,
         dynaudnorm=f=150:g=12,loudnorm=I=-16:TP=-1.5:LRA=11,
         aresample=48000[a]
  " \
  -map "[v]" -map "[a]" -shortest \
  -c:v libx264 -profile:v high -preset slow -crf 16 -pix_fmt yuv420p \
  -r 60 -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$analysis"

ffmpeg -y -hide_banner -loglevel error \
  -i "$episode_dir/episode-004-controlled-roll.mp4" \
  -i "$analysis" \
  -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -profile:v high -preset slow -crf 16 -pix_fmt yuv420p \
  -r 60 -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$output"

echo "$output"
