#!/usr/bin/env bash
set -euo pipefail
episode_dir="$(cd "$(dirname "$0")" && pwd)"
output="$episode_dir/episode-005-boat-balance.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -t 12 -i "$episode_dir/01-intro.png" \
  -loop 1 -t 18 -i "$episode_dir/02-problem.png" \
  -loop 1 -t 30 -i "$episode_dir/03-attempts.png" \
  -loop 1 -t 20 -i "$episode_dir/04-gates.png" \
  -loop 1 -t 26 -i "$episode_dir/05-diagnosis.png" \
  -loop 1 -t 28 -i "$episode_dir/06-next.png" \
  -loop 1 -t 30 -i "$episode_dir/07-outro.png" \
  -i "$episode_dir/harbor-3-seeds.mp4" \
  -i "$episode_dir/chop-3-seeds.mp4" \
  -i "$episode_dir/narration.wav" \
  -filter_complex "
    [0:v]fps=60,format=yuv420p[v0]; [1:v]fps=60,format=yuv420p[v1];
    [2:v]fps=60,format=yuv420p[v2]; [3:v]fps=60,format=yuv420p[v3];
    [4:v]fps=60,format=yuv420p[v4]; [5:v]fps=60,format=yuv420p[v5];
    [6:v]fps=60,format=yuv420p[v6];
    [7:v]trim=0:20,setpts=PTS-STARTPTS[h0];
    [7:v]trim=20:40,setpts=PTS-STARTPTS[h1];
    [7:v]trim=40:60,setpts=PTS-STARTPTS[h2];
    [8:v]setpts=PTS-STARTPTS[c0];
    [8:v]trim=0:8.04,setpts=(PTS-STARTPTS)/0.72[c1];
    [v0][v1][v3][v2][h0][h1][h2][v4][c0][c1][v5][v6]concat=n=12:v=1:a=0[v];
    [9:a]highpass=f=70,lowpass=f=14500,afftdn=nf=-30,
      dynaudnorm=f=150:g=12,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]
  " \
  -map "[v]" -map "[a]" -shortest \
  -c:v libx264 -profile:v high -preset slow -crf 16 -pix_fmt yuv420p -r 60 \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$output"

echo "$output"
