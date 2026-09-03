#!/usr/bin/env bash
set -euo pipefail

episode_dir="$(cd "$(dirname "$0")" && pwd)"
output="$episode_dir/episode-004-controlled-roll.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -i "$episode_dir/search-evaluation-montage.mp4" \
  -i "$episode_dir/controller-diagram.mp4" \
  -i "$episode_dir/success-gates.mp4" \
  -i "$episode_dir/timing-sweep.mp4" \
  -i "$episode_dir/roll-failure-0.84s-seed-23.mp4" \
  -i "$episode_dir/roll-evidence-3-seeds.mp4" \
  -i "$episode_dir/result-table.mp4" \
  -i "$episode_dir/lesson-outro.mp4" \
  -i "$episode_dir/narration.wav" \
  -filter_complex "
    [0:v]trim=0:20,setpts=PTS-STARTPTS[m0];
    [1:v]setpts=PTS-STARTPTS[g0];
    [0:v]trim=20:40,setpts=PTS-STARTPTS[m1];
    [2:v]setpts=PTS-STARTPTS[g1];
    [3:v]setpts=PTS-STARTPTS[g2];
    [0:v]trim=40:90,setpts=PTS-STARTPTS[m2];
    [4:v]setpts=PTS-STARTPTS[f0];
    [0:v]trim=90:120,setpts=PTS-STARTPTS[m3];
    [5:v]setpts=PTS-STARTPTS[s0];
    [0:v]trim=120:135,setpts=PTS-STARTPTS[m4];
    [6:v]setpts=PTS-STARTPTS[g3];
    [0:v]trim=135:167.7,setpts=PTS-STARTPTS[m5];
    [7:v]setpts=PTS-STARTPTS[g4];
    [m0][g0][m1][g1][g2][m2][f0][m3][s0][m4][g3][m5][g4]concat=n=13:v=1:a=0[v];
    [8:a]highpass=f=70,lowpass=f=14500,afftdn=nf=-30,
         dynaudnorm=f=150:g=12,loudnorm=I=-16:TP=-1.5:LRA=11,
         aresample=48000[a]
  " \
  -map "[v]" -map "[a]" -shortest \
  -c:v libx264 -profile:v high -preset slow -crf 16 -pix_fmt yuv420p \
  -r 60 -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$output"

echo "$output"
