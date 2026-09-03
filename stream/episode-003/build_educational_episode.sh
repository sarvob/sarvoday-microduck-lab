#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
e="$repo_root/stream/episode-003"

ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -framerate 60 -i "$e/landscape-intro.jpg" \
  -i "$e/raw-scene-orbit-landscape.mp4" \
  -i "$e/raw-seed-17-side-landscape.mp4" \
  -loop 1 -framerate 60 -i "$e/landscape-method.jpg" \
  -i "$e/raw-unassisted-landscape.mp4" \
  -i "$e/animated-physics-comparison.mp4" \
  -loop 1 -framerate 60 -i "$e/landscape-method.jpg" \
  -i "$e/raw-seed-17-landscape.mp4" \
  -i "$e/raw-scene-orbit-landscape.mp4" \
  -i "$e/raw-untrained-baseline-landscape.mp4" \
  -i "$e/animated-controller-diagram.mp4" \
  -i "$e/raw-seed-17-side-landscape.mp4" \
  -i "$e/animated-training-curve.mp4" \
  -i "$e/raw-untrained-baseline-landscape.mp4" \
  -i "$e/raw-seed-17-side-landscape.mp4" \
  -i "$e/animated-results.mp4" \
  -i "$e/raw-seed-71-overhead-landscape.mp4" \
  -i "$e/raw-seed-17-landscape.mp4" \
  -i "$e/raw-seed-71-overhead-landscape.mp4" \
  -i "$e/raw-seed-173-front-landscape.mp4" \
  -i "$e/animated-results.mp4" \
  -loop 1 -framerate 60 -i "$e/landscape-results.jpg" \
  -i "$e/raw-seed-173-landscape.mp4" \
  -i "$e/raw-scene-orbit-landscape.mp4" \
  -loop 1 -framerate 60 -i "$e/landscape-outro.jpg" \
  -i "$e/raw-seed-17-side-landscape.mp4" \
  -i "$e/raw-seed-71-overhead-landscape.mp4" \
  -i "$e/raw-seed-173-front-landscape.mp4" \
  -i "$e/narration-educational.wav" \
  -filter_complex "
    [0:v]trim=duration=6,setpts=PTS-STARTPTS,fps=60,format=yuv420p[v0];
    [1:v]setpts=2*PTS,fps=60,trim=duration=12[v1];
    [2:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v2];
    [3:v]trim=duration=6,setpts=PTS-STARTPTS,fps=60,format=yuv420p[v3];
    [4:v]setpts=3*PTS,fps=60,trim=duration=12[v4];
    [5:v]fps=60,trim=duration=16,setpts=PTS-STARTPTS[v5];
    [6:v]trim=duration=8,setpts=PTS-STARTPTS,fps=60,format=yuv420p[v6];
    [7:v]setpts=3.3334*PTS,fps=60,trim=duration=10[v7];
    [8:v]setpts=1.3334*PTS,fps=60,trim=duration=8[v8];
    [9:v]setpts=4*PTS,fps=60,trim=duration=12[v9];
    [10:v]fps=60,trim=duration=18,setpts=PTS-STARTPTS[v10];
    [11:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v11];
    [12:v]fps=60,trim=duration=18,setpts=PTS-STARTPTS[v12];
    [13:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v13];
    [14:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v14];
    [15:v]fps=60,trim=duration=18,setpts=PTS-STARTPTS[v15];
    [16:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v16];
    [17:v]setpts=3.3334*PTS,fps=60,trim=duration=10[v17];
    [18:v]setpts=3.3334*PTS,fps=60,trim=duration=10[v18];
    [19:v]setpts=3.3334*PTS,fps=60,trim=duration=10[v19];
    [20:v]trim=start=10:duration=8,setpts=PTS-STARTPTS,fps=60[v20];
    [21:v]trim=duration=8,setpts=PTS-STARTPTS,fps=60,format=yuv420p[v21];
    [22:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v22];
    [23:v]setpts=1.6667*PTS,fps=60,trim=duration=10[v23];
    [24:v]trim=duration=12,setpts=PTS-STARTPTS,fps=60,format=yuv420p[v24];
    [25:v]setpts=2*PTS,fps=60,trim=duration=6[v25];
    [26:v]setpts=2*PTS,fps=60,trim=duration=6[v26];
    [27:v]setpts=2*PTS,fps=60,trim=duration=6[v27];
    [v0][v1][v2][v3][v4][v5][v6][v7][v8][v9][v10][v11][v12][v13][v14][v15][v16][v17][v18][v19][v20][v21][v22][v23][v24][v25][v26][v27]concat=n=28:v=1:a=0[outv];
    [28:a]aresample=48000,apad=pad_dur=8,afade=t=out:st=269:duration=5[outa]
  " \
  -map "[outv]" -map "[outa]" \
  -c:v libx264 -profile:v high -preset slow -crf 16 -pix_fmt yuv420p -r 60 \
  -c:a aac -ar 48000 -b:a 192k -movflags +faststart -shortest \
  "$e/episode-003-educational-4m.mp4"

ffprobe -v error \
  -show_entries stream=codec_name,profile,width,height,r_frame_rate,pix_fmt,sample_rate,bit_rate \
  -show_entries format=duration,size -of json "$e/episode-003-educational-4m.mp4"
