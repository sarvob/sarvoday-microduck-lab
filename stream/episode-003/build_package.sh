#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
episode="$repo_root/stream/episode-003"
python3 "$episode/generate_graphics.py"

ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -framerate 60 -i "$episode/landscape-intro.jpg" \
  -i "$episode/raw-unassisted-landscape.mp4" \
  -loop 1 -framerate 60 -i "$episode/landscape-constraint.jpg" \
  -loop 1 -framerate 60 -i "$episode/landscape-method.jpg" \
  -i "$episode/raw-seed-17-landscape.mp4" \
  -i "$episode/raw-seed-71-landscape.mp4" \
  -i "$episode/raw-seed-173-landscape.mp4" \
  -loop 1 -framerate 60 -i "$episode/landscape-results.jpg" \
  -i "$episode/raw-seed-17-landscape.mp4" \
  -loop 1 -framerate 60 -i "$episode/landscape-outro.jpg" \
  -i "$episode/narration-landscape.wav" \
  -filter_complex "
    [0:v]trim=duration=7,setpts=PTS-STARTPTS,format=yuv420p[v0];
    [1:v]setpts=2.5*PTS,fps=60,trim=duration=10[v1];
    [2:v]trim=duration=10,setpts=PTS-STARTPTS,format=yuv420p[v2];
    [3:v]trim=duration=12,setpts=PTS-STARTPTS,format=yuv420p[v3];
    [4:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v4];
    [5:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v5];
    [6:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v6];
    [7:v]trim=duration=12,setpts=PTS-STARTPTS,format=yuv420p[v7];
    [8:v]setpts=2.6667*PTS,fps=60,trim=duration=8[v8];
    [9:v]trim=duration=12,setpts=PTS-STARTPTS,format=yuv420p[v9];
    [v0][v1][v2][v3][v4][v5][v6][v7][v8][v9]concat=n=10:v=1:a=0[outv];
    [10:a]aresample=48000,apad=pad_dur=4,afade=t=out:st=92:duration=3[outa]
  " \
  -map "[outv]" -map "[outa]" -c:v libx264 -profile:v high -preset slow -crf 16 \
  -pix_fmt yuv420p -r 60 -c:a aac -ar 48000 -b:a 192k -movflags +faststart -shortest \
  "$episode/episode-003-landscape.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -framerate 60 -i "$episode/portrait-physics-intro.jpg" \
  -i "$episode/raw-unassisted-portrait.mp4" \
  -loop 1 -framerate 60 -i "$episode/portrait-physics-metric.jpg" \
  -loop 1 -framerate 60 -i "$episode/portrait-physics-method.jpg" \
  -i "$episode/narration-short-physics.wav" \
  -filter_complex "
    [0:v]trim=duration=5,setpts=PTS-STARTPTS,format=yuv420p[v0];
    [1:v]setpts=2.5*PTS,fps=60,trim=duration=10[v1];
    [2:v]trim=duration=6,setpts=PTS-STARTPTS,format=yuv420p[v2];
    [3:v]trim=duration=8,setpts=PTS-STARTPTS,format=yuv420p[v3];
    [v0][v1][v2][v3]concat=n=4:v=1:a=0[outv];
    [4:a]aresample=48000,apad=pad_dur=3,afade=t=out:st=26:duration=3[outa]
  " \
  -map "[outv]" -map "[outa]" -c:v libx264 -profile:v high -preset slow -crf 16 \
  -pix_fmt yuv420p -r 60 -c:a aac -ar 48000 -b:a 192k -movflags +faststart -shortest \
  "$episode/episode-003-short-physics.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -loop 1 -framerate 60 -i "$episode/portrait-result-intro.jpg" \
  -i "$episode/raw-seed-17-portrait.mp4" \
  -i "$episode/raw-seed-71-portrait.mp4" \
  -i "$episode/raw-seed-173-portrait.mp4" \
  -loop 1 -framerate 60 -i "$episode/portrait-result-metric.jpg" \
  -loop 1 -framerate 60 -i "$episode/portrait-result-outro.jpg" \
  -i "$episode/narration-short-result.wav" \
  -filter_complex "
    [0:v]trim=duration=5,setpts=PTS-STARTPTS,format=yuv420p[v0];
    [1:v]setpts=1.6667*PTS,fps=60,trim=duration=5[v1];
    [2:v]setpts=1.6667*PTS,fps=60,trim=duration=5[v2];
    [3:v]setpts=1.6667*PTS,fps=60,trim=duration=5[v3];
    [4:v]trim=duration=6,setpts=PTS-STARTPTS,format=yuv420p[v4];
    [5:v]trim=duration=4,setpts=PTS-STARTPTS,format=yuv420p[v5];
    [v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[outv];
    [6:a]aresample=48000,apad=pad_dur=5,afade=t=out:st=27:duration=3[outa]
  " \
  -map "[outv]" -map "[outa]" -c:v libx264 -profile:v high -preset slow -crf 16 \
  -pix_fmt yuv420p -r 60 -c:a aac -ar 48000 -b:a 192k -movflags +faststart -shortest \
  "$episode/episode-003-short-result.mp4"

for output in \
  "$episode/episode-003-landscape.mp4" \
  "$episode/episode-003-short-physics.mp4" \
  "$episode/episode-003-short-result.mp4"
do
  ffprobe -v error -show_entries stream=codec_name,profile,width,height,r_frame_rate,pix_fmt,sample_rate,bit_rate \
    -show_entries format=duration,size -of json "$output"
done
