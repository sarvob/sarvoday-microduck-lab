#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
art="$root/artifacts/011-vision-guided-goalkeeper"
out="$here/episode-006-goalkeeper-5min.mp4"

for path in \
  "$here/blender-arena-opener.mp4" "$here/narration.wav" \
  "$art/baseline-shot-1.mp4" "$art/baseline-shot-2.mp4" "$art/baseline-shot-3.mp4" \
  "$art/predictor-shot-1.mp4" "$art/predictor-shot-2.mp4" "$art/predictor-shot-3.mp4" \
  "$here/card-product-bet.png" "$here/card-vision-pipeline.png" "$here/card-training.png" \
  "$here/card-results.png" "$here/card-limitations.png" "$here/card-next.png" "$here/card-outro.png"; do
  [[ -s "$path" ]] || { printf 'missing: %s\n' "$path" >&2; exit 2; }
done

duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$here/narration.wav")"
tmp="$here/render-segments"
mkdir -p "$tmp"

encode_video() {
  local src="$1" seconds="$2" name="$3"
  ffmpeg -y -hide_banner -loglevel error -stream_loop -1 -i "$src" -t "$seconds" \
    -vf "fps=60,scale=2560:1440" -an -c:v h264_videotoolbox -b:v 20M \
    -maxrate 28M -bufsize 40M -pix_fmt yuv420p -r 60 -movflags +faststart "$tmp/$name.mp4"
}

encode_still() {
  local src="$1" seconds="$2" name="$3"
  ffmpeg -y -hide_banner -loglevel error -loop 1 -framerate 60 -i "$src" -t "$seconds" \
    -an -c:v h264_videotoolbox -b:v 20M -maxrate 28M -bufsize 40M \
    -pix_fmt yuv420p -r 60 -movflags +faststart "$tmp/$name.mp4"
}

encode_video "$here/blender-arena-opener.mp4" 10 00-opener
encode_still "$here/card-product-bet.png" 18 01-bet
encode_video "$art/predictor-shot-2.mp4" 20 02-hero
encode_still "$here/card-vision-pipeline.png" 22 03-vision
encode_video "$art/predictor-shot-1.mp4" 18 04-predictor
encode_video "$art/baseline-shot-1.mp4" 20 05-baseline-miss
encode_video "$art/baseline-shot-3.mp4" 18 06-baseline-block
encode_still "$here/card-product-bet.png" 12 07-bet-return
encode_video "$art/baseline-shot-2.mp4" 18 08-baseline
encode_video "$art/predictor-shot-2.mp4" 18 09-predictor
encode_still "$here/card-training.png" 20 10-training
encode_video "$art/baseline-shot-1.mp4" 12 11-match-baseline
encode_video "$art/predictor-shot-1.mp4" 12 12-match-predictor
encode_still "$here/card-results.png" 20 13-results
encode_video "$art/predictor-shot-1.mp4" 8 14-eval-one
encode_video "$art/predictor-shot-2.mp4" 10 15-eval-two
encode_video "$art/predictor-shot-3.mp4" 10 16-eval-three
encode_still "$here/card-limitations.png" 16 17-limitations
encode_still "$here/card-next.png" 15 18-next
encode_still "$here/card-outro.png" 15.637333 19-outro

list="$tmp/concat.txt"
: > "$list"
for segment in "$tmp"/[0-9][0-9]-*.mp4; do
  printf "file '%s'\n" "$segment" >> "$list"
done

ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$list" \
  -i "$here/narration.wav" -map 0:v:0 -map 1:a:0 -shortest -c:v copy \
  -af "highpass=f=70,lowpass=f=14500,afftdn=nf=-30,dynaudnorm=f=150:g=12,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000" \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$out"

printf '%s\n' "$out"
