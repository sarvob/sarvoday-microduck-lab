#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
story="$root/artifacts/011-vision-guided-goalkeeper/story-v3"
tmp="$here/render-segments"
mkdir -p "$tmp"

for f in "$here"/narration-parts/{01,02,03,04,05,06,07}.wav \
  "$here/soundtrack.wav" \
  "$here/stock-broll/pexels-9502506.mp4" \
  "$here/stock-broll/pexels-6084018.mp4" \
  "$here"/{constraints,training,result}.png \
  "$here"/{cold-title,baseline-title,predictor-title,final-test}.png \
  "$story"/{baseline-hard-miss,baseline-center-save,baseline-wide-save,predictor-hard-save,predictor-center-save,predictor-wide-save}.mp4; do
  [[ -s "$f" ]] || { printf 'missing: %s\n' "$f" >&2; exit 2; }
done

# Space the short narration beats so motion, music, and crowd sounds get room.
ffmpeg -y -hide_banner -loglevel error \
  -i "$here/narration-parts/01.wav" -i "$here/narration-parts/02.wav" \
  -i "$here/narration-parts/03.wav" -i "$here/narration-parts/04.wav" \
  -i "$here/narration-parts/05.wav" -i "$here/narration-parts/06.wav" \
  -i "$here/narration-parts/07.wav" \
  -filter_complex "[0:a]adelay=3000|3000[a0];[1:a]adelay=40000|40000[a1];[2:a]adelay=91000|91000[a2];[3:a]adelay=135000|135000[a3];[4:a]adelay=180000|180000[a4];[5:a]adelay=225000|225000[a5];[6:a]adelay=291000|291000[a6];[a0][a1][a2][a3][a4][a5][a6]amix=inputs=7:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=9,apad=pad_dur=318,atrim=duration=318[n]" \
  -map "[n]" -ar 48000 -c:a pcm_s16le "$here/narration-v2.wav"

encode_video() {
  local src="$1" seconds="$2" name="$3" start="${4:-0}"
  ffmpeg -y -hide_banner -loglevel error -stream_loop -1 -ss "$start" -i "$src" -t "$seconds" \
    -vf "fps=60,scale=2560:1440,eq=contrast=1.04:saturation=1.08" -an \
    -c:v h264_videotoolbox -b:v 18M -maxrate 26M -bufsize 36M \
    -pix_fmt yuv420p -r 60 -movflags +faststart "$tmp/$name.mp4"
}

encode_overlay() {
  local src="$1" overlay="$2" seconds="$3" name="$4" start="${5:-0}"
  ffmpeg -y -hide_banner -loglevel error -stream_loop -1 -ss "$start" -i "$src" \
    -loop 1 -i "$overlay" -t "$seconds" \
    -filter_complex "[0:v]fps=60,scale=2560:1440,eq=contrast=1.05:saturation=1.08[b];[b][1:v]overlay=0:0:format=auto" \
    -an -c:v h264_videotoolbox -b:v 18M -maxrate 26M -bufsize 36M \
    -pix_fmt yuv420p -r 60 -movflags +faststart "$tmp/$name.mp4"
}

encode_still() {
  local src="$1" seconds="$2" name="$3"
  ffmpeg -y -hide_banner -loglevel error -loop 1 -framerate 60 -i "$src" -t "$seconds" \
    -vf "scale=2560:1440" -an -c:v h264_videotoolbox -b:v 18M \
    -maxrate 26M -bufsize 36M -pix_fmt yuv420p -r 60 -movflags +faststart "$tmp/$name.mp4"
}

encode_portrait_broll() {
  local src="$1" seconds="$2" name="$3" start="${4:-0}"
  ffmpeg -y -hide_banner -loglevel error -stream_loop -1 -ss "$start" -i "$src" -t "$seconds" \
    -filter_complex "[0:v]split=2[bg][fg];[bg]scale=2560:1440:force_original_aspect_ratio=increase,crop=2560:1440,boxblur=28:2,eq=brightness=-0.18:saturation=0.75[back];[fg]scale=-2:1320[front];[back][front]overlay=(W-w)/2:(H-h)/2,eq=contrast=1.04:saturation=1.08,fps=60" \
    -an -c:v h264_videotoolbox -b:v 18M -maxrate 26M -bufsize 36M \
    -pix_fmt yuv420p -r 60 -movflags +faststart "$tmp/$name.mp4"
}

encode_overlay "$here/stock-broll/pexels-9502506.mp4" "$here/cold-title.png" 8 00-cold
encode_overlay "$story/baseline-hard-miss.mp4" "$here/cold-title.png" 12 01-the-bet
encode_overlay "$story/predictor-hard-save.mp4" "$here/predictor-title.png" 20 02-can-it-save
encode_still "$here/constraints.png" 8 03-constraints
encode_video "$story/predictor-center-save.mp4" 15 04-pivot 0.4
encode_video "$story/predictor-wide-save.mp4" 15 05-camera 0
encode_video "$story/baseline-center-save.mp4" 13 06-almost-right 0.5
encode_overlay "$story/baseline-center-save.mp4" "$here/baseline-title.png" 15 07-baseline-good 1.1
encode_video "$story/baseline-wide-save.mp4" 15 08-baseline-more 0
encode_video "$story/baseline-hard-miss.mp4" 14 09-baseline-fails 0
encode_overlay "$story/predictor-hard-save.mp4" "$here/predictor-title.png" 20 10-predictor-reveal 0
encode_video "$story/predictor-center-save.mp4" 15 11-same-walk 1.5
encode_video "$story/predictor-hard-save.mp4" 10 12-contact 1.8
encode_still "$here/training.png" 8 13-training-board
encode_video "$story/baseline-center-save.mp4" 12 14-training-reactive 0
encode_video "$story/predictor-hard-save.mp4" 12 15-training-predict 0
encode_video "$story/predictor-center-save.mp4" 13 16-score-hidden 0.7
encode_overlay "$story/predictor-wide-save.mp4" "$here/final-test.png" 12 17-final-test
encode_video "$story/predictor-hard-save.mp4" 12 18-final-hard 0.1
encode_video "$story/predictor-center-save.mp4" 8 19-final-center 1.7
encode_video "$story/predictor-wide-save.mp4" 8 20-final-wide 0
encode_video "$story/predictor-hard-save.mp4" 8 21-final-hard-replay 2.0
encode_video "$story/predictor-center-save.mp4" 8 22-final-center-replay 0.3
encode_video "$story/predictor-wide-save.mp4" 10 23-final-wide-replay 1.2
encode_still "$here/result.png" 12 24-result
encode_video "$story/predictor-hard-save.mp4" 10 25-victory 0.5
encode_portrait_broll "$here/stock-broll/pexels-6084018.mp4" 5 26-keeper-button 1.0

list="$tmp/concat-v2.txt"
: > "$list"
for segment in "$tmp"/[0-9][0-9]-*.mp4; do
  printf "file '%s'\n" "$segment" >> "$list"
done

ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$list" \
  -i "$here/narration-v2.wav" -i "$here/soundtrack.wav" \
  -filter_complex "[2:a]volume=0.20[m];[1:a]asplit=2[n1][n2];[m][n1]sidechaincompress=threshold=0.025:ratio=7:attack=18:release=420[ducked];[ducked][n2]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-15.5:TP=-1.2:LRA=10[a]" \
  -map 0:v:0 -map "[a]" -t 318 -c:v copy -c:a aac -b:a 256k -ar 48000 \
  -movflags +faststart "$here/episode-006-goalkeeper-story-v4.mp4"

printf '%s\n' "$here/episode-006-goalkeeper-story-v4.mp4"
