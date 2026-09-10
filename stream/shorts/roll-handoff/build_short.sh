#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
failure="$root/stream/episode-004/roll-failure-0.84s-seed-23.mp4"
success="$root/stream/episode-004/roll-evidence-3-seeds.mp4"
tmp="$here/segments"
mkdir -p "$tmp"

for f in "$failure" "$success" "$here"/{hook,win,decision,evidence,cta}.png "$here/narration.wav"; do
  [[ -s "$f" ]] || { printf 'missing: %s\n' "$f" >&2; exit 2; }
done

vertical="crop=810:1440:875:0,scale=1080:1920:flags=lanczos,eq=contrast=1.03:saturation=1.08"
encode_action() {
  local src="$1" seconds="$2" name="$3" overlay="$4" speed="${5:-1}"
  ffmpeg -y -hide_banner -loglevel error -stream_loop -1 -i "$src" -loop 1 -i "$overlay" -t "$seconds" \
    -filter_complex "[0:v]setpts=PTS/$speed,$vertical,fps=30[b];[b][1:v]overlay=0:0:format=auto" \
    -an -c:v h264_videotoolbox -b:v 12M -maxrate 18M -bufsize 24M \
    -pix_fmt yuv420p -r 30 -movflags +faststart "$tmp/$name.mp4"
}

encode_card() {
  local src="$1" seconds="$2" name="$3"
  ffmpeg -y -hide_banner -loglevel error -loop 1 -framerate 30 -i "$src" -t "$seconds" \
    -vf "scale=1080:1920" -an -c:v h264_videotoolbox -b:v 9M \
    -pix_fmt yuv420p -r 30 -movflags +faststart "$tmp/$name.mp4"
}

encode_action "$failure" 3.84 00-fail "$here/hook.png"
encode_action "$failure" 5.90 01-fail-replay "$here/hook.png" 0.65
encode_card "$here/decision.png" 3.60 02-decision
encode_action "$success" 12.67 03-three-tries "$here/win.png"
encode_card "$here/evidence.png" 5.80 04-evidence
encode_action "$success" 12.67 05-proof "$here/win.png"
encode_action "$success" 7.00 06-slow-win "$here/win.png" 0.52
encode_card "$here/cta.png" 5.00 07-cta

list="$tmp/concat.txt"
: > "$list"
for segment in "$tmp"/[0-9][0-9]-*.mp4; do
  printf "file '%s'\n" "$segment" >> "$list"
done

# Light pulse bed plus two small confirmation chimes; narration remains dominant.
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "sine=frequency=110:sample_rate=48000:duration=56.48" \
  -f lavfi -i "sine=frequency=740:sample_rate=48000:duration=0.16" \
  -f lavfi -i "sine=frequency=980:sample_rate=48000:duration=0.20" \
  -filter_complex "[0:a]volume=0.035,tremolo=f=2.0:d=0.35[bed];[1:a]adelay=13400|13400,volume=0.18[c1];[2:a]adelay=26400|26400,volume=0.16[c2];[bed][c1][c2]amix=inputs=3:duration=longest,afade=t=in:st=0:d=0.4,afade=t=out:st=54.5:d=1.8[m]" \
  -map "[m]" -c:a pcm_s16le "$here/music.wav"

ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$list" \
  -i "$here/narration.wav" -i "$here/music.wav" \
  -filter_complex "[1:a]adelay=550|550,apad=pad_dur=56.48,atrim=duration=56.48,asplit=2[n1][n2];[2:a][n1]sidechaincompress=threshold=0.025:ratio=6:attack=15:release=360[ducked];[ducked][n2]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-15.5:TP=-1.2:LRA=9[a]" \
  -map 0:v:0 -map "[a]" -t 56.48 -c:v copy -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart "$here/microduck-20ms-short.mp4"

printf '%s\n' "$here/microduck-20ms-short.mp4"
