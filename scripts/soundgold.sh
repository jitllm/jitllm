#!/bin/bash
# Record ffmpeg's mono 16 kHz f32 of fixed clips, once.
#   -> testdata/golden/sound/<clip>.wav and <clip>.f32
#
#   scripts/soundgold.sh            (ffmpeg on PATH; the goldens were made by ffmpeg 6.1.1)
#
# THE ORACLE FOR THE SOUND OP (graph.OpSound, composer.SWR, composer.MonoMix). An audio tower's processor reads
# 16 kHz mono samples, and transformers' audio pipeline makes them from a file with
# `ffmpeg -i - -ac 1 -ar 16000 -f f32le`: libswresample's rematrix to mono and its resampler at its defaults. A clip
# is held to the samples ffmpeg writes for it, not to a resampler that sounds right. Each clip is generated here
# (aevalsrc: tones, a chirp and noise, per channel), written as a WAV in its sample format and layout, and resampled
# by ffmpeg the way the pipeline asks.
set -e
cd "$(dirname "$0")/.."
D=testdata/golden/sound
mkdir -p $D
tone="0.4*sin(2*PI*440*t)+0.3*sin(2*PI*3100*t*(1+t))+0.1*random(0)-0.05"
mk() { # name rate codec layout seconds [atrim end_sample]
  local exprs=$tone n
  case $4 in mono) n=1 ;; stereo) n=2 ;; 2.1) n=3 ;; 4.0) n=4 ;; 5.0) n=5 ;; 5.1) n=6 ;; 6.1) n=7 ;; 7.1) n=8 ;; esac
  for ((c = 1; c < n; c++)); do exprs="$exprs|0.3*sin(2*PI*$((220 * (c + 1)))*t)+0.2*random($c)-0.1"; done
  local trim=()
  [ -n "$6" ] && trim=(-af "atrim=end_sample=$6")
  ffmpeg -hide_banner -loglevel error -y -f lavfi -i "aevalsrc=$exprs:s=$2:d=$5:c=$4" "${trim[@]}" -c:a "$3" $D/$1.wav
  ffmpeg -hide_banner -loglevel error -y -i $D/$1.wav -ac 1 -ar 16000 -f f32le $D/$1.f32
}
mk s16-16000-mono 16000 pcm_s16le mono 0.25
mk s16-16000-stereo 16000 pcm_s16le stereo 0.25
mk s16-44100-stereo 44100 pcm_s16le stereo 0.25
mk s16-48000-mono 48000 pcm_s16le mono 0.25
mk s16-8000-mono 8000 pcm_s16le mono 0.25
mk s16-22050-stereo 22050 pcm_s16le stereo 0.25
mk u8-11025-mono 11025 pcm_u8 mono 0.25
mk s24-96000-stereo 96000 pcm_s24le stereo 0.25
mk s32-32000-mono 32000 pcm_s32le mono 0.25
mk f32-44100-stereo 44100 pcm_f32le stereo 0.25
mk f64-24000-mono 24000 pcm_f64le mono 0.25
mk s16-16001-mono 16001 pcm_s16le mono 0.25
mk s16-22051-mono 22051 pcm_s16le mono 0.25
mk s16-7999-mono 7999 pcm_s16le mono 0.25
mk s16-48000-2.1 48000 pcm_s16le 2.1 0.1
mk s16-48000-4.0 48000 pcm_s16le 4.0 0.1
mk s16-48000-5.0 48000 pcm_s16le 5.0 0.1
mk s16-48000-5.1 48000 pcm_s16le 5.1 0.1
mk s16-48000-6.1 48000 pcm_s16le 6.1 0.1
mk s16-48000-7.1 48000 pcm_s16le 7.1 0.1
mk s16-44100-short60 44100 pcm_s16le mono 0.1 60
mk s16-44100-short91 44100 pcm_s16le mono 0.1 91
mk s16-44100-short100 44100 pcm_s16le mono 0.1 100
mk s16-44100-short141 44100 pcm_s16le mono 0.1 141
du -sh $D
