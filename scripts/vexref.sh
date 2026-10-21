#!/bin/sh
# Ground truth for jit_old/cpu/asm_test.go: assemble instructions with the system
# assembler and print their bytes. No one's memory is involved.
#
#   printf '%s\n' "vpaddd ymm1, ymm2, ymm3" | scripts/vexref.sh
#
# USE {vex} FOR ANY VNNI INSTRUCTION. GNU as defaults vpdpbusd to the EVEX
# (AVX-512-VNNI) encoding, exactly as Go's assembler does, and that encoding
# SIGILLs on this CPU -- verified: 62 f2 6d 28 50 cb faults, c4 e2 6d 50 cb runs.
set -e
tmp=$(mktemp -d); trap "rm -rf $tmp" EXIT
{ echo ".intel_syntax noprefix"; echo ".text"; while read -r ins; do
    [ -z "$ins" ] && continue
    echo "$ins"
  done; } > "$tmp/a.s"
as --64 -o "$tmp/a.o" "$tmp/a.s"
objdump -d -M intel "$tmp/a.o" | sed -n '/>:/,$p' | tail -n +2 | sed 's/^ *//' | grep -v '^$'
