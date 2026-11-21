#!/bin/bash
# decisionbench-cell.sh: one cell of the decision-model board, ON the benchmark host
# (AGENTS.md RULE 1-2; the rows are in .notes/perf/current.md, "Decision models").
#
#   decisionbench-cell.sh cpu|cuda MODEL WINDOW_SECONDS ROUNDS
#
# Expects, in ~/dbr: jitllmd, decisionbench, centipedes-large.tsv, decisionbench-ab.sh,
# abstats.py, llama-cpu/ (a llama.cpp b11514 release, CPU) and llama.cpp/build/bin
# (b11514 built with -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70), and in
# /mnt/models/jitllm/dbr both MODEL.jlm (jitllm convert) and the MODEL.gguf it came
# from, which llama-server reads. The cell runs, serially: the A/A control (two
# jitllmd, same model), one full pass of the file per engine for accuracy, then two
# interleaved A/B runs of jitllmd against llama-server. Servers are pinned to one
# socket's cores 0-11 with numactl --membind=0, both at 12 threads.
set -u
be=$1 m=$2 win=$3 rounds=$4
cd ~/dbr
M=/mnt/models/jitllm/dbr
out=res/$be-$m; mkdir -p res
pin="numactl --membind=0 taskset -c 0-11"
if [ $be = cpu ]; then
  L=$HOME/dbr/llama-cpu; dev=cpu; lenv="LD_LIBRARY_PATH=$L"; lng="-ngl 0"; cuda=""
else
  L=$HOME/dbr/llama.cpp/build/bin; dev=cuda:0; lenv="LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=0"; lng="-ngl 99"; cuda="CUDA_VISIBLE_DEVICES=0"
fi
pids=""
cleanup() { for p in $pids; do kill -CONT $p 2>/dev/null; kill $p 2>/dev/null; done; sleep 2; }
trap cleanup EXIT
startj() { # port
  env $cuda $pin ./jitllmd serve -models $M -load $m.jlm -id $m -devices $dev -addr 127.0.0.1:$1 > $out.jd$1.log 2>&1 &
  echo $!
}
waitj() { for i in $(seq 1 200); do curl -s localhost:$1/v1/models | grep -q "\"$m\".*\"loaded\":true" && return 0; sleep 1; done; echo "jitllmd $1 not up"; exit 1; }
echo "load: $(cut -d' ' -f1-3 /proc/loadavg); psi: $(head -1 /proc/pressure/cpu)"
J1=$(startj 8081); J2=$(startj 8083); pids="$J1 $J2"
waitj 8081; waitj 8083
echo "jitllmd threads: $(ls /proc/$J1/task | wc -l)"; grep -i "cores\|pool\|tier\|workers" $out.jd8081.log | head -5
# AA first: jitllm vs jitllm
q='{"type":"noul","instructions":"Is this the name of a centipede?"}'
for p in 8081 8083; do ./decisionbench -url http://127.0.0.1:$p -model $m -file centipedes-large.tsv -cycle -window 5s -raw >/dev/null; done
echo "== AA jitllm/jitllm $be $m window=${win}s rounds=$rounds"
ARM_A=8081:$J1:$m ARM_B=8083:$J2:$m ./decisionbench-ab.sh ./decisionbench centipedes-large.tsv $rounds $win $out.aa.tsv >/dev/null
python3 abstats.py $out.aa.tsv $win
# accuracy: one full pass
kill -CONT $J1; kill -STOP $J2
echo "== accuracy jitllm (full pass, 233 words)"; ./decisionbench -url http://127.0.0.1:8081 -model $m -file centipedes-large.tsv -window 30m
kill -CONT $J2; kill $J2; wait $J2 2>/dev/null; pids="$J1"
env $lenv $pin $L/llama-server -m $M/$m.gguf --port 8082 --host 127.0.0.1 -t 12 -c 8192 -np 1 $lng > $out.ll.log 2>&1 &
LP=$!; pids="$J1 $LP"
for i in $(seq 1 300); do curl -s localhost:8082/health | grep -q ok && break; sleep 1; done
grep -i "CUDA\|backend\|ggml_cuda_init\|offloaded" $out.ll.log | head -4
kill -STOP $J1
echo "== accuracy llama.cpp (full pass)"; ./decisionbench -url http://127.0.0.1:8082 -model x -file centipedes-large.tsv -window 30m
kill -CONT $J1
for run in 1 2; do
  echo "== AB run $run: A=jitllm B=llama.cpp $be $m window=${win}s rounds=$rounds; load $(cut -d' ' -f1-3 /proc/loadavg) psi: $(head -1 /proc/pressure/cpu)"
  ARM_A=8081:$J1:$m ARM_B=8082:$LP:x ./decisionbench-ab.sh ./decisionbench centipedes-large.tsv $rounds $win $out.ab$run.tsv >/dev/null
  python3 abstats.py $out.ab$run.tsv $win
done
echo "end psi: $(head -1 /proc/pressure/cpu)"
