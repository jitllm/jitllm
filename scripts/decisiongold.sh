#!/bin/sh
# decisiongold.sh writes the decision models' llama.cpp goldens:
# engine/model/testdata/decision/<model>.<request>.llamacpp.json, the answer
# llama-server's POST /v1/systemone gives for each request on the same GGUF
# the gate converts (engine/model/decide_test.go).
#
#   JITLLM_LCPP=/path/to/llama-b11514  JITLLM_MODELS=/path/to/models  scripts/decisiongold.sh
#
# The sources: LiquidAI/d1-3B-GGUF d1-3B-Q8_0.gguf into $JITLLM_MODELS/d1/,
# ggml-org/lev-GGUF lev-Q8_0.gguf into $JITLLM_MODELS/lev/, ggml-org/Laya-GGUF
# Laya-BF16.gguf into $JITLLM_MODELS/laya/gguf/, ggml-org/Clef-Flash-GGUF
# Clef-Flash-Q8_0.gguf into $JITLLM_MODELS/clef/ and clefgold.py's fixture into
# $JITLLM_MODELS/clef/fixture/, LiquidAI/d1-omni-600M-GGUF d1-omni-600M-F16.gguf and
# mmproj-d1-omni-600M-F16.gguf into $JITLLM_MODELS/d1-omni/, and d1's mmproj-d1-3B-F16.gguf into
# $JITLLM_MODELS/d1/ for the picture requests (pictures.json,
# chatpicture.json), and ggml-org/tinyopenjev-for-testing-gguf
# tinyopenjev-for-testing-Q8_0.gguf and its mmproj into $JITLLM_MODELS/openjev/,
# and ggml-org/vllm-d3-edge-GGUF vllm-d3-edge-Q8_0.gguf and its mmproj into $JITLLM_MODELS/vllm-d3/,
# and ggml-org/Kev-4B-GGUF Kev-4B-BF16.gguf into $JITLLM_MODELS/kev/.
# llama.cpp b11514 (ggml-org/llama.cpp releases, llama-b11514-bin-ubuntu-x64)
# or newer.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/engine/model/testdata/decision
: "${JITLLM_LCPP:?JITLLM_LCPP names the llama.cpp build directory holding llama-server}"
: "${JITLLM_MODELS:?JITLLM_MODELS names the model directory}"
port=${PORT:-18089}

gold() { # gold <gguf> <name> [<mmproj> <request>...]
	gguf=$1 name=$2
	shift 2
	proj=
	reqs="support ambiguous"
	if [ $# -gt 0 ]; then
		proj="--mmproj $JITLLM_MODELS/$1"
		shift
		reqs=$*
	fi
	# shellcheck disable=SC2086 # $proj is two words or none
	"$here/scripts/cap" 12G -- taskset -c 0,2,4,6,8,10 "$JITLLM_LCPP/llama-server" \
		-m "$JITLLM_MODELS/$gguf" $proj --port "$port" -t 6 -c 8192 -b 4096 -ub 4096 >"$out/.srv.log" 2>&1 &
	pid=$!
	i=0
	until curl -s "localhost:$port/health" | grep -q ok; do
		i=$((i + 1))
		if [ $i -gt 300 ]; then
			echo "llama-server did not come up: $out/.srv.log" >&2
			kill $pid
			exit 1
		fi
		sleep 1
	done
	for req in $reqs; do
		curl -s "localhost:$port/v1/systemone" -H 'Content-Type: application/json' \
			-d @"$out/$req.json" >"$out/$name.$req.llamacpp.json"
		# The model as its file name: a golden carries no path of this box.
		sed -i 's#"model":"[^"]*/#"model":"#' "$out/$name.$req.llamacpp.json"
		echo "$name $req: $(head -c 200 "$out/$name.$req.llamacpp.json")"
	done
	# The cap wrapper may not pass a signal on; the server is named by its model.
	pkill -f "llama-server -m $JITLLM_MODELS/$gguf" || true
	wait $pid || true
	rm -f "$out/.srv.log"
}
# Every model, or those named: scripts/decisiongold.sh Clef-Flash-Q8_0
want() { [ $# -eq 0 ] || [ -z "$only" ] || case " $only " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
only="$*"
want d1-3B-Q8_0 && gold d1/d1-3B-Q8_0.gguf d1-3B-Q8_0
want d1-3B-Q8_0 && gold d1/d1-3B-Q8_0.gguf d1-3B-Q8_0 d1/mmproj-d1-3B-F16.gguf pictures chatpicture
want lev-Q8_0 && gold lev/lev-Q8_0.gguf lev-Q8_0
want Laya-BF16 && gold laya/gguf/Laya-BF16.gguf Laya-BF16
want Clef-Flash-Q8_0 && gold clef/Clef-Flash-Q8_0.gguf Clef-Flash-Q8_0
want clef-fixture && gold clef/fixture/clef-fixture.gguf clef-fixture
want d1-omni-600M-F16 && gold d1-omni/d1-omni-600M-F16.gguf d1-omni-600M-F16
want tinyopenjev-for-testing-Q8_0 && gold openjev/tinyopenjev-for-testing-Q8_0.gguf tinyopenjev-for-testing-Q8_0
want tinyopenjev-for-testing-Q8_0 && gold openjev/tinyopenjev-for-testing-Q8_0.gguf tinyopenjev-for-testing-Q8_0 openjev/mmproj-tinyopenjev-for-testing-Q8_0.gguf chatpicture
want d1-omni-600M-F16 && gold d1-omni/d1-omni-600M-F16.gguf d1-omni-600M-F16 d1-omni/mmproj-d1-omni-600M-F16.gguf pictures voice
want vllm-d3-edge-Q8_0 && gold vllm-d3/vllm-d3-edge-Q8_0.gguf vllm-d3-edge-Q8_0
want vllm-d3-edge-Q8_0 && gold vllm-d3/vllm-d3-edge-Q8_0.gguf vllm-d3-edge-Q8_0 vllm-d3/mmproj-vllm-d3-edge-Q8_0.gguf pictures chatpicture
want Kev-4B-BF16 && gold kev/Kev-4B-BF16.gguf Kev-4B-BF16
want Bespoke-Nimble-9B-v3-Q8_0 && gold nimble/Bespoke-Nimble-9B-v3-Q8_0.gguf Bespoke-Nimble-9B-v3-Q8_0
true
