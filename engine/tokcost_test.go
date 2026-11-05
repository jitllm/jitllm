package engine

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok"
)

// heapNow is the live heap after a full collection.
func heapNow() uint64 {
	runtime.GC()
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.HeapAlloc
}

// tokCostModels spans every tokenizer family: llama3, qwen2 and
// gpt2-style byte-level BPE, and SentencePiece in its llama, gemma and phi3
// forms.
func tokCostModels() []string {
	return []string{
		"Llama-3.2-1B-Instruct-Q4_K_M.jlm", "Qwen3-1.7B-Q4_K_M.jlm", "Qwen2-1.5B-Instruct-Q4_K_M.jlm",
		"olmoe-1b-7b-0924-instruct-Q4_K_M.jlm", "SmolLM2-360M-Instruct-Q8_0.jlm",
		"tinyllama-1.1b-q3_K_M.jlm", "gemma-3-1b-it-Q4_K_M.jlm", "gemma-2b.jlm",
		"Phi-3.5-mini-instruct-Q4_K_M.jlm", "stories15M-q8_0.jlm",
	}
}

// tokCostTexts is prose, code, and a mix built to reach the special-token
// scanner and the byte fallbacks: every family's specials, CJK, emoji, and the
// case-folding traps U+017F and U+212A.
func tokCostTexts(t *testing.T) []string {
	var texts []string
	for _, f := range []string{"../AGENTS.md", "forward.go"} {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		texts = append(texts, string(b))
	}
	mix := "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi there<|eot_id|>" +
		"<start_of_turn>model\nOK<end_of_turn><s> a </s><|im_start|>assistant\n<|im_end|>" +
		"<|endoftext|><|end|><|user|> 日本語のテキスト、中文字符 🎉🚀👍🏽 ÀÉÎõ ﬁ ſ K it'ſ " +
		"<<not special>> <|not_special|> \t\t  \n\n\n   x = 12345678 + 0.5e-3;\r\n"
	return append(texts, strings.Repeat(mix, 400))
}

// TestTokenizerCost prices Encode on real vocabularies over real text, so the
// question "is the tokenizer worth generating code for" is answered with a
// rate beside the prefill it feeds rather than with a guess. JITLLM_TOKCOST=1.
//
// JITLLM_TOKDUMP=<dir> also writes every id sequence there, so two builds can
// be compared token for token.
func TestTokenizerCost(t *testing.T) {
	if os.Getenv("JITLLM_TOKCOST") == "" {
		t.Skip("a measurement, not a gate: JITLLM_TOKCOST=1")
	}
	texts := tokCostTexts(t)
	dump := os.Getenv("JITLLM_TOKDUMP")
	for _, name := range tokCostModels() {
		if f := os.Getenv("JITLLM_TOKMODEL"); f != "" && !strings.Contains(name, f) {
			continue
		}
		m, err := model.Open(testmodels.Path(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		// What a loaded vocabulary keeps, and what building it costs: a second
		// build from the same container section, measured as live heap.
		base := heapNow()
		t0 := time.Now()
		v, err := tok.New(m.Container.Vocab())
		built := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		resident := heapNow() - base
		runtime.KeepAlive(v)
		t.Logf("%-38s vocab resident %6.2f MiB, built in %6.1f ms", name, float64(resident)/(1<<20),
			float64(built.Microseconds())/1e3)
		for i, text := range texts {
			var ds []time.Duration
			var ids []int32
			for r := 0; r < 5; r++ {
				t0 := time.Now()
				ids = m.Vocab.Encode(text, true)
				ds = append(ds, time.Since(t0))
			}
			slices.Sort(ds)
			d := ds[len(ds)/2]
			// Allocation of ONE Encode, from the runtime's own counters.
			var a, b runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&a)
			m.Vocab.Encode(text, true)
			runtime.ReadMemStats(&b)
			allocs, bytes := b.Mallocs-a.Mallocs, b.TotalAlloc-a.TotalAlloc
			t.Logf("%-38s text %d: %7d bytes -> %6d tokens in %8.2f ms = %8.0f tok/s; %8d allocs, %7.2f MiB "+
				"allocated = %5.2f allocs/token, %6.1f B/token, %4.1fx the text",
				name, i, len(text), len(ids), float64(d.Microseconds())/1e3, float64(len(ids))/d.Seconds(),
				allocs, float64(bytes)/(1<<20), float64(allocs)/float64(len(ids)),
				float64(bytes)/float64(len(ids)), float64(bytes)/float64(len(text)))
			if dump != "" {
				b := make([]byte, 4*len(ids))
				for j, id := range ids {
					binary.LittleEndian.PutUint32(b[4*j:], uint32(id))
				}
				if err := os.WriteFile(filepath.Join(dump, fmt.Sprintf("%s.%d.ids", name, i)), b, 0o644); err != nil {
					t.Fatal(err)
				}
			}
		}
		m.Close()
	}
}
