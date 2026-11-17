package tok

import (
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// BenchmarkEncodeLong encodes a long text -- AGENTS.md, prose and code,
// repeated to a quarter of a mebibyte -- with each model's own vocabulary:
// the whole of tokenization, its pre-split and its merges, as a prompt asks
// for it. Run at one thread under perf stat, its instruction count is the
// work.
func BenchmarkEncodeLong(b *testing.B) {
	doc, err := os.ReadFile("../AGENTS.md")
	if err != nil {
		b.Fatal(err)
	}
	var s strings.Builder
	for s.Len() < 256<<10 {
		s.Write(doc)
	}
	text := s.String()
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-1.7B-Q4_K_M", "gpt-oss-20b-Q4_K_M"} {
		f, err := jlm.Open(testmodels.Path(name + ".jlm"))
		if err != nil {
			b.Fatalf("%s: %v", name, err)
		}
		vc := f.Vocab()
		f.Close()
		v, err := New(vc)
		if err != nil {
			b.Fatalf("%s: %v", name, err)
		}
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(len(text)))
			n := 0
			for b.Loop() {
				n = len(v.Encode(text, false))
			}
			b.ReportMetric(float64(n), "tokens")
		})
	}
}
