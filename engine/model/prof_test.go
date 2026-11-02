//go:build amd64 && linux

package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

func BenchmarkGemmaDecode(b *testing.B) {
	path := testmodels.Path("gemma-2b.gguf")
	if p := os.Getenv("JITLLM_BENCH_MODEL"); p != "" {
		path = testmodels.Resolve(p)
	}
	m, err := Open(jlmOf(b, path))
	if err != nil {
		b.Skip(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is", true)
	st := m.NewInstance(len(ids) + b.N + 2)
	defer st.Close()
	var lg []float32
	for _, id := range ids {
		if lg, err = st.Forward(id); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if lg, err = st.Forward(Greedy(lg)); err != nil {
			b.Fatal(err)
		}
	}
}
