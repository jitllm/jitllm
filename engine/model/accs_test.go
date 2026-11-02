//go:build amd64 && linux

package model

import (
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestABKQuantAccsEndToEnd tests accumulator-chain differences in decode,
// at the shipping thread count. Each instance gets its own emitted kernel via
// nn.WithAccs set between NewInstance calls.
func TestABKQuantAccsEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("opens a model")
	}
	path := testmodels.Resolve(os.Getenv("JITLLM_BENCH_MODEL"))
	if path == "" {
		t.Skip("set JITLLM_BENCH_MODEL to a k-quant model")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	const rounds, iters = 21, 8
	ids := m.Vocab.Encode("The capital of France is", true)
	// One arm: an instance plus the logits it last produced. bench.AB alternates
	// them, so each advances its own position and its own KV cache.
	type arm struct {
		st *Instance
		lg []float32
	}
	mk := func(accs string) *arm {
		n, err := strconv.Atoi(accs)
		if err != nil || n < 1 || n > 4 {
			t.Fatalf("accs %q is not 1..4", accs)
		}
		// WithTune(TuneOff) keeps the pack tuner out of the measurement.
		m.jit = []nn.Option{nn.WithAccs(n), nn.WithTune(nn.TuneOff)}
		st := m.NewInstance(len(ids) + rounds*iters*2 + 8)
		a := &arm{st: st}
		for _, id := range ids {
			if a.lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return a
	}
	av, bv := os.Getenv("JITLLM_ACCS_A"), os.Getenv("JITLLM_ACCS_B")
	if av == "" {
		av, bv = "1", "2"
	}
	a, b := mk(av), mk(bv)
	defer a.st.Close()
	defer b.st.Close()

	step := func(x *arm) func(int) {
		return func(n int) {
			for i := 0; i < n; i++ {
				lg, err := x.st.Forward(Greedy(x.lg))
				if err != nil {
					// The window is sized for the whole run; running out
					// here would silently skew the comparison.
					t.Fatal(err)
				}
				x.lg = lg
			}
		}
	}
	bpt := m.BytesPerToken()
	r := bench.AB(
		bench.Case{Name: "accs" + av, BytesPerIter: bpt, Threads: 6, Fn: step(a)},
		bench.Case{Name: "accs" + bv, BytesPerIter: bpt, Threads: 6, Fn: step(b)},
		rounds, iters)
	t.Logf("%s", r)
	t.Logf("accs%s %.2f tok/s   accs%s %.2f tok/s   (%d B/token)",
		av, float64(iters)/r.AMedian.Seconds(),
		bv, float64(iters)/r.BMedian.Seconds(), bpt)
}
