package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/model"
)

// TestCPUInputRowsAreComposed requires every input row a CPU instance reads to
// come from the composed input rows' graph (cpuembed.go): each prompt row's
// and each decode token's, a position row and per-layer inputs among them,
// counted served and none declined. The values are the greedy and
// transformers gates'; this gate says that path is the one they ran
// (RULE 10).
func TestCPUInputRowsAreComposed(t *testing.T) {
	type fixture struct {
		name string
		open func(t *testing.T) (*model.Model, []Option)
	}
	plain := func(name string) func(t *testing.T) (*model.Model, []Option) {
		return func(t *testing.T) (*model.Model, []Option) {
			if _, err := os.Stat(models()[name]); err != nil {
				t.Fatalf("MODEL MISSING: %v", err)
			}
			m, err := model.Open(jlmOf(t, models()[name]))
			if err != nil {
				t.Fatal(err)
			}
			return m, nil
		}
	}
	for _, fx := range []fixture{
		{"stories15M", plain("stories15M")},
		{"qwen3moe", plain("qwen3moe")},
		{"qwen2vl", plain("qwen2vl")},
		// Per-layer inputs, and a learned position table.
		{"synth-gemma4-e", func(t *testing.T) (*model.Model, []Option) { m, _, o := openGemma4(t, "synth-gemma4-e"); return m, o }},
		{"synth-starcoder", func(t *testing.T) (*model.Model, []Option) { m, _, o := openC6(t, "synth-starcoder"); return m, o }},
	} {
		t.Run(fx.name, func(t *testing.T) {
			m, openOpts := fx.open(t)
			defer m.Close()
			st := testInstance(t, m, 32, openOpts...)
			defer st.Close()
			prompt := []int32{3, 7, 11, 2, 5}
			if _, err := st.Prefill(prompt); err != nil {
				t.Fatal(err)
			}
			want := len(prompt)
			for _, id := range []int32{9, 4, 6} {
				if _, err := st.Forward(id); err != nil {
					t.Fatal(err)
				}
				want++
			}
			served, declined := st.Account(nn.CPU).Steps.Of(nn.ComposedEmbed)
			if served != want || len(nonZero(declined[:])) != 0 {
				t.Fatalf("the composed input rows served %d of %d rows (declines %v)", served, want, nonZero(declined[:]))
			}
			t.Logf("%d input rows from the composed graph", served)
		})
	}
}
