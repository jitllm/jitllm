package model

import (
	"os"
	"testing"
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
		open func(t *testing.T) *Model
	}
	plain := func(name string) func(t *testing.T) *Model {
		return func(t *testing.T) *Model {
			if _, err := os.Stat(models()[name]); err != nil {
				t.Fatalf("MODEL MISSING: %v", err)
			}
			m, err := Open(jlmOf(t, models()[name]), noTune())
			if err != nil {
				t.Fatal(err)
			}
			return m
		}
	}
	for _, fx := range []fixture{
		{"stories15M", plain("stories15M")},
		{"qwen3moe", plain("qwen3moe")},
		{"qwen2vl", plain("qwen2vl")},
		// Per-layer inputs, and a learned position table.
		{"synth-gemma4-e", func(t *testing.T) *Model { m, _ := openGemma4(t, "synth-gemma4-e"); return m }},
		{"synth-starcoder", func(t *testing.T) *Model { m, _ := openC6(t, "synth-starcoder"); return m }},
	} {
		t.Run(fx.name, func(t *testing.T) {
			m := fx.open(t)
			defer m.Close()
			st := m.NewInstance(32)
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
			served, declined := st.CPUEmbedSteps()
			if served != want || len(nonZero(declined[:])) != 0 {
				t.Fatalf("the composed input rows served %d of %d rows (declines %v)", served, want, nonZero(declined[:]))
			}
			t.Logf("%d input rows from the composed graph", served)
		})
	}
}
