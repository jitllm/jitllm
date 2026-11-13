package engine

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/model"
)

// TestACutAtTheLeastPKeepsOneToken: a top-p or typical-p cut at the least
// normal binary32 keeps one token, the one llama.cpp's chain keeps at that
// p (oracle.Chain), as its cut at 0 or below keeps one: /completion asks for
// a p at or below 0 so. Every draw at temperature 1 is that token, whatever
// its uniform.
func TestACutAtTheLeastPKeepsOneToken(t *testing.T) {
	if _, err := os.Stat(models()["stories15M"]); err != nil {
		t.Fatalf("MODEL MISSING: %v", err)
	}
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const least = 0x1p-126
	for _, c := range []struct {
		name  string
		smp   Sampler
		stage string
	}{
		{"top-p", Sampler{Temp: 1, TopP: least, Seed: 3}, "top_p"},
		{"typical-p", Sampler{Temp: 1, TypicalP: least, Seed: 5}, "typ_p"},
	} {
		t.Run(c.name, func(t *testing.T) {
			st := testInstance(t, m, 64)
			defer st.Close()
			smp := c.smp
			lg, err := st.Prefill([]int32{1, 450, 3014, 471, 263})
			if err != nil {
				t.Fatal(err)
			}
			for i := range 16 {
				row, _ := oracle.Chain(lg, []oracle.ChainStage{{Name: c.stage, P: least}})
				want, kept := int32(-1), 0
				for id, v := range row {
					if !math.IsInf(float64(v), -1) {
						want, kept = int32(id), kept+1
					}
				}
				if kept != 1 {
					t.Fatalf("draw %d: llama.cpp's cut at %g keeps %d tokens", i, least, kept)
				}
				tok, err := st.Sample(&smp, lg)
				if err != nil {
					t.Fatal(err)
				}
				if tok != want {
					t.Fatalf("draw %d: drew %d, the cut keeps %d alone", i, tok, want)
				}
				if lg, err = st.Forward(tok); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
