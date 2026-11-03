package composer_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUNGramHash is TestComposedCPUNGramHash on every GPU device
// present: OpNGramHash formed at a decode step's one row, over each of
// composer.NGramCases' windows, held word for word to composer.NGramRef, guard words
// past the outputs; the reset ignored and a head dropped must part.
func TestComposedGPUNGramHash(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			ran, reset, dropped := 0, 0, 0
			for _, c := range composer.NGramCases() {
				g, win, rows, next := composer.NGramGraph(c)
				_, codes, err := composer.Default().Form(composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
				if err != nil {
					t.Fatalf("%d-grams x%d: %v", c.N, c.Per, err)
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Fatal(err)
				}
				heads := (c.N - 1) * c.Per
				for i, w := range c.Wins {
					vals := map[graph.ValueID][]uint32{win: slices.Clone(w)}
					if err := runGraphValues(d, codes, op.kerns, g, vals); err != nil {
						op.close()
						t.Fatalf("%d-grams x%d window %d: %v", c.N, c.Per, i, err)
					}
					gotRows, gotNext := vals[rows][:heads], vals[next][:c.N-1]
					wantRows, wantNext := composer.NGramRef(w, c.Words, c.Per, c.Reset, -1)
					if !slices.Equal(gotRows, wantRows) || !slices.Equal(gotNext, wantNext) {
						op.close()
						t.Fatalf("%d-grams x%d reset %v, window %v: rows %v next %v, want %v %v", c.N, c.Per, c.Reset, w, gotRows, gotNext, wantRows, wantNext)
					}
					if r, _ := composer.NGramRef(w, c.Words, c.Per, !c.Reset, -1); !slices.Equal(r, gotRows) {
						reset++
					}
					if r, _ := composer.NGramRef(w, c.Words, c.Per, c.Reset, heads-1); heads > 1 && !slices.Equal(r, gotRows) {
						dropped++
					}
				}
				op.close()
				ran++
			}
			if reset == 0 || dropped == 0 {
				t.Fatalf("the reset ignored parts %d windows and a head dropped %d: the gate cannot see them", reset, dropped)
			}
			t.Logf("%d shapes held; the reset ignored parts %d windows, a head dropped %d", ran, reset, dropped)
		})
	}
}
