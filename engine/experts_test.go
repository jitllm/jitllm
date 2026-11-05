package engine

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestEveryExpertIsADifferentExpert asserts that expert e's weights are not
// expert 0's, on both bank layouts. An expert is a view of a bank, and binding a
// view by name once replaced its slice with the whole bank, so every expert read
// expert 0's rows. It checks the layout rather than an answer: a real mixture's
// experts often differ too little to see (Qwen3-MOE-4x0.6B's are near-clones).
func TestEveryExpertIsADifferentExpert(t *testing.T) {
	for _, name := range []string{
		"tiny-qwen3moe-f32.gguf",       // F32 bank: views are byte slices
		"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", // Q4_K bank: views are nn.Packed.Row
	} {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !m.Cfg.MoE() {
				t.Fatalf("%s is not a mixture -- this gate proved nothing", name)
			}
			// Fault the block in first: Open touches no page, so an expert view
			// has no bytes until pageIn gives it some.
			if err := m.PageIn(0, &m.Layers[0]); err != nil {
				t.Fatalf("page in block 0: %v", err)
			}
			// And the experts separately: pageIn deliberately skips them, and
			// unread spans would compare equal.
			all := make([]int32, m.Cfg.NExpert)
			for i := range all {
				all[i] = int32(i)
			}
			// Every expert is its own page; ensureExperts binds the views and
			// the release keeps the pages held until the comparison is done.
			var hold model.ExpertHold
			err = m.EnsureExperts(0, all, &hold, &m.Layers[0])
			defer hold.Release()
			if err != nil {
				t.Fatalf("ensure experts: %v", err)
			}
			l := &m.Layers[0]
			if l.LegacyExperts {
				t.Skipf("%s stores experts as separate tensors, not a bank", name)
			}
			checked, packed := 0, 0
			for _, r := range []struct {
				what string
				pick func(int) *model.Tensor
			}{
				{"gate", func(i int) *model.Tensor { return &l.Experts[i].Gate }},
				{"up", func(i int) *model.Tensor { return &l.Experts[i].Up }},
				{"down", func(i int) *model.Tensor { return &l.Experts[i].Down }},
			} {
				a := r.pick(0)
				for i := 1; i < len(l.Experts); i++ {
					b := r.pick(i)
					switch {
					case a.Packed != nil && b.Packed != nil:
						packed++
						// An expert is a sheet: n_expert independent packed
						// matrices back to back, each starting at row 0 with the
						// sheet's own stride, so what must differ is the bytes.
						// Disjoint spans cannot alias.
						if b.Packed.Row != 0 {
							t.Errorf("%s: expert %d starts at packed row %d, want 0 -- "+
								"a sheet is addressed from its own base", r.what, i, b.Packed.Row)
						}
						if b.Packed.Stride != b.Rows {
							t.Errorf("%s: expert %d has stride %d, want %d (one sheet's rows)",
								r.what, i, b.Packed.Stride, b.Rows)
						}
						for _, sp := range []struct {
							name string
							x, y []byte
						}{
							{"qs", a.Packed.QS, b.Packed.QS},
							{"d", a.Packed.D, b.Packed.D},
							{"sc", a.Packed.SC, b.Packed.SC},
						} {
							if len(sp.y) == 0 && len(sp.x) == 0 {
								continue
							}
							if len(sp.x) != len(sp.y) {
								t.Errorf("%s: expert %d's %s span is %d bytes, expert 0's is %d",
									r.what, i, sp.name, len(sp.y), len(sp.x))
								continue
							}
							if &sp.x[0] == &sp.y[0] {
								t.Errorf("%s: expert %d's %s span IS expert 0's -- the sheets alias",
									r.what, i, sp.name)
							}
							if bytes.Equal(sp.x, sp.y) {
								t.Errorf("%s: expert %d's %s bytes equal expert 0's", r.what, i, sp.name)
							}
						}
					case a.Packed == nil && b.Packed == nil:
						if len(a.Data) != len(b.Data) {
							t.Errorf("%s: expert %d holds %d bytes, expert 0 holds %d",
								r.what, i, len(b.Data), len(a.Data))
						}
						if bytes.Equal(a.Data, b.Data) {
							t.Errorf("%s: expert %d's bytes are expert 0's -- the bank view was clobbered",
								r.what, i)
						}
					default:
						t.Fatalf("%s: expert 0 and %d disagree about being packed", r.what, i)
					}
					checked++
				}
			}
			if checked == 0 {
				t.Fatal("no expert pairs compared -- this gate proved nothing")
			}
			t.Logf("%s: %d expert pairs distinct (%d through the packed layout)", name, checked, packed)
		})
	}
}
