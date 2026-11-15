package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestAReaderHoldsNoHistory holds KV to being held once across blocks: a
// block reading another block's history (graph.Graph.ReadsHistoryOf) sizes
// and holds no page of its own after a prefill and a decode, while the
// block it reads holds its pages.
func TestAReaderHoldsNoHistory(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(*testing.T, string) (*model.Model, *c6Golden, []Option)
	}{{"synth-gemma3n", openGemma3n}, {"synth-gemma4-e", openGemma4}} {
		t.Run(f.name, func(t *testing.T) {
			m, _, opts := f.open(t, f.name)
			s := testInstance(t, m, 96, opts...)
			prompt := make([]int32, 40)
			for i := range prompt {
				prompt[i] = int32((7 + 13*i) % m.Cfg.NVocab)
			}
			lg, err := s.Prefill(prompt)
			if err != nil {
				t.Fatal(err)
			}
			for range 4 {
				if lg, err = s.Forward(int32(argmax(lg))); err != nil {
					t.Fatal(err)
				}
			}
			readers := 0
			for li := s.lo; li < s.hi; li++ {
				src := s.histSrc(li)
				if src == li {
					continue
				}
				readers++
				if pg := &s.kv.layers[li]; pg.pp != 0 || pg.mem.Len() != 0 {
					t.Errorf("block %d reads block %d's history and holds %d pages of %d floats of its own", li, src, pg.mem.Len(), pg.pp)
				}
				if s.kv.layers[src].mem.Len() == 0 {
					t.Errorf("block %d's history, which block %d reads, holds no page", src, li)
				}
			}
			if readers == 0 {
				t.Fatal("no block reads another's history: the gate selected nothing")
			}
		})
	}
}
