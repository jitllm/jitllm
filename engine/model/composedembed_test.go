package model

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestComposedEmbeddingDecoderOnCPU is the composed CPU arm of a decoder
// embedding model (qwen3-embedding, EmbeddingGemma): its readout -- the
// output norm, the pooling, the dense heads and the unit length
// (poolGraph) -- served by the composer once a text, every block served
// (a non-causal decoder's chunk one run both ways, its symmetric windows
// the graph's), held to main's vectors and to llama.cpp's at the reference
// gate's bound, and each arm's violation seen: the other pooling, and a
// non-causal decoder's rows attending causally.
func TestComposedEmbeddingDecoderOnCPU(t *testing.T) {
	ran := 0
	for _, g := range loadEmbedGoldens(t) {
		t.Run(g.Model, func(t *testing.T) {
			m := openEmbedModel(t, g)
			if m.IsEncoder() {
				t.Skip("an encoder: TestComposedEncoderOnCPU")
			}
			ran++
			ref := openEmbedModel(t, g)
			re, err := ref.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			defer re.Close()
			_, want := encodeTexts(t, ref, re, g)
			e, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			_, got := encodeTexts(t, m, e, g)
			ps, pd := e.st.CPUPoolSteps()
			bs, bd := e.st.CPUBlockSteps()
			t.Logf("%s: %d readouts served by the composed pool (declines %v); %d block runs served, declines %v",
				g.Model, ps, nonZero(pd[:]), bs, nonZero(bd[:]))
			if ps != len(g.Texts) || len(nonZero(pd[:])) != 0 {
				t.Fatalf("the composed pool served %d of %d readouts, declines %v", ps, len(g.Texts), nonZero(pd[:]))
			}
			if n := coveredBlocks(m); n != m.Cfg.NLayer {
				t.Fatalf("the front end states %d of %d blocks (block 0: %s, %v)", n, m.Cfg.NLayer, m.Uncovered(0), m.UncoveredErr(0))
			}
			// Every block of every chunk: a text longer than a chunk runs
			// its blocks once a chunk.
			if bs < m.Cfg.NLayer*len(g.Texts) || bs%m.Cfg.NLayer != 0 || len(nonZero(bd[:])) != 0 {
				t.Fatalf("%d block runs served (%d blocks, %d texts), declines %v", bs, m.Cfg.NLayer, len(g.Texts), nonZero(bd[:]))
			}
			c := worstCos(got, want)
			bound := embedDevCos
			if g.Model == "qwen3-embedding-0.6b" {
				bound = embedDevCosQuant
			}
			t.Logf("%s: worst cosine to main's %.9f (bound %v)", g.Model, c, bound)
			if !(c >= bound) {
				t.Errorf("cosine to main's %.9f < %v", c, bound)
			}
			llamaCos(t, g, got, "composed CPU")

			was := m.Cfg.Pooling
			m.Cfg.Pooling = jlm.PoolMean
			if was == jlm.PoolMean {
				m.Cfg.Pooling = jlm.PoolLast
			}
			ve, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			_, vbad := encodeTexts(t, m, ve, g)
			vs, vd := ve.st.CPUPoolSteps()
			// A mean over a text of several chunks is main's (DeclinePrefill).
			vs += vd[composer.DeclinePrefill]
			ve.Close()
			m.Cfg.Pooling = was
			cb := worstCos(vbad, want)
			t.Logf("%s: violation (pooled other than %v): worst cosine to main's %.6f, %d readouts served", g.Model, was, cb, vs)
			if vs != len(g.Texts) || cb >= bound {
				t.Fatalf("the gate cannot see the wrong pooling: cosine %.9f, %d composed readouts", cb, vs)
			}
			if !m.Cfg.NonCausal {
				return
			}
			// The blocks' violation: the rows attend causally, the graphs
			// restated so.
			m.Cfg.NonCausal = false
			m.front.restated++
			ce, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			_, cbad := encodeTexts(t, m, ce, g)
			cs, cd := ce.st.CPUBlockSteps()
			ce.Close()
			m.Cfg.NonCausal = true
			m.front.restated++
			cc := worstCos(cbad, want)
			t.Logf("%s: violation (causal rows): worst cosine to main's %.6f, %d block runs served", g.Model, cc, cs)
			if cs < m.Cfg.NLayer*len(g.Texts) || len(nonZero(cd[:])) != 0 || cc >= bound {
				t.Fatalf("the gate cannot see causal rows: cosine %.9f, %d composed block runs", cc, cs)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no decoder embedding model among the goldens: the gate proved nothing")
	}
}
