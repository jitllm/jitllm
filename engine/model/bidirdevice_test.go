package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestBidirRunOnDevice holds a device prefill of a prompt with an image to the
// host's, both bidirectional, on every backend present -- the device widens
// the run's rows' key counts (tier's bidir.go) where the host widens its own.
// The distance between the bidirectional and the causal answer on the host is
// the scale: the device must sit a decade inside it.
func TestBidirRunOnDevice(t *testing.T) {
	m, tw := openGemma3(t)
	defer m.Close()
	ts := tw.testState()
	e, err := ts.Encode(rainbow(tw.Cfg.ImageSz))
	if err != nil {
		t.Fatal(err)
	}
	emb := append([]float32(nil), e...)
	ts.Close()
	spans := func(bidir bool) []Span {
		return []Span{{Tokens: m.Vocab.Encode("Look:", true)}, {Embd: emb, Bidir: bidir},
			{Tokens: m.Vocab.Encode(" This picture shows", false)}}
	}
	prefill := func(g *tier.GPU, bidir bool) ([]float32, int) {
		sp := spans(bidir)
		s := m.NewState(SpanPositions(sp, m.Cfg.NEmbd) + 1)
		defer s.Close()
		if g != nil {
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
		}
		lg, err := s.PrefillMixed(sp...)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...), s.GPULayers()
	}
	host, _ := prefill(nil, true)
	causal, _ := prefill(nil, false)
	scale, _ := nmseOf(causal, host)
	t.Logf("host: causal image rows against bidirectional: logit NMSE %.3e", scale)
	ran := 0
	for _, spec := range stepDevices() {
		g, err := openTier(t, tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			dev, placed := prefill(g, true)
			if placed == 0 {
				t.Fatalf("no text block placed on %s: this arm proved nothing (%s)", spec, g.Err())
			}
			nm, worst := nmseOf(dev, host)
			t.Logf("%s, %d of %d blocks: logit NMSE %.3e, max|d| %.3f against the host", spec, placed,
				m.Cfg.NLayer, nm, worst)
			if math.IsNaN(nm) || nm > scale/10 {
				t.Errorf("%s reads NMSE %.3e against the host, where the mask itself is worth %.3e", spec, nm, scale)
			}
			cd, _ := prefill(g, false)
			vn, _ := nmseOf(cd, host)
			t.Logf("violation on %s, causal image rows: NMSE %.3e", spec, vn)
			if vn < 10*nm {
				t.Errorf("causal image rows on %s read NMSE %.3e against a clean %.3e: the gate "+
					"cannot see the mask", spec, vn, nm)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
