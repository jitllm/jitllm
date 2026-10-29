package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// TestAVerdictDecidesWhatTheTierServes: a tier opened to serve by verdict
// takes the composer's parts for a model whose verdict covers its devices and
// not for one whose does not, per model as the tier changes hands, and a
// caller's own Composed or an off switch decides over the verdict.
func TestAVerdictDecidesWhatTheTierServes(t *testing.T) {
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	offer := func(g *GPU, model uint64, on nn.Backends) {
		t.Helper()
		p := fakePlan()
		p.Model, p.ComposedOn = model, on
		kv := p.NKVHead * p.HeadDim
		w := &nn.LayerWeights{
			AttnNorm: make([]float32, p.NEmbd), FFNNorm: make([]float32, p.NEmbd),
			Wq: q4(p.NHead*p.HeadDim, p.NEmbd), Wk: q4(kv, p.NEmbd), Wv: q4(kv, p.NEmbd),
			Wo: q4(p.NEmbd, p.NHead*p.HeadDim), Gate: q4(p.NFFN, p.NEmbd),
			Up: q4(p.NFFN, p.NEmbd), Down: q4(p.NEmbd, p.NFFN),
		}
		if !g.PrepLayer(0, p, w) {
			t.Fatalf("model %d was refused: %s", model, g.Err())
		}
	}
	open := func(opts ...Option) *GPU {
		t.Helper()
		g, err := New([]Slot{{Dev: &fakeDev{api: "ptx"}, Bytes: 1 << 40}}, append([]Option{WithDeviceTune(TuneOff)}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}

	g := open(WithComposedByVerdict())
	defer g.Close()
	offer(g, 1, nn.BackendPTX)
	if g.Composed != ComposedStep {
		t.Fatalf("a model ready on this backend is served with parts %v, want ComposedStep", g.Composed)
	}
	g.ReleaseModel(1)
	offer(g, 2, nn.BackendSPIRV)
	if g.Composed != 0 {
		t.Fatalf("a model ready on another backend only is served with parts %v", g.Composed)
	}
	g.ReleaseModel(2)
	offer(g, 3, 0)
	if g.Composed != 0 {
		t.Fatalf("a model with no verdict is served with parts %v", g.Composed)
	}

	for name, opt := range map[string]Option{
		"explicit parts": WithConfig(func(c *Config) { c.Composed = ComposedBlock }),
		"off":            WithConfig(func(c *Config) { c.ComposedAuto = false }),
	} {
		g := open(WithComposedByVerdict(), opt)
		offer(g, 1, nn.BackendPTX)
		want := Parts(0)
		if name == "explicit parts" {
			want = ComposedBlock
		}
		if g.Composed != want {
			t.Errorf("%s: parts %v, want %v", name, g.Composed, want)
		}
		g.Close()
	}

	plain := open()
	defer plain.Close()
	offer(plain, 1, nn.BackendPTX)
	if plain.Composed != 0 {
		t.Fatalf("a tier not opened to serve by verdict took parts %v", plain.Composed)
	}
}
