package gpu

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/nn"
)

// TestCompileFailureIsDeclined places one block on the fake device twice:
// once where every kernel compiles, the control, and once where the driver
// rejects every kernel it is handed (fakeDev.failCompile). The second block
// is declined, not placed, and LastErr names the compile that failed, so a
// card that cannot build a block's code says so where placement reads it.
func TestCompileFailureIsDeclined(t *testing.T) {
	p := fakePlan()
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	kv := p.NKVHead * p.HeadDim
	w := &nn.LayerWeights{
		AttnNorm: make([]float32, p.NEmbd), FFNNorm: make([]float32, p.NEmbd),
		Wq:   q4(p.NHead*p.HeadDim, p.NEmbd),
		Wk:   q4(kv, p.NEmbd),
		Wv:   q4(kv, p.NEmbd),
		Wo:   q4(p.NEmbd, p.NHead*p.HeadDim),
		Gate: q4(p.NFFN, p.NEmbd),
		Up:   q4(p.NFFN, p.NEmbd),
		Down: q4(p.NEmbd, p.NFFN),
	}
	run := func(fail bool) (bool, string, int) {
		d := &fakeDev{}
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		// Armed after New, so only the block's own kernels fail.
		d.failCompile = fail
		one := g.devs[0]
		ok := one.PrepLayer(0, p, w)
		return ok, one.LastErr, d.kerns
	}
	ok, why, kerns := run(false)
	if !ok || kerns == 0 {
		t.Fatalf("control: placed %v with %d kernels compiled (%q): the failing arm below proves nothing", ok, kerns, why)
	}
	ok, why, kerns = run(true)
	if ok {
		t.Fatal("a block none of whose kernels compiled was placed")
	}
	if kerns != 0 || !strings.Contains(why, "does not compile") {
		t.Fatalf("declined with %d kernels compiled and LastErr %q: the compile failure is not what it names", kerns, why)
	}
	t.Logf("declined: %s", why)
}
