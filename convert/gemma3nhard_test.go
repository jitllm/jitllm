package convert

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGemma3nHardRowsAreTheEmbedders holds the rows synth-gemma3nv's tower
// states for its own tokens (mm.hard_embd) to transformers'
// Gemma3nMultimodalEmbedder's hard path, computed here in float64 from the
// tower file's own tensors: each row of the embedder's table over its root
// mean square times the norm's weight (hard_embedding_norm), through the
// projection (embedding_projection), over its root mean square unweighted
// (embedding_post_projection_norm); the offset is the tokens' first id. No
// end-to-end gate sees the last norm (the end-of-image row is one input of
// many), so this one holds it: the rows without it are the violation.
func TestGemma3nHardRowsAreTheEmbedders(t *testing.T) {
	src := testmodels.Path("mmproj-synth-gemma3nv.gguf")
	if !fileExists(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/gemma3nvisiongold.py) -- this gate proved nothing", src)
	}
	f, err := gguf.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	floats := func(name string) ([]float64, []uint64) {
		mt, ok := f.Get(name)
		if !ok {
			t.Fatalf("the tower file carries no %s", name)
		}
		n := uint64(1)
		for _, d := range mt.Dims {
			n *= d
		}
		v := make([]float32, n)
		if err := quant.Dequant32(mt.Type, f.Bytes(mt), v); err != nil {
			t.Fatal(err)
		}
		out := make([]float64, n)
		for i, x := range v {
			out[i] = float64(x)
		}
		return out, mt.Dims
	}
	tab, td := floats("mm.embedding.weight")
	nw, _ := floats("mm.hard_emb_norm.weight")
	proj, pd := floats("mm.input_projection.weight")
	k, n, m := int(td[0]), int(td[1]), int(pd[1])
	if len(nw) != k || int(pd[0]) != k {
		t.Fatalf("a table of %v, a norm of %d and a projection of %v", td, len(nw), pd)
	}
	eps := float64(float32(f.FloatKey("vision.attention.layer_norm_epsilon", 1e-6)))
	want, unnormed := make([]float64, n*m), make([]float64, n*m)
	for i := range n {
		x := tab[i*k : (i+1)*k]
		var ss float64
		for _, v := range x {
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(k)+eps)
		y := want[i*m : (i+1)*m]
		ss = 0
		for r := range m {
			var acc float64
			for j, v := range x {
				acc += proj[r*k+j] * v * inv * nw[j]
			}
			y[r], unnormed[i*m+r] = acc, acc
			ss += acc * acc
		}
		inv = 1 / math.Sqrt(ss/float64(m)+eps)
		for r := range y {
			y[r] *= inv
		}
	}
	s, err := sourceOf(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	var hard, offset []float32
	for i := range s.Tensors {
		switch s.Tensors[i].Role {
		case jlm.RoleVMNHardEmbd:
			if hard, err = vecOf(&s.Tensors[i]); err != nil {
				t.Fatal(err)
			}
		case jlm.RoleVMNHardOffset:
			if offset, err = vecOf(&s.Tensors[i]); err != nil {
				t.Fatal(err)
			}
		}
	}
	if len(hard) != n*m || len(offset) != 1 || offset[0] != 262144 {
		t.Fatalf("the tower states %d hard values for %d rows of %d, and the offset %v", len(hard), n, m, offset)
	}
	off := func(ref []float64) float64 {
		var worst float64
		for i, v := range hard {
			if math.IsNaN(float64(v)) {
				return math.Inf(1)
			}
			worst = max(worst, math.Abs(float64(v)-ref[i])/max(1, math.Abs(ref[i])))
		}
		return worst
	}
	e := off(want)
	t.Logf("%d hard rows of %d: the worst element %.3g from the embedder's, relative", n, m, e)
	if !(e < 1e-6) {
		t.Fatalf("the hard rows sit %.3g from the embedder's (bound 1e-6)", e)
	}
	if v := off(unnormed); !(v > 1e-3) {
		t.Fatalf("VIOLATION: the rows without their last norm sit %.3g from the tower's, inside the bound", v)
	}
}
