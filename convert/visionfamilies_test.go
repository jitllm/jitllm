package convert

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

func f32Tensor(t *testing.T, role jlm.Role, block int32, vals []float32, dims ...uint64) jlm.Tensor {
	t.Helper()
	x := jlm.Tensor{Role: role, Block: block, Index: -1, Name: role.String()}
	setF32Dims(&x, vals, dims...)
	return x
}

func randVals(r *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// mlp is the tower MLP in float64: fc2 . gelu(fc1 . x + b1) + b2, with fc1
// [rows=f, k=e] and fc2 [rows=e, k=f] laid out k fastest.
func mlp(x, fc1, b1, fc2, b2 []float32, e, f int) []float64 {
	h := make([]float64, f)
	for r := 0; r < f; r++ {
		s := float64(b1[r])
		for j := 0; j < e; j++ {
			s += float64(fc1[r*e+j]) * float64(x[j])
		}
		h[r] = 0.5 * s * (1 + math.Tanh(0.7978845608*(s+0.044715*s*s*s)))
	}
	y := make([]float64, e)
	for r := 0; r < e; r++ {
		s := float64(b2[r])
		for j := 0; j < f; j++ {
			s += float64(fc2[r*f+j]) * h[j]
		}
		y[r] = s
	}
	return y
}

// TestPadFFNIsTheSameFunction widens a 40-wide MLP to 64 and demands the same
// output for every input: zero rows, a zero bias and zero columns add exactly
// nothing, which is the whole of padFFN's claim.
func TestPadFFNIsTheSameFunction(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const e, f = 32, 40
	fc1, b1, fc2, b2 := randVals(r, f*e), randVals(r, f), randVals(r, e*f), randVals(r, e)
	s := &jlm.Source{Vision: &jlm.Vision{NEmbd: e, NFFN: f}}
	s.Tensors = []jlm.Tensor{
		f32Tensor(t, jlm.RoleVFC1, 0, fc1, e, f), f32Tensor(t, jlm.RoleVFC1Bias, 0, b1, f),
		f32Tensor(t, jlm.RoleVFC2, 0, fc2, f, e), f32Tensor(t, jlm.RoleVFC2Bias, 0, b2, e),
	}
	if err := padFFN(s); err != nil {
		t.Fatal(err)
	}
	if s.Vision.NFFN != 64 {
		t.Fatalf("NFFN %d after padding, want 64", s.Vision.NFFN)
	}
	get := func(i int) []float32 {
		v, err := vecOf(&s.Tensors[i])
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	p1, pb1, p2, pb2 := get(0), get(1), get(2), get(3)
	if len(p1) != 64*e || len(pb1) != 64 || len(p2) != e*64 || len(pb2) != e {
		t.Fatalf("padded shapes %d %d %d %d", len(p1), len(pb1), len(p2), len(pb2))
	}
	for trial := 0; trial < 8; trial++ {
		x := randVals(r, e)
		want, got := mlp(x, fc1, b1, fc2, b2, e, f), mlp(x, p1, pb1, p2, pb2, e, 64)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("output %d is %v padded and %v not: the padding is not exact", i, got[i], want[i])
			}
		}
	}
}

// TestFoldLayerScalesScalesTheRowsTheyFollow checks InternViT's layer scales
// land on the attention output's rows and on the MLP contraction's -- found by
// shape, whatever the file calls it -- with their biases, and nowhere else.
func TestFoldLayerScalesScalesTheRowsTheyFollow(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	const e, f = 32, 64
	wo, bo := randVals(r, e*e), randVals(r, e)
	up, bup := randVals(r, f*e), randVals(r, f) // the expansion, named FC1
	dn, bdn := randVals(r, e*f), randVals(r, e) // the contraction, named FC2
	ls1, ls2 := randVals(r, e), randVals(r, e)
	s := &jlm.Source{Vision: &jlm.Vision{NLayer: 1, NEmbd: e, NFFN: f}}
	s.Tensors = []jlm.Tensor{
		f32Tensor(t, jlm.RoleVAttnOut, 0, wo, e, e), f32Tensor(t, jlm.RoleVAttnOutBias, 0, bo, e),
		f32Tensor(t, jlm.RoleVFC1, 0, up, e, f), f32Tensor(t, jlm.RoleVFC1Bias, 0, bup, f),
		f32Tensor(t, jlm.RoleVFC2, 0, dn, f, e), f32Tensor(t, jlm.RoleVFC2Bias, 0, bdn, e),
	}
	if err := foldLayerScales(s, map[int32][2][]float32{0: {ls1, ls2}}); err != nil {
		t.Fatal(err)
	}
	check := func(i int, orig, ls []float32, k int, what string) {
		v, err := vecOf(&s.Tensors[i])
		if err != nil {
			t.Fatal(err)
		}
		for j := range v {
			want := orig[j]
			if ls != nil {
				want *= ls[j/k]
			}
			if v[j] != want {
				t.Fatalf("%s element %d is %v, want %v", what, j, v[j], want)
			}
		}
	}
	check(0, wo, ls1, e, "attn_out")
	check(1, bo, ls1, 1, "attn_out bias")
	check(2, up, nil, e, "the expansion (must be untouched)")
	check(3, bup, nil, 1, "the expansion's bias (must be untouched)")
	check(4, dn, ls2, f, "the contraction")
	check(5, bdn, ls2, 1, "the contraction's bias")

	if err := foldLayerScales(s, map[int32][2][]float32{0: {ls1, nil}}); err == nil {
		t.Error("a block with one layer scale of two folded")
	}
}

// TestGemma3ProjectionIsTransposed lays the [text, tower] source matrix out k
// first, element for element.
func TestGemma3ProjectionIsTransposed(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const tower, text = 32, 48
	src := randVals(r, tower*text) // ggml [text, tower]: tower rows of text values
	s := &jlm.Source{Vision: &jlm.Vision{NEmbd: tower, ProjDim: text, Projector: jlm.ProjGemma3}}
	s.Tensors = []jlm.Tensor{f32Tensor(t, jlm.RoleVProj, jlm.DenseBlock, src, text, tower)}
	if err := transposeProjection(s); err != nil {
		t.Fatal(err)
	}
	got, err := vecOf(&s.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	if d := s.Tensors[0].Dims; d[0] != tower || d[1] != text {
		t.Fatalf("dims %v, want k=%d rows=%d", d[:2], tower, text)
	}
	for row := 0; row < text; row++ {
		for k := 0; k < tower; k++ {
			if got[row*tower+k] != src[k*text+row] {
				t.Fatalf("row %d k %d: %v, want %v", row, k, got[row*tower+k], src[k*text+row])
			}
		}
	}
}

// TestPermuteMergerIsTheSameFunction holds Mistral 3's merger, permuted into
// the shuffle's order, to the reference's product on the reference's order:
// the unfold's channel-major group read by the source matrix equals the
// shuffle's patch-major group read by the permuted one, exactly. A merger
// left unpermuted fails it.
func TestPermuteMergerIsTheSameFunction(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const d, s2 = 32, 4
	w := randVals(r, d*d*s2)
	for _, permute := range []bool{true, false} {
		src := &jlm.Source{Vision: &jlm.Vision{NEmbd: d, Scale: 2}}
		src.Tensors = []jlm.Tensor{f32Tensor(t, jlm.RoleVMerge, jlm.DenseBlock, w, d*s2, d)}
		if permute {
			if err := permuteMerger(src); err != nil {
				t.Fatal(err)
			}
		}
		pw, err := vecOf(&src.Tensors[0])
		if err != nil {
			t.Fatal(err)
		}
		patches := randVals(r, s2*d) // patch p's channels at p*d
		exact := true
		for o := 0; o < d; o++ {
			var want, got float32
			for c := 0; c < d; c++ {
				for p := 0; p < s2; p++ {
					want += w[o*d*s2+c*s2+p] * patches[p*d+c]
				}
			}
			for k := 0; k < d*s2; k++ {
				got += pw[o*d*s2+k] * patches[k]
			}
			if math.Abs(float64(got-want)) > 1e-4*math.Abs(float64(want))+1e-5 {
				exact = false
			}
		}
		if exact != permute {
			t.Fatalf("permuted %v: the merger reads the shuffle's rows as the reference reads the unfold's: %v",
				permute, exact)
		}
	}
}
