package engine

import (
	"math"
	"slices"
	"testing"
)

// llamaNormalize is llama.cpp's common_embd_normalize over v in float64, the
// gate's reference: v as it is below 0, scaled into int16's range by its
// largest magnitude at 0, unit length at 2, over its k-norm otherwise; a
// norm of 0 gives zeros.
func llamaNormalize(v []float32, k int) []float64 {
	sum := 0.0
	switch {
	case k < 0:
		sum = 1
	case k == 0:
		for _, x := range v {
			sum = max(sum, math.Abs(float64(x)))
		}
		sum /= 32760
	case k == 2:
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		sum = math.Sqrt(sum)
	default:
		for _, x := range v {
			sum += math.Pow(math.Abs(float64(x)), float64(k))
		}
		sum = math.Pow(sum, 1/float64(k))
	}
	norm := 0.0
	if sum > 0 {
		norm = 1 / sum
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x) * norm
	}
	return out
}

// TestEmbedNormFollowsLlamaCpp holds each kind of llama.cpp's
// embd_normalize to llama.cpp's own arithmetic over the model's pooled row:
// the row as the readout leaves it (-1) has a length of its own and the
// default vector's direction, and every other kind is that row as
// common_embd_normalize leaves it, within the composed form's f32 rounding.
func TestEmbedNormFollowsLlamaCpp(t *testing.T) {
	ran := 0
	for _, g := range loadEmbedGoldens(t) {
		t.Run(g.Model, func(t *testing.T) {
			m, openOpts := openEmbedModel(t, g)
			e, err := NewEmbedder(m, openOpts...)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			embed := func(ids []int32, k int) []float32 {
				v, err := e.EmbedNorm(ids, k)
				if err != nil {
					t.Fatalf("embd_normalize %d: %v", k, err)
				}
				for _, x := range v {
					if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
						t.Fatalf("embd_normalize %d: a non-finite component", k)
					}
				}
				return slices.Clone(v)
			}
			for i, text := range g.Texts[:min(2, len(g.Texts))] {
				ids := m.EmbedIDs(text)
				unit, raw := embed(ids, NormUnit), embed(ids, -1)
				n := l2(raw)
				if math.Abs(n-1) < 1e-4 {
					t.Errorf("text %d: the row as the readout leaves it has length %.7f: it was normalised", i, n)
				}
				for j := range raw {
					if d := math.Abs(float64(raw[j])/n - float64(unit[j])); d > 1e-5 {
						t.Fatalf("text %d: the row over its length is %g from the default vector at %d", i, d, j)
					}
				}
				for _, k := range []int{0, 1, 2, 3, 6} {
					got, want := embed(ids, k), llamaNormalize(raw, k)
					big, worst := 0.0, 0.0
					for j := range want {
						big = max(big, math.Abs(want[j]))
						worst = max(worst, math.Abs(float64(got[j])-want[j]))
					}
					t.Logf("text %d, embd_normalize %d: worst %.3g of the largest component %.4g", i, k, worst/big, big)
					if !(worst <= 2e-5*big) {
						t.Errorf("text %d, embd_normalize %d: %.3g of the largest component from llama.cpp's", i, k, worst/big)
					}
				}
				ran++
			}
		})
	}
	if ran == 0 {
		t.Fatal("no golden text ran: this gate proved nothing")
	}
}
