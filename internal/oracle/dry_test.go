package oracle

import (
	"math"
	"slices"
	"testing"
)

// TestDryMatchesTheOraclesCases: Dry over the GGUF oracle's own DRY cases
// (its sampling tests, a context of 1024): the logits the probabilities'
// logs, the history accepted, then the softmax, sorted, held to the cases'
// expected probabilities within their 1e-3.
func TestDryMatchesTheOraclesCases(t *testing.T) {
	for i, c := range []struct {
		probs    []float32
		last     []int32
		want     []float32
		mult     float32
		base     float32
		allowed  int
		lastN    int
		breakers [][]int32
	}{
		{[]float32{0.25, 0.25, 0.25, 0.25}, []int32{0, 1}, []float32{0.25, 0.25, 0.25, 0.25}, 1, 1.1, 2, 4, nil},
		{[]float32{0.25, 0.25, 0.25, 0.25}, []int32{0, 1, 2, 0, 1}, []float32{0.296923, 0.296923, 0.296923, 0.109232}, 1, 1.1, 2, 5, nil},
		{[]float32{0.2, 0.2, 0.2, 0.2, 0.2}, []int32{0, 1, 3, 4, 0, 1}, []float32{0.2, 0.2, 0.2, 0.2, 0.2}, 1, 1.1, 2, 6, [][]int32{{3}}},
		{[]float32{0.2, 0.2, 0.2, 0.2, 0.2}, []int32{0, 1, 2, 0, 1}, []float32{0.241818, 0.241818, 0.241818, 0.241818, 0.032727}, 2, 1.1, 2, 5, nil},
		{[]float32{0.2, 0.2, 0.2, 0.2, 0.2}, []int32{0, 1, 2, 3, 4, 0, 1}, []float32{0.2, 0.2, 0.2, 0.2, 0.2}, 1, 1.1, 4, 7, nil},
	} {
		v := make([]float32, len(c.probs))
		for j, p := range c.probs {
			v[j] = float32(math.Log(float64(p)))
		}
		br := map[int32][][]int32{}
		for _, b := range c.breakers {
			br[b[0]] = append(br[b[0]], b[1:])
		}
		Dry(v, c.last, c.mult, c.base, c.allowed, c.lastN, 1024, br)
		got := make([]float32, len(v))
		sum := 0.0
		for _, x := range v {
			sum += math.Exp(float64(x))
		}
		for j, x := range v {
			got[j] = float32(math.Exp(float64(x)) / sum)
		}
		slices.SortFunc(got, func(a, b float32) int {
			switch {
			case a > b:
				return -1
			case a < b:
				return 1
			}
			return 0
		})
		for j := range got {
			if math.Abs(float64(got[j]-c.want[j])) >= 1e-3 {
				t.Errorf("case %d: probabilities %v, want %v", i, got, c.want)
				break
			}
		}
	}
}
