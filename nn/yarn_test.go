package nn

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestYarnFreqsMatchTransformers holds YarnFreqs to transformers' own
// _compute_yarn_parameters (scripts/yarngold.py) on gpt-oss's rotary and a
// second shape, with the correction range truncated and not: the inverse
// frequency base^(-2p/d)/Freqs[p] against transformers' inv_freq.
func TestYarnFreqsMatchTransformers(t *testing.T) {
	raw, err := os.ReadFile("testdata/yarn.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		HeadDim  int `json:"head_dim"`
		Base     float64
		Factor   float64
		Orig     int
		Truncate bool
		InvFreq  []float64 `json:"inv_freq"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no golden cases")
	}
	exactDiffers := false
	for _, c := range cases {
		f := YarnFreqs(c.HeadDim, c.Base, c.Factor, c.Orig, 32, 1, !c.Truncate)
		if len(f) != len(c.InvFreq) {
			t.Fatalf("%+v: %d pairs, golden has %d", c, len(f), len(c.InvFreq))
		}
		other := YarnFreqs(c.HeadDim, c.Base, c.Factor, c.Orig, 32, 1, c.Truncate)
		for p, want := range c.InvFreq {
			got := math.Pow(c.Base, -2*float64(p)/float64(c.HeadDim)) / float64(f[p])
			if math.Abs(got-want) > 2e-6*want {
				t.Fatalf("d=%d truncate=%v pair %d: inv_freq %.9g, transformers %.9g",
					c.HeadDim, c.Truncate, p, got, want)
			}
			if other[p] != f[p] {
				exactDiffers = true
			}
		}
	}
	// The control: truncating must change something, or the flag is untested.
	if !exactDiffers {
		t.Fatal("truncated and exact ranges give identical frequencies -- the golden cannot tell them apart")
	}
}
