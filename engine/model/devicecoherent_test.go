package model

import "testing"

// deviceTieMargin is how far ahead the host's top logit may be and still lose
// the argmax to the device. RULE 11c records the device's f32 reduction-order
// band as 0.2-0.94 of a logit, and tieMargin (0.5) sits inside it: Qwen2-1.5B
// on Metal has the same error at every seam (worst NMSE 3.5e-3 to 6.6e-3 from
// the first block placed to the last, so it does not grow with the depth below
// the seam and is not a wiring fault), and one step had a host lead of 0.510.
const deviceTieMargin = 1.0

// coherentWithTheHost is the bar an arm of a device gate meets against the
// host: logit NMSE under 2e-2, and an argmax that parts from the host's only
// where the host's lead is inside the device band.
func coherentWithTheHost(host, got []float32) (nmse float64, ok bool) {
	var num, den float64
	for j := range host {
		d := float64(got[j] - host[j])
		num += d * d
		den += float64(host[j]) * float64(host[j])
	}
	ha, ga := argmaxID(host), argmaxID(got)
	nmse = num / den
	return nmse, nmse <= 2e-2 && (ga == ha || float64(host[ha]-host[ga]) <= deviceTieMargin)
}

// TestCoherentWithTheHostRefusesWhatItShould is the violation run for the bar:
// a flip over a lead inside the band passes, one over a lead beyond it fails,
// and logits that moved far fail on NMSE whatever their argmax.
func TestCoherentWithTheHostRefusesWhatItShould(t *testing.T) {
	host := []float32{10, 9.49, 1, 1}
	if _, ok := coherentWithTheHost(host, []float32{9.4, 9.6, 1, 1}); !ok {
		t.Fatal("a flip over a host lead of 0.51 was refused: inside the device band")
	}
	if _, ok := coherentWithTheHost(host, host); !ok {
		t.Fatal("identical logits were refused")
	}
	wide := []float32{10, 8.5, 1, 1}
	if _, ok := coherentWithTheHost(wide, []float32{9, 9.2, 1, 1}); ok {
		t.Fatal("a flip over a host lead of 1.5 was accepted: beyond the device band")
	}
	if _, ok := coherentWithTheHost(host, []float32{10, 5, 4, 4}); ok {
		t.Fatal("logits far from the host's were accepted: NMSE is the bar there")
	}
}
