package model

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestComposedD1OmniOnCPU is TestD1OmniMatchesReference's composed CPU
// arm: every block of d1-omni served by the composer on the CPU device --
// its LFM2 encoder's trunk (the centred convolution and the attention, each
// row inside its window: a media prefix's rows the prefix alone) and its
// decision head's (Laya's) -- every block of every encode counted, the
// logits held to the reference at the main gate's bound, and the head's
// rows made causal seen.
func TestComposedD1OmniOnCPU(t *testing.T) {
	m := openDecision(t, d1OmniModel, jlm.DecisionD1Omni)
	defer m.Close()
	trunk, nb := m.enc.mb.trunk, m.encBlocks()
	for li := range nb {
		if u := m.Uncovered(li); u != Covered {
			t.Fatalf("block %d (the trunk's %d): %s, %v", li, trunk, u, m.UncoveredErr(li))
		}
	}
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	encodes := 0
	worst := func() float64 {
		w := 0.0
		for _, req := range decisionRequests {
			g := readD1OmniGold(t, req)
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
			byID := map[string]*DecisionQuestion{}
			for i := range qs {
				byID[qs[i].ID] = &qs[i]
			}
			for _, gq := range g.Questions {
				q := byID[gq.ID]
				ids, markers, err := d.d1OmniSequence(state, q, d.d1OmniMaxLen())
				if err != nil {
					t.Fatal(err)
				}
				s, err := d.d1OmniForward(ids, q.Type, markers, 0)
				if err != nil {
					t.Fatal(err)
				}
				encodes++
				for j := range s {
					if math.IsNaN(float64(s[j])) || math.IsInf(float64(s[j]), 0) {
						t.Fatalf("%s: score %d is %v", q.ID, j, s[j])
					}
					w = max(w, math.Abs(float64(s[j])-gq.Logits[j]))
				}
			}
		}
		return w
	}
	clean := worst()
	seg := d.emb.seg
	served, declined := seg.CPUBlockSteps()
	t.Logf("d1-omni: %d block runs served on the composed CPU, declines %v (form error %v, arena %v, %d rows); worst logit difference %.2e (bound 3e-4)",
		served, nonZero(declined[:]), seg.CPUFormErr(), seg.nc().arena != nil, seg.maxSeq, clean)
	if served != nb*encodes || len(nonZero(declined[:])) != 0 {
		t.Fatalf("%d encodes: %d block runs served (want %d), declines %v", encodes, served, nb*encodes, nonZero(declined[:]))
	}
	if clean > 3e-4 {
		t.Errorf("worst logit difference %.2e from the reference (bound 3e-4)", clean)
	}

	seg.enc.causalFault = true
	bad := worst()
	seg.enc.causalFault = false
	t.Logf("d1-omni: violation (the head's rows causal): worst logit difference %.2e", bad)
	if !(bad > 3e-3) {
		t.Fatalf("the gate cannot see the head's rows made causal: %.2e", bad)
	}
}
