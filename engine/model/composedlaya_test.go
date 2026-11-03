package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestComposedLayaOnCPU is TestLayaMatchesReference's composed CPU arm:
// Laya's ModernBERT trunk -- its local blocks in their symmetric windows at
// their own rotary -- and its decision head's blocks, every one the front
// end states, served by the composer on the CPU device,
// held to the reference's logits at the host arm's tolerance, the served
// blocks counted (every block of every encode, none declined), and the
// rows made causal seen. The type rows and the scorer stay main's.
func TestComposedLayaOnCPU(t *testing.T) {
	for _, c := range layaCases() {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, jlm.DecisionLaya)
			defer m.Close()
			trunk, nb := m.enc.mb.trunk, m.encBlocks()
			if n := coveredBlocks(m); n != trunk {
				t.Fatalf("the front end states %d of %d trunk blocks (block 0: %s, %v)", n, trunk, m.Uncovered(0), m.UncoveredErr(0))
			}
			for li := trunk; li < nb; li++ {
				if u := m.Uncovered(li); u != Covered {
					t.Fatalf("decision head block %d: %s, %v", li, u, m.UncoveredErr(li))
				}
			}
			d, err := m.NewDecider(0)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			worst := func(d *Decider) float64 {
				w := 0.0
				for _, req := range decisionRequests() {
					var g layaGold
					b, err := os.ReadFile(filepath.Join("testdata", "decision", c.gold+"."+req+".reference.json"))
					if err != nil {
						t.Fatalf("%v (scripts/layagold.py writes it)", err)
					}
					if err := json.Unmarshal(b, &g); err != nil {
						t.Fatal(err)
					}
					state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
					for i := range qs {
						s, err := d.layaRun(state, &qs[i])
						if err != nil {
							t.Fatal(err)
						}
						want := g.Logits[qs[i].ID]
						if len(s) != len(want) {
							t.Fatalf("%s: %d scores, the reference %d", qs[i].ID, len(s), len(want))
						}
						for j := range s {
							if math.IsNaN(float64(s[j])) || math.IsInf(float64(s[j]), 0) {
								t.Fatalf("%s: score %d is %v", qs[i].ID, j, s[j])
							}
							w = max(w, math.Abs(float64(s[j])-want[j]))
						}
					}
				}
				return w
			}
			clean := worst(d)
			ref := openDecision(t, c.gguf, jlm.DecisionLaya)
			defer ref.Close()
			rd, err := ref.NewDecider(0)
			if err != nil {
				t.Fatal(err)
			}
			defer rd.Close()
			t.Logf("%s: main's host reads %.6f from the reference", c.name, worst(rd))
			seg := d.emb.seg
			served, declined := seg.CPUBlockSteps()
			t.Logf("%s: %d block runs served on the composed CPU (%d trunk and %d head blocks an encode), declines %v (form error %v); worst logit difference %.6f (tolerance %g)",
				c.name, served, trunk, nb-trunk, nonZero(declined[:]), seg.CPUFormErr(), clean, c.tol)
			if served == 0 || served%nb != 0 || len(nonZero(declined[:])) != 0 {
				t.Fatalf("the CPU served %d block runs (%d blocks an encode), declines %v, form error %v", served, nb,
					nonZero(declined[:]), seg.CPUFormErr())
			}
			if !(clean <= c.tol) {
				t.Errorf("worst logit difference %.6f from the reference > %g", clean, c.tol)
			}

			seg.enc.causalFault = true
			bad := worst(d)
			seg.enc.causalFault = false
			t.Logf("%s: violation (causal rows): worst logit difference %.6f", c.name, bad)
			if !(bad > 10*c.tol) {
				t.Fatalf("the gate cannot see causal rows: %.6f against a tolerance of %g", bad, c.tol)
			}
		})
	}
}
