package model

import (
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestGraphFindsTheFFNSite: with the block graph, the device offers the FFN
// site's recipes where graph.Gate2In matched and was legal -- every block of
// llama's and qwen3's graphs -- and its tuned decode gives the logits of the
// model without the graph bit for bit; with the verdict turned to a refusal,
// the device offers nothing there (the graph decides, not ffnKey's
// structural list), so the counters fall to zero.
func TestGraphFindsTheFFNSite(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0"} {
		t.Run(name, func(t *testing.T) {
			p := graphModel(t, name)
			ctl, err := Open(p, noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer ctl.Close()
			gm, err := Open(p, noTune, WithBlockGraph(true))
			if err != nil {
				t.Fatal(err)
			}
			defer gm.Close()
			for li := 0; li < gm.Cfg.NLayer; li++ {
				// Every rewrite matches once and is legal on a dense block.
				ds := gm.BlockRewrites(li)
				if len(ds) != len(graph.Rewrites) {
					t.Fatalf("block %d: %v", li, ds)
				}
				for _, d := range ds {
					if d.Err != nil {
						t.Fatalf("block %d: %v", li, d)
					}
				}
			}
			ids := gm.Vocab.Encode("The capital of France is", true)
			toks := []int32{ids[1], ids[2], ids[3], ids[1], ids[2], ids[3], ids[1], ids[2]}
			arms := []tier.FFNForm{tier.FFNGate2In, tier.FFNTuned}
			// One verdict cache for both models: the split tuner times per
			// process, and two verdicts would be two reduction orders.
			cache := t.TempDir()
			want, _ := ffnRecipeDecode(t, ctl, ids, toks, false, -1, cache, arms)
			if want == nil {
				t.Skip("NO CUDA DEVICE: this gate proved nothing")
			}
			got, sts := ffnRecipeDecode(t, gm, ids, toks, false, -1, cache, arms)
			if sts[0].Gate2InRuns == 0 || sts[1].FFNTimed == 0 {
				t.Fatalf("the graph's site ran no gate2in (%d) or timed no key (%d)", sts[0].Gate2InRuns, sts[1].FFNTimed)
			}
			for a := range arms {
				for i := range want[a] {
					for j := range want[a][i] {
						if got[a][i][j] != want[a][i][j] {
							t.Fatalf("arm %v step %d logit %d: graph %v, without %v", arms[a], i, j, got[a][i][j], want[a][i][j])
						}
					}
				}
			}
			t.Logf("gate2in in %d FFNs; %d keys timed, %d kept", sts[0].Gate2InRuns, sts[1].FFNTimed, sts[1].FFNKept)
			// The violation: the graph refuses the site, and the tier offers
			// no recipe there although its structural checks would.
			// Two refusals: no gate2in region in the graph, and a region
			// whose Pays guard refuses at the device's Target -- which only
			// the tier, building that Target, can ask.
			saved := make([][]graph.Offered, len(gm.graphs))
			for li, lw := range gm.graphs {
				saved[li] = lw.offered
			}
			refuseAt := *graph.Gate2In
			var asked []graph.Target
			refuseAt.Pays = func(_ *graph.Graph, _ graph.Region, tg graph.Target) error {
				asked = append(asked, tg)
				return fmt.Errorf("refused at %s", tg.Key())
			}
			for _, arm := range []string{"no-region", "target-refuses"} {
				for li, lw := range gm.graphs {
					lw.offered = nil
					if arm == "target-refuses" {
						// gate2in's region under the refusing guard; the
						// prologue sites keep their own rewrites.
						for _, o := range saved[li] {
							if o.Rewrite == graph.Gate2In {
								o.Rewrite = &refuseAt
							}
							lw.offered = append(lw.offered, o)
						}
					}
				}
				_, sts = ffnRecipeDecode(t, gm, ids, toks, false, -1, t.TempDir(), []tier.FFNForm{tier.FFNTuned})
				if sts[0].FFNTimed != 0 || sts[0].Gate2InRuns != 0 {
					t.Fatalf("%s: with the site refused, %d keys timed and gate2in ran in %d FFNs", arm, sts[0].FFNTimed, sts[0].Gate2InRuns)
				}
				// The prologue sites: refused with no region, offered (and
				// so timed) where only gate2in's guard refuses.
				if prologue := sts[0].ActInTimed != 0; prologue != (arm == "target-refuses") {
					t.Fatalf("%s: the prologue tuner timed %d sites", arm, sts[0].ActInTimed)
				}
			}
			// The Target the tier built is the device's: its backend, an ISA
			// tier the code generator names, and the one-row decode class.
			if len(asked) == 0 {
				t.Fatal("the tier never asked the rewrite's Pays guard: no Target was built")
			}
			tg := asked[0]
			if tg.API != "ptx" || !strings.HasPrefix(tg.ISA, "sm_") || tg.Rows != graph.RowsDecode || tg.Subgroup != 32 || tg.SharedMem == 0 {
				t.Fatalf("the tier's Target %+v is not a CUDA device's at the decode row class", tg)
			}
			t.Logf("the tier's Target: %s", tg.Key())
			for li, lw := range gm.graphs {
				lw.offered = saved[li]
			}
		})
	}
}
