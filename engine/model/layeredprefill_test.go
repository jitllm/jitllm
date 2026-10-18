//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestLayeredPrefillPagesEachBlockOncePerChunk prefills a prompt of two
// device pieces on a card whose slots hold fewer blocks than the model has,
// every block streamed through them. The chunk goes layered (GPU.layered): each
// block group pages in once and both pieces run through it. The selection
// checks are Stats.LayeredPieces (the path ran) and PageIns (each block
// crossed once, not once per piece); the reference is the host's prompt. The
// violation pieceouter pages the run in again for every piece and must read
// more page-ins with the same logits. A dense model and qwen4exp (four
// stream-major streams, cut by cutRows) both run it.
func TestLayeredPrefillPagesEachBlockOncePerChunk(t *testing.T) {
	for _, name := range []string{"synth-phi2", "synth-qwen4exp"} {
		t.Run(name, func(t *testing.T) {
			m := openSynth(t, name)
			defer m.Close()
			nl := m.Cfg.NLayer
			base := m.Vocab.Encode("The river rose every spring until the bridges were islands, "+
				"and the clerk counted barrels of salt on the upper floor. ", true)
			var ids []int32
			for len(ids) < 700 {
				ids = append(ids, base...)
			}
			ids = ids[:700]

			host := m.NewState(len(ids) + 4)
			want, err := host.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			want = slices.Clone(want)
			host.Close()

			ran := 0
			for _, spec := range stepDevices() {
				// The budget that holds all the weights, then smaller ones until
				// the card places every block and holds fewer than nl at once.
				open := func(budget uint64) (*tier.GPU, *State) {
					opts := append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)
					if budget > 0 {
						opts = append(opts, tier.WithBudget(budget))
					}
					g, err := openTier(t, opts...)
					if err != nil || g == nil {
						return nil, nil
					}
					for li := 0; li < nl; li++ {
						g.Stream(li, true)
					}
					st := m.NewState(len(ids) + 4)
					st.SetDeviceLayers(g, -1)
					return g, st
				}
				g, st := open(0)
				if g == nil {
					t.Logf("%s: not present", spec)
					continue
				}
				full := g.Stats().BudgetUsed
				st.Close()
				g.Close()
				var found bool
				for k := 1; k < 64 && !found; k++ {
					budget := full - uint64(k)*m.WeightBytes()/64
					g, st = open(budget)
					if g != nil {
						t.Logf("%s: budget %d (full %d, weights %d): %d of %d blocks, layered x%d", spec, budget, full, m.WeightBytes(), st.GPULayers(), nl, g.LayeredChunks())
					}
					if g != nil && st.GPULayers() == nl && g.LayeredChunks() > 1 {
						found = true
						break
					}
					if g != nil {
						st.Close()
						g.Close()
					}
				}
				if !found {
					t.Logf("%s: no budget places every block with fewer slots than blocks", spec)
					continue
				}
				ran++
				run := func(fault string) ([]float32, tier.Stats) {
					g.SetFault("rows", fault)
					defer g.SetFault("rows", "")
					s := m.NewState(len(ids) + 4)
					defer s.Close()
					s.SetDeviceLayers(g, -1)
					t.Logf("prompt chunk %d, runs paging, layered x%d", s.PromptChunk(), g.LayeredChunks())
					s0 := g.Stats()
					lg, err := s.Prefill(ids)
					if err != nil {
						t.Fatal(err)
					}
					s1 := g.Stats()
					return slices.Clone(lg), tier.Stats{PageIns: s1.PageIns - s0.PageIns, LayeredPieces: s1.LayeredPieces - s0.LayeredPieces}
				}
				st.Close()
				got, ds := run("")
				nm, _ := nmse32(want, got)
				bad, dsBad := run("pieceouter")
				nmBad, _ := nmse32(want, bad)
				t.Logf("%s: %d blocks, %d layered pieces, %d page-ins, NMSE %.3e; pieces outer: %d page-ins, NMSE %.3e",
					spec, nl, ds.LayeredPieces, ds.PageIns, nm, dsBad.PageIns, nmBad)
				if ds.LayeredPieces == 0 {
					t.Fatalf("the layered path never ran (%q)", g.Err())
				}
				if math.IsNaN(nm) || nm > 1e-3 || Greedy(got) != Greedy(want) {
					t.Errorf("layered prefill: NMSE %.3e against the host, argmax %d, want %d", nm, Greedy(got), Greedy(want))
				}
				if ds.PageIns > nl {
					t.Errorf("%d page-ins for %d blocks: a block crossed more than once per chunk", ds.PageIns, nl)
				}
				if dsBad.PageIns <= ds.PageIns {
					t.Errorf("the violation (pieces outer) reads %d page-ins against %d: the gate cannot see it", dsBad.PageIns, ds.PageIns)
				}
				g.Close()
			}
			if ran == 0 {
				t.Skip("no device on this host paged every block")
			}
		})
	}
}
