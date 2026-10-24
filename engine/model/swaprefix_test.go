package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// prefixWindowFixtures are the models the prefix-cache window gate runs:
// exaone4's sliding window of 4 beside full layers, and DeepSeek V4's every
// block sliding, whose entries decide the restore (ds4kv.go).
var prefixWindowFixtures = []string{"synth-exaone4.gguf"}

// TestPrefixCacheRestoresAWindowedLayer runs the prefix cache on a model with
// windowed layers, on the host and with every block on each device: a prompt
// long enough that the device releases its windowed pages during the prefill
// fills the store, and a second prompt sharing it must restore nearly all of
// the shared part and continue with the tokens of an uncached run.
//
// A windowed layer stores only what its window reached, so its pages are not
// walked from zero: the coverage is the full layers', and restoreWin faults the
// window of the restored length. The violation walks a windowed layer from zero
// as a full layer is walked (kvCache.walkWindow); on a device, where the pages
// behind the window were never sent home, that restores nothing.
func TestPrefixCacheRestoresAWindowedLayer(t *testing.T) {
	specs := append([]string{""}, testdevices.Specs("cuda", "vulkan", "metal")...)
	for _, fx := range prefixWindowFixtures {
		t.Run(fx, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(fx)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			// f32: an f16 host cache cannot come home from a device (MigrateKV takes
			// []float32), and then nothing is stored at all.
			m.SetKVF16(false)
			defer m.ClearKVF16()
			const maxSeq, gen = 1024, 12
			shared := make([]int32, 700)
			for i := range shared {
				shared[i] = int32(3 + (i*11)%61)
			}
			first := append(append([]int32{}, shared...), 41, 42, 43)
			second := append(append([]int32{}, shared...), 51, 52, 53, 54, 55)
			for _, spec := range specs {
				name := spec
				if name == "" {
					name = "host"
				}
				t.Run(name, func(t *testing.T) {
					var g *tier.GPU
					if spec != "" {
						var err error
						g, err = tier.OpenWith(append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
						if err != nil || g == nil {
							noDevice(t, spec, err)
						}
						defer g.Close()
					}
					run := func(st KVStore, p []int32, walk bool) ([]int32, int, uint64) {
						s := m.NewState(maxSeq)
						defer s.Close()
						if g != nil {
							if err := s.SetDevice(g); err != nil {
								t.Fatal(err)
							}
							if s.GPULayers() != m.Cfg.NLayer {
								t.Skipf("CARD TOO SMALL: %d of %d blocks -- this arm proved nothing",
									s.GPULayers(), m.Cfg.NLayer)
							}
						}
						s.kv.walkWindow = walk
						if st != nil {
							s.SetKVStore(st)
							mustKey(t, s, "window/prefix")
						}
						before := 0
						if g != nil {
							before = composedBlockSteps(g)
						}
						lg, err := s.PrefillCached(p)
						if err != nil {
							t.Fatal(err)
						}
						out := make([]int32, 0, gen)
						for i := 0; i < gen; i++ {
							id := int32(argmax(lg))
							out = append(out, id)
							if lg, err = s.Forward(id); err != nil {
								t.Fatal(err)
							}
						}
						if g != nil {
							// The decode's steps, and the prompt's last row
							// where it is the one the store left to prefill.
							steps := gen
							if len(p)-s.KVRestored() == 1 {
								steps++
							}
							composedBlocksRan(t, composedOf(m, g), before, steps*s.GPULayers(), g)
						}
						stored, fetched, _ := s.KVPageStats()
						t.Logf("restored %d: %d page(s) stored, %d fetched, %d sync failure(s), %d unmigratable, %d mismatched",
							s.KVRestored(), stored, fetched, s.KVDeviceSyncFailures(), s.KVPrefixUnmigratable(), s.KVStoreMismatches())
						wb := uint64(0)
						for li := range s.kv.layers {
							if s.kv.layers[li].win > 0 {
								wb = max(wb, s.kv.layers[li].bytes())
							}
						}
						return out, s.KVRestored(), wb
					}
					want, _, _ := run(nil, second, false)
					st := NewMemStore()
					if _, n, _ := run(st, first, false); n != 0 {
						t.Fatalf("the first pass restored %d positions from an empty store", n)
					}
					got, reused, wb := run(st, second, false)
					floor := len(shared) / 32 * 32
					t.Logf("%d of %d shared positions restored; a windowed layer holds %d host bytes after",
						reused, len(shared), wb)
					if reused < floor || reused > len(shared) {
						t.Fatalf("restored %d positions of a %d-token shared prefix, want at least %d",
							reused, len(shared), floor)
					}
					sameTokens(t, "windowed prefix", want, got)
					// The violation, on a fresh store so nothing of the shipping arm's
					// pages is reused.
					sv := NewMemStore()
					run(sv, first, true)
					bad, badN, _ := run(sv, second, true)
					t.Logf("violation (windowed layers walked from zero): %d restored", badN)
					if spec != "" && badN >= floor && equalTokens(bad, want) {
						t.Fatal("walking a windowed layer from zero restored the prefix on a device that released " +
							"its pages: the gate cannot tell the window restore from the full walk")
					}
				})
			}
		})
	}
}
