//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestStreamResidualsAreCutByStream forces the device to run a prompt chunk
// without its batched scratch (the room fault noscratch)
// for the two models whose residual is several streams, stream-major: Kimi-K3
// and DeepSeek V4. The fallback cuts the chunk a row at a time, and a row of
// such a chunk is that row of every stream, not a slice of x. The prompt must
// match the host's, with the fallback counted (PromptSplits, RULE 10: the
// path ran), and with the cut armed to take every stream from the first
// stream's rows (streamcutrows) it must not.
func TestStreamResidualsAreCutByStream(t *testing.T) {
	const n = 96
	for _, md := range []struct {
		name string
		open func(t *testing.T) *Model
	}{
		{"deepseek4", func(t *testing.T) *Model { return openSynth(t, "synth-deepseek4") }},
		{"kimik3", func(t *testing.T) *Model { return openSynth(t, "synth-kimik3") }},
	} {
		t.Run(md.name, func(t *testing.T) {
			m := md.open(t)
			defer m.Close()
			ids := make([]int32, n)
			for i := range ids {
				ids[i] = int32(3 + (i*11)%61)
			}
			nl := m.Cfg.NLayer
			ran := 0
			for _, spec := range stepDevices() {
				gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff),
					tier.WithConfig(func(c *tier.Config) { c.NoExactPrompt = true })}, testTierOpts(t)...)...)
				if err != nil || gpu == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				ran++
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					run := func(dev bool) ([]float32, int) {
						st := m.NewState(n + 1)
						defer st.Close()
						if dev {
							st.SetDeviceLayers(gpu, -1)
							if st.GPULayers() != nl || !st.HeadOnDevice() {
								t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), nl, st.HeadOnDevice(), gpu.Err())
							}
						}
						s0 := gpu.Stats()
						lg, err := st.Prefill(ids)
						if err != nil {
							t.Fatal(err)
						}
						return slices.Clone(lg), gpu.Stats().PromptSplits - s0.PromptSplits
					}
					want, _ := run(false)
					for _, room := range []string{"noscratch", "noscratch-split"} {
						t.Run(room, func(t *testing.T) {
							gpu.SetFault("room", room)
							defer gpu.SetFault("room", "")
							got, splits := run(true)
							nm, _ := nmse32(want, got)
							t.Logf("%d rows, %s: %d split chunk(s), NMSE %.3e against the host, argmax %d / %d",
								n, room, splits, nm, Greedy(got), Greedy(want))
							if splits == 0 {
								t.Fatalf("the chunk was not split: the fallback never ran (%q)", gpu.Err())
							}
							if math.IsNaN(nm) || nm > 1e-5 || Greedy(got) != Greedy(want) {
								t.Errorf("NMSE %.3e against the host, argmax %d, want %d", nm, Greedy(got), Greedy(want))
							}
							gpu.SetFault("rows", "streamcutrows")
							bad, _ := run(true)
							gpu.SetFault("rows", "")
							nmBad, _ := nmse32(want, bad)
							t.Logf("with the later streams cut one row on: NMSE %.3e", nmBad)
							if nmBad <= 1e-5 && Greedy(bad) == Greedy(want) {
								t.Errorf("the violation (the later streams cut one row on) reads NMSE %.3e: the gate cannot see it", nmBad)
							}
						})
					}
				})
			}
			if ran == 0 {
				t.Skip("no device on this host")
			}
		})
	}
}

// openSynth opens a synthetic fixture without its golden: this gate compares
// the device with the host, not with transformers.
func openSynth(t *testing.T, name string) *Model {
	t.Helper()
	path, ok := existingModel(testmodels.Path(name + ".gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name+".gguf"))
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	return m
}
