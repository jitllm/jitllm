package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// embedDevCos is how close a device's embedding must sit to the host's: the
// f32 reduction-order band of a unit vector, an order of magnitude inside
// the reference gate's 1e-4 (embedCosBound), so a device arm that passes
// here passes that gate wherever the host does.
const embedDevCos = 1 - 1e-5

// embedDevCosQuant is the bound for a decoder embedding model with quantized
// weights, whose device runs its matrices over int8 activations where the
// host's prefill does not: qwen3-embedding-0.6b (Q8_0) reads 0.99939-0.99942
// on a V100's CUDA and Vulkan at half and every block, where its F16 file
// reads 0.9999994 -- the band is the quantized activations, not the wiring.
const embedDevCosQuant = 1 - 1e-3

// TestEmbeddingsOnEveryDevice runs every embedding model -- the BERT family
// through its encoder segment, a decoder (qwen3-embedding, EmbeddingGemma)
// through its State, EmbeddingGemma's blocks planned non-causal -- with
// every encoder block on each device present, and with half of them, against
// the host and against the references TestEmbeddingsMatchReference holds the
// host to -- the encoder's segment through the one placement path
// (encdev.go). The violation is the output LayerNorm's weight taken out of
// what the device is handed: BERT's post-norm order is the device's own wiring
// (nn.LayerPlan.PostResidNorm), which a gate running the pre-norm block would
// not see.
func TestEmbeddingsOnEveryDevice(t *testing.T) {
	for _, g := range loadEmbedGoldens(t) {
		t.Run(g.Model, func(t *testing.T) {
			m := openEmbedModel(t, g)
			// A decoder embedding model (qwen3-embedding, EmbeddingGemma)
			// places its State's blocks; its history is f32 on the device.
			nb := m.encBlocks()
			bound := embedDevCos
			if g.Model == "qwen3-embedding-0.6b" {
				bound = embedDevCosQuant
			}
			if !m.IsEncoder() {
				nb = m.Cfg.NLayer
				m.SetKVF16(false)
			}
			embed := func(e *Embedder) [][]float32 {
				var out [][]float32
				for _, text := range g.Texts {
					v, err := e.Embed(m.EmbedIDs(text))
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), v...))
				}
				return out
			}
			he, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			host := embed(he)
			he.Close()
			ran := 0
			for _, spec := range stepDevices() {
				probe, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || probe == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				probe.Close()
				ran++
				t.Run(spec, func(t *testing.T) {
					arm := func(n int) ([][]float32, int) {
						gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
						if err != nil {
							t.Fatal(err)
						}
						defer gpu.Close()
						e, err := m.NewEmbedder()
						if err != nil {
							t.Fatal(err)
						}
						defer e.Close()
						if err := e.SetDeviceLayers(gpu, n); err != nil {
							t.Fatal(err)
						}
						placed := e.DeviceBlocks()
						if want := n; n < 0 && placed != nb || n >= 0 && placed != want {
							t.Fatalf("placed %d blocks, offered %d of %d: %v %s", placed, n, nb, e.DeviceDeclines(), gpu.Err())
						}
						// The device ran them: one submission a run of placed
						// blocks a text, counted by the device itself.
						s0 := gpu.Stats().Submits
						out := embed(e)
						if s := gpu.Stats().Submits - s0; s < len(g.Texts) {
							t.Fatalf("%d submission(s) for %d texts: the placed blocks did not run on the device", s, len(g.Texts))
						}
						return out, placed
					}
					closest := func(got [][]float32) float64 {
						w := 1.0
						for i := range host {
							h := make([]float64, len(host[i]))
							for j, x := range host[i] {
								h[j] = float64(x)
							}
							w = math.Min(w, cosine(got[i], h))
						}
						return w
					}
					for _, n := range []int{nb / 2, -1} {
						got, placed := arm(n)
						c := closest(got)
						t.Logf("%d of %d blocks on %s: worst cosine to the host %.9f", placed, nb, spec, c)
						if !(c >= bound) {
							t.Errorf("%d blocks on %s: cosine to the host %.9f < %v", placed, spec, c, bound)
						}
						if placed == nb {
							for i := range got {
								if i < len(g.Llamacpp) {
									if lc := cosine(got[i], g.Llamacpp[i]); !(lc >= embedCosBound) && g.LGGUF == "" {
										t.Errorf("text %d on %s: cosine to llama.cpp %.7f < %v", i, spec, lc, embedCosBound)
									}
								}
							}
						}
					}
					// violation runs the every-block arm with cut applied and
					// fails unless it moves the answer off the host's.
					violation := func(name string, cut func() func()) {
						undo := cut()
						bad, _ := arm(-1)
						undo()
						c := closest(bad)
						t.Logf("%s on %s: worst cosine to the host %.6f", name, spec, c)
						if c >= bound {
							t.Errorf("%s reads cosine %.9f on %s: the gate does not see it", name, c, spec)
						}
					}
					if m.IsEncoder() {
						// The output norm's weight gone from what the device is
						// handed (encWeights reads it at the offer).
						violation("the output LayerNorm without its weight", func() func() {
							saved := make([][]float32, len(m.enc.blocks))
							for i := range m.enc.blocks {
								b := &m.enc.blocks[i]
								saved[i] = b.ln2W
								b.ln2W = onesLike(b.ln2W)
							}
							return func() {
								for i := range m.enc.blocks {
									m.enc.blocks[i].ln2W = saved[i]
								}
							}
						})
						return
					}
					// A decoder's: the attention norm's weight gone from what
					// the device is handed at the offer.
					violation("the attention norm without its weight", func() func() {
						saved := make([][]float32, len(m.layers))
						for i := range m.layers {
							saved[i] = m.layers[i].attnNorm
							m.layers[i].attnNorm = onesLike(saved[i])
						}
						return func() {
							for i := range m.layers {
								m.layers[i].attnNorm = saved[i]
							}
						}
					})
					if !m.Cfg.NonCausal {
						return
					}
					// A bidirectional decoder's mask is load-bearing on the
					// device: run causal there, and with the sliding layers'
					// windows as wide as the sequence (the long text is past
					// the window).
					violation("a causal mask on the device", func() func() {
						m.Cfg.NonCausal = false
						return func() { m.Cfg.NonCausal = true }
					})
					violation("no symmetric window on the sliding layers", func() func() {
						was := m.Cfg.SWAWindow
						m.Cfg.SWAWindow = 1 << 20
						return func() { m.Cfg.SWAWindow = was }
					})
				})
			}
			if ran == 0 {
				noDevice(t, "device", nil)
			}
		})
	}
}
