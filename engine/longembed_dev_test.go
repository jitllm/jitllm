package engine

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// longEmbedText is a text of at least n of m's tokens whose sentences do not
// repeat: a run cut short of its end reads a different text, which a
// repeated sentence would hide.
func longEmbedText(ids func(string) []int32, n int) string {
	nouns := []string{"river", "engine", "garden", "ledger", "harbor", "violin", "glacier", "lantern", "orchard", "compass",
		"meadow", "furnace", "archive", "canyon", "beacon", "quarry", "atlas", "thimble", "falcon", "vineyard"}
	verbs := []string{"carries", "measures", "hides", "repairs", "follows", "paints", "weighs", "remembers", "outlasts", "echoes"}
	var b strings.Builder
	for i := 0; len(ids(b.String())) < n; i++ {
		fmt.Fprintf(&b, "On day %d the %s %s the %s near the %s of %d. ", 3*i+7, nouns[i%len(nouns)], verbs[(i*7)%len(verbs)],
			nouns[(i*3+5)%len(nouns)], nouns[(i*11+2)%len(nouns)], i*i%97)
	}
	return b.String()
}

// TestLongEmbeddingsRunWhole embeds a text longer than a device's reserved
// width (nn.MaxDevicePrefillChunk rows) with every block on each card, against
// the CPU device's embedding of it: its rows attend each other both ways, so
// a card that ran it in pieces, each attending its own, parts from the CPU.
// The same text cut to the width must part from the whole, or the gate
// cannot see a cut.
func TestLongEmbeddingsRunWhole(t *testing.T) {
	const want = 700
	long := 0
	for _, g := range loadEmbedGoldens(t) {
		t.Run(g.Model, func(t *testing.T) {
			m, openOpts := openEmbedModel(t, g)
			eo := append(append([]Option(nil), openOpts...), WithKVF16(false))
			bound := embedDevCos
			if g.Model == "qwen3-embedding-0.6b" {
				bound = embedDevCosQuant
			}
			he, err := NewEmbedder(m, eo...)
			if err != nil {
				t.Fatal(err)
			}
			if mx := he.MaxTokens(); mx < want {
				he.Close()
				t.Logf("%s embeds at most %d tokens: no run past the reserved width", g.Model, mx)
				return
			}
			long++
			ids := m.EmbedIDs(longEmbedText(m.EmbedIDs, want))
			if len(ids) < want {
				t.Fatalf("the text is %d tokens, want %d", len(ids), want)
			}
			embed := func(e *Embedder, ids []int32) []float64 {
				v, err := e.Embed(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := make([]float64, len(v))
				for i, x := range v {
					out[i] = float64(x)
				}
				return out
			}
			host := embed(he, ids)
			cut := embed(he, ids[:512])
			he.Close()
			f32 := func(x []float64) []float32 {
				out := make([]float32, len(x))
				for i, v := range x {
					out[i] = float32(v)
				}
				return out
			}
			if c := cosine(f32(cut), host); !(c < bound) {
				t.Fatalf("the text cut to 512 tokens is %.9f from the whole: the gate cannot see a cut", c)
			}
			ran := 0
			for _, spec := range stepDevices() {
				gpux, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || gpux == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				ran++
				t.Run(spec, func(t *testing.T) {
					defer gpux.Close()
					e, err := NewEmbedder(m, eo...)
					if err != nil {
						t.Fatal(err)
					}
					defer e.Close()
					if err := e.SetDeviceLayers(gpux, -1); err != nil {
						t.Fatal(err)
					}
					if placed := e.DeviceBlocks(); placed != m.Cfg.NLayer {
						t.Fatalf("placed %d of %d blocks: %v %s", placed, m.Cfg.NLayer, e.DeviceDeclines(), gpux.Err())
					}
					s0 := gpux.Stats()
					got := embed(e, ids)
					s1 := gpux.Stats()
					c := cosine(f32(got), host)
					t.Logf("%d tokens on %s: cosine to the CPU %.9f (%d submissions, %d prompt splits)", len(ids), spec, c,
						s1.Submits-s0.Submits, s1.PromptSplits-s0.PromptSplits)
					if s1.Submits == s0.Submits {
						t.Fatal("the placed blocks did not run on the device")
					}
					if !(c >= bound) || math.IsNaN(c) {
						t.Errorf("%d tokens on %s: cosine to the CPU %.9f < %v", len(ids), spec, c, bound)
					}
				})
			}
			if ran == 0 {
				t.Skip("no device present")
			}
		})
	}
	if long == 0 {
		t.Fatalf("no embedding model takes %d tokens", want)
	}
}
