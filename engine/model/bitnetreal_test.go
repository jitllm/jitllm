package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// bitnetRealModels are the trained BitNet b1.58 checkpoints, each relative to
// JITLLM_MODELS, held to scripts/bitnetreal.py's golden of transformers'
// BitNetForCausalLM on that checkpoint's own weights.
//
// Teacher-forced through every position, a trained BitNet parts from its
// reference the way the fixtures do, only more often: an activation within a
// few ulps of an int8 rounding boundary lands on the other side under another
// summation order, and the logits downstream of that one int8 step move by up
// to 4e-2 in NMSE without moving the token. So the gate is the token: argmax
// agreement over every position (agree, a fraction), no confident flip (a
// reference margin over tieMargin), and the NMSE bound only as a backstop at
// several times what a flip reads.
var bitnetRealModels = []struct {
	file, golden string
	typ          quant.Type
	nmse, agree  float64
	flips        int
}{
	// The release itself (microsoft/bitnet-b1.58-2B-4T), from safetensors:
	// uint8 codes and bf16 weight_scales, TQ2_0, its scales exact.
	{"bitnet-b1.58-2B-4T", "bitnet-b1.58-2B-4T", quant.TQ2_0, 2e-1, 0.98, 0},
	// The bf16 master weights (microsoft/bitnet-b1.58-2B-4T-bf16), ternarized
	// as BitNetForCausalLM does under its torch_dtype, in bf16: other codes
	// than the release's (98-99% of each projection agrees), so its own
	// golden. Its distributions are flatter at more positions than the
	// release's (six to nine reference ties within tieMargin, against two),
	// and every position that parts is one of them, so it agrees at 95%.
	{"bitnet-b1.58-2B-4T-bf16", "bitnet-b1.58-2B-4T-bf16", quant.TQ2_0, 2e-1, 0.95, 0},
	// Microsoft's I2_S GGUF (microsoft/bitnet-b1.58-2B-4T-gguf, bitnet.cpp's
	// converter), its per-tensor f32 scales carried as TQ2_F32 and its
	// embedding F16. Its weights are neither golden's -- its scales are f32
	// means, the release's bf16 -- so it is held to the release's within the
	// two models' own distance.
	{"bitnet-b1.58-2B-4T-i2_s.gguf", "bitnet-b1.58-2B-4T", quant.TQ2_F32, 2e-1, 0.98, 2},
}

// TestBitNetRealMatchesReference holds BitNet b1.58 2B-4T to transformers'
// BitNetForCausalLM on the release's own weights: every position of three
// prompts and the reference's greedy continuations, teacher-forced through
// decode from the first token and through a prefill of the prompt, on the
// host and with every block and the head on each device.
func TestBitNetRealMatchesReference(t *testing.T) {
	ran := 0
	for _, c := range bitnetRealModels {
		t.Run(c.file, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "bitnet", "real-"+c.golden+".json"))
			if err != nil {
				t.Fatal(err)
			}
			g := &k3RealGolden{}
			if err := json.Unmarshal(raw, g); err != nil {
				t.Fatal(err)
			}
			path := testmodels.Path(c.file)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("MODEL MISSING: %s (download it to JITLLM_MODELS) -- this subtest proved nothing", path)
			}
			ran++
			var container string
			if strings.HasSuffix(c.file, ".gguf") {
				container = jlmOf(t, path)
			} else {
				container = jlmOfSafetensors(t, path)
			}
			m, err := Open(container, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !bitnetSelected(m, c.typ) {
				t.Fatalf("%s does not carry BitNet's graph with %s projections: %+v", c.file, c.typ, m.Cfg)
			}
			// The release's tokenizer.json is the authority. Microsoft's GGUF
			// states no tokenizer.ggml.pre, so it splits as llama.cpp's
			// "default" does ("(n):\n" into four pieces, not tokenizer.json's
			// two); its arm reads the golden's ids either way.
			gguf := strings.HasSuffix(c.file, ".gguf")
			for _, s := range g.Seqs {
				if got := m.Vocab.Encode(s.Text, true); !slices.Equal(got, s.Prompt) {
					if gguf {
						t.Logf("Encode(%q) = %v, tokenizer.json %v (the GGUF names no pre-tokenizer)",
							s.Text, got, s.Prompt)
						continue
					}
					t.Errorf("Encode(%q) = %v, the reference's tokenizer %v", s.Text, got, s.Prompt)
				}
			}
			arm := func(t *testing.T, gpu *tier.GPU) {
				worst, flips, ties, agree, total, placed := 0.0, 0, 0, 0, 0, 0
				for _, s := range g.Seqs {
					ids := append(slices.Clone(s.Prompt), s.Cont[:len(s.Cont)-1]...)
					check := func(what string, p int, lg []float32) {
						r := &s.Rows[p]
						var num, den float64
						for i, id := range r.Top {
							d := float64(lg[id]) - r.TopV[i]
							num, den = num+d*d, den+r.TopV[i]*r.TopV[i]
						}
						nmse := max(num/den, llama4Cmp(r.Head, lg))
						if math.IsNaN(nmse) || math.IsInf(nmse, 0) {
							t.Fatalf("%q %s pos %d: logits not finite", s.Text, what, p)
						}
						worst = max(worst, nmse)
						total++
						if am := Greedy(lg); am != r.Argmax {
							if r.TopV[0]-r.TopV[1] > tieMargin {
								flips++
								t.Logf("%q %s pos %d: argmax %q, the reference %q (margin %.3f)", s.Text, what, p,
									m.Vocab.Text(am), m.Vocab.Text(r.Argmax), r.TopV[0]-r.TopV[1])
							} else {
								ties++
							}
						} else {
							agree++
						}
					}
					state := func() *State {
						st := m.NewState(len(ids) + 1)
						if gpu != nil {
							// As many blocks as the card holds: every one on a card with
							// room (a V100), a partial seam on a small one, which runs
							// both tiers in one token. None placed proves nothing.
							st.SetDeviceLayers(gpu, -1)
							if st.GPULayers() == 0 {
								t.Fatalf("placed none of %d blocks: %v", m.Cfg.NLayer, gpu.Err())
							}
							placed = st.GPULayers()
						}
						return st
					}
					st := state()
					for p, id := range ids {
						lg, err := st.Forward(id)
						if err != nil {
							t.Fatal(err)
						}
						check("decode", p, lg)
					}
					st.Close()
					st = state()
					lg, err := st.Prefill(s.Prompt)
					if err != nil {
						t.Fatal(err)
					}
					check("prefill", len(s.Prompt)-1, lg)
					for p := len(s.Prompt); p < len(ids); p++ {
						if lg, err = st.Forward(ids[p]); err != nil {
							t.Fatal(err)
						}
						check("after prefill", p, lg)
					}
					st.Close()
				}
				t.Logf("%d sequences teacher-forced, decode and prefill, %d of %d blocks on the device: worst NMSE "+
					"%.3e (bound %.0e), %d of %d argmax the reference's, %d confident flips (allowed %d), %d ties",
					len(g.Seqs), placed, m.Cfg.NLayer, worst, c.nmse, agree, total, flips, c.flips, ties)
				if worst > c.nmse || flips > c.flips || float64(agree) < c.agree*float64(total) {
					t.Errorf("worst NMSE %.3e, %d flips, %d of %d agree (want %.0f%%)", worst, flips, agree, total,
						100*c.agree)
				}
			}
			t.Run("host", func(t *testing.T) {
				arm(t, nil)
				// The free-running read: the host's greedy text from each prompt.
				for _, s := range g.Seqs {
					st := m.NewState(len(s.Prompt) + len(s.Cont) + 1)
					lg, err := st.Prefill(s.Prompt)
					if err != nil {
						t.Fatal(err)
					}
					var out []int32
					for range s.Cont {
						id := Greedy(lg)
						out = append(out, id)
						if lg, err = st.Forward(id); err != nil {
							t.Fatal(err)
						}
					}
					st.Close()
					same := 0
					for same < len(out) && out[same] == s.Cont[same] {
						same++
					}
					t.Logf("%q -> %q (%d of %d tokens the reference's)", s.Text, m.Vocab.Decode(out), same, len(out))
				}
			})
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
						tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
					if err != nil || gpu == nil {
						t.Skipf("%s: not present (%v)", spec, err)
					}
					defer gpu.Close()
					arm(t, gpu)
				})
			}
		})
	}
	if ran == 0 {
		t.Skip("no BitNet checkpoint in JITLLM_MODELS -- this gate proved nothing")
	}
}
