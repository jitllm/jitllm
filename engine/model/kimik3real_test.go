package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// k3RealGolden is scripts/k3gold.py's real-<name> golden: Moonshot's remote
// code, as the checkpoint ships it, run in float32 on the checkpoint's own
// weights. Each sequence is a prompt and the reference's greedy continuation,
// and each row is one position's logits teacher-forced over that sequence:
// the vocabulary's first ids (Head) and the reference's highest (Top, TopV).
type k3RealGolden struct {
	Vocab int `json:"vocab"`
	Seqs  []struct {
		Text   string  `json:"text"`
		Prompt []int32 `json:"prompt"`
		Cont   []int32 `json:"cont"`
		Reply  string  `json:"reply"`
		Rows   []struct {
			Argmax int32     `json:"argmax"`
			Max    float64   `json:"max"`
			Head   []float64 `json:"head"`
			Top    []int32   `json:"top"`
			TopV   []float64 `json:"topv"`
		} `json:"rows"`
	} `json:"seqs"`
}

// k3RealModels are the trained Kimi-K3 checkpoints held to the reference, each
// relative to JITLLM_MODELS: a GGUF (llama.cpp's converter's from the
// checkpoint the golden names, or a published one), or the checkpoint's own
// safetensors directory converted by this tree. A missing file is skipped by
// name.
//
// The bound is the worst position's NMSE over the reference's top logits and
// the vocabulary's head, and flips the confident argmax disagreements allowed
// (a reference margin over tieMargin). An exact arm reads the weights its
// golden ran: the F32 conversion against the float32 reference, the F16 one
// against the reference on its weights rounded to float16 (K3_ROUND) -- a
// rounding that alone moves the reference's own logits by 5.0e-3 on this
// sharply trained model -- and the Kazakh checkpoint, float16 already. An
// exact arm runs the violations too. Q8_0 and MXFP4 are quantizations, held to
// the float32 reference within their band.
var k3RealModels = []struct {
	golden, file string
	nmse         float64
	flips        int
	exact        bool
}{
	// llama.cpp's convert_hf_to_gguf.py --outtype f32 on
	// inference-optimization/Kimi-K3-0.40B, the reference's weights exactly.
	{"Kimi-K3-0.40B", "kimik3/Kimi-K3-0.40B-F32.gguf", 1e-8, 0, true},
	// The published F16 conversion of it (murillo2000/Kimi-K3-0.40B-GGUF).
	{"Kimi-K3-0.40B-float16", "kimik3/Kimi-K3-0.40B-F16.gguf", 1e-8, 0, true},
	// The published quantizations: mradermacher's Q8_0, and murillo2000's
	// MXFP4 of inference-optimization/Kimi-K3-0.40B-MXFP4 (the routed experts
	// MXFP4, the rest F16).
	{"Kimi-K3-0.40B", "kimik3/Kimi-K3-0.40B.Q8_0.gguf", 2e-2, 0, false},
	// The checkpoint itself, converted from safetensors: the same container
	// as the F32 GGUF's (convert.TestK3SafetensorsMatchesItsGGUF).
	{"Kimi-K3-0.40B", "kimik3/Kimi-K3-0.40B-hf", 1e-8, 0, true},
	// inference-optimization/Kimi-K3-0.40B-MXFP4 converted from safetensors,
	// its compressed-tensors MXFP4 experts (routed and shared) moved into the
	// container byte for byte, against the reference on those same weights
	// (compressed-tensors' own decompression). Its only distance is the
	// engine's MXFP4 kernels, which round the activation they multiply:
	// 3.2e-5 on the host, CUDA and Vulkan, every greedy token the
	// reference's. The same container held to the unquantized reference
	// (real-Kimi-K3-0.40B) reads 4.3e-2, the experts' quantization, over
	// 400 times this bound.
	{"Kimi-K3-0.40B-MXFP4", "kimik3/Kimi-K3-0.40B-MXFP4-hf", 1e-4, 0, false},
	// A mixed quantization: attn_q at Q4_K beside attn_k and attn_v at Q8_0,
	// which convert.k3JoinRows re-stores as one Q8_0 projection. Its one flip
	// is "The capital of France" -> "ith" where the reference says " my", a
	// 0.955 margin, and llama.cpp reading the same file answers "ith" too.
	{"Kimi-K3-0.40B", "kimik3/Kimi-K3-0.40B.Q4_K_M.gguf", 2e-1, 1, false},
	{"Kimi-K3-0.40B", "kimik3/Kimi-K3-0.40B-MXFP4.gguf", 1e-1, 1, false},
	// convert_hf_to_gguf.py --outtype f16 on
	// Eraly-ml/Kimi-K3-0.40B-Kazakh-CPT-59M, whose weights are float16.
	{"Kimi-K3-0.40B-Kazakh-CPT-59M", "kimik3/Kimi-K3-0.40B-Kazakh-CPT-59M-F16.gguf", 1e-8, 0, true},
}

// k3RealViolations are the fixture's violations, less two, and the one only a
// checkpoint with no decay bound can show: K3's bound applied where the file
// states none (the fixture's -5). The unbounded decay is this checkpoint's own
// graph, and the MLA latent norms' epsilon is inert on trained weights: at
// rms_norm_eps (1e-5 against the class's 1e-6) the 0.40B moves 2.1e-7 and
// its Kazakh continuation 1.7e-12, so synth-kimik3 stays that choice's gate.
var k3RealViolations = func() []c6Violation {
	out := []c6Violation{{"a decay bound the checkpoint does not state (-5)", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.KDALowerBound = -5 })
	}}}
	for _, v := range k3Violations {
		switch v.name {
		case "the KDA decay unbounded (Kimi-Linear's softplus)",
			"the MLA latent norms at rms_norm_eps (llama.cpp's)":
		default:
			out = append(out, v)
		}
	}
	return out
}()

// TestKimiK3RealMatchesReference holds trained Kimi-K3 checkpoints to their
// own remote code (scripts/k3gold.py real-<name>): every position of three
// prompts and the reference's greedy continuations, teacher-forced through
// decode from the first token and through a prefill of the prompt, on the host
// and with every block and the head on each device. Kimi-K3-0.40B is a test
// model trained on one copypasta, so its first two prompts continue that text
// with wide margins; its weights are trained rather than drawn, the
// distribution synth-kimik3 cannot stand for. Its KDA blocks carry no
// gate_lower_bound, so the decay is Kimi-Linear's softplus under K3's
// full-rank gate, and it checkpoints every four of eight blocks.
func TestKimiK3RealMatchesReference(t *testing.T) {
	ran := 0
	for _, c := range k3RealModels {
		t.Run(filepath.Base(c.file), func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "kimik3", "real-"+c.golden+".json"))
			if err != nil {
				t.Fatal(err)
			}
			g := &k3RealGolden{}
			if err := json.Unmarshal(raw, g); err != nil {
				t.Fatal(err)
			}
			path, ok := existingModel(testmodels.Path(c.file))
			if !ok {
				t.Skipf("MODEL MISSING: %s (internal/testmodels/fetch.sh kimi, or the conversion its "+
					"k3RealModels line names) -- this subtest proved nothing", testmodels.Path(c.file))
			}
			ran++
			m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			k3RealSelected(t, m)
			for _, s := range g.Seqs {
				if got := m.Vocab.Encode(s.Text, false); !slices.Equal(got, s.Prompt) {
					t.Errorf("Encode(%q) = %v, the reference's tokenizer %v", s.Text, got, s.Prompt)
				}
			}
			arm := func(t *testing.T, gpu *tier.GPU) {
				worst, flips, ties := 0.0, 0, 0
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
						if am := Greedy(lg); am != r.Argmax {
							if r.TopV[0]-r.TopV[1] > tieMargin {
								flips++
								t.Logf("%q %s pos %d: argmax %q, the reference %q (margin %.3f)", s.Text, what, p,
									m.Vocab.Text(am), m.Vocab.Text(r.Argmax), r.TopV[0]-r.TopV[1])
							} else {
								ties++
							}
						}
					}
					state := func() *State {
						st := m.NewState(len(ids) + 1)
						if gpu != nil {
							st.SetDeviceLayers(gpu, -1)
							if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
								t.Fatalf("placed %d of %d blocks, head %v: %v", st.GPULayers(), m.Cfg.NLayer,
									st.HeadOnDevice(), gpu.Err())
							}
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
				t.Logf("%d sequences teacher-forced, decode and prefill: worst NMSE %.3e (bound %.0e), "+
					"%d confident argmax flips (allowed %d), %d ties", len(g.Seqs), worst, c.nmse, flips, c.flips, ties)
				if worst > c.nmse || flips > c.flips {
					t.Errorf("worst NMSE %.3e, %d flips", worst, flips)
				}
			}
			t.Run("host", func(t *testing.T) {
				arm(t, nil)
				if c.exact {
					k3RealViolate(t, m, g, c.nmse)
				}
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
		t.Skip("no trained Kimi-K3 checkpoint here (internal/testmodels/fetch.sh kimi)")
	}
}

// k3RealSelected fails a container that does not carry Kimi-K3-0.40B's
// graph: eight blocks, MLA at 4 and 8 (one-indexed) and KDA elsewhere, a
// checkpoint every four blocks, a dense lead then latent mixtures (1024 ->
// 512) of 8 experts top-2 beside one shared expert, situ, the MLA output gate
// and compressed query, and KDA's full-rank gate with no decay bound.
func k3RealSelected(t *testing.T, m *Model) {
	t.Helper()
	c := m.Cfg
	lin, full := jlm.LayerLinearAttn, jlm.LayerFullAttn
	ok := c.Arch == "kimi-k3" && c.ResAttn() && c.AttnResBlock == 4 && c.ExpertLatent == 512 &&
		c.Act == nn.ActSitu && c.KDALowerBound == 0 && c.ChanDecay() && c.MLA() && c.NoPosEnc &&
		c.QLoraRank == 256 && c.KVLoraRank == 128 && c.NDenseLead == 1 && c.NExpert == 8 &&
		c.NExpertUsed == 2 && c.ExpertSigmoid && c.NEmbd == 1024 &&
		slices.Equal(c.LayerKinds, []jlm.LayerKind{lin, lin, lin, full, lin, lin, lin, full})
	for i := range m.layers {
		l := &m.layers[i]
		rec := c.LayerKind(i).Recurrent()
		ok = ok && (l.mlaGate.e != nil) == !rec && (l.ssmGate.e != nil) == rec && l.ssmGA.e == nil &&
			(l.routedDown.e != nil) == (i >= 1)
	}
	if !ok {
		t.Fatalf("the container does not carry Kimi-K3-0.40B's graph: %+v", c)
	}
}

// k3RealViolate runs each violation through decode over every sequence and
// demands it land far past the arm's bound: a feature the trained weights make
// inert would otherwise pass for one the engine runs.
func k3RealViolate(t *testing.T, m *Model, g *k3RealGolden, bound float64) {
	for _, v := range k3RealViolations {
		t.Run(v.name, func(t *testing.T) {
			undo := v.mut(m)
			defer undo()
			worst := 0.0
			for _, s := range g.Seqs {
				ids := append(slices.Clone(s.Prompt), s.Cont[:len(s.Cont)-1]...)
				st := m.NewState(len(ids) + 1)
				for p, id := range ids {
					lg, err := st.Forward(id)
					if err != nil {
						t.Fatal(err)
					}
					worst = max(worst, llama4Cmp(s.Rows[p].Head, lg))
				}
				st.Close()
			}
			t.Logf("worst NMSE %.3e", worst)
			if !(worst > 1e3*bound) {
				t.Errorf("the gate cannot see it: worst NMSE %.3e against a bound of %.0e", worst, bound)
			}
		})
	}
}
