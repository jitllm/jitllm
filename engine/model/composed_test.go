package model

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// composedDecode decodes toks teacher-forced after ids on a fresh tier of
// dev serving the parts (Config.Composed), every block placed, and returns
// every step's logits, the tier's stats and its census of the launches of
// its own kernels on the steps the composer was asked for.
func composedDecode(t *testing.T, m *Model, dev string, ids, toks []int32, parts tier.Parts, noGraph, float bool, extra ...tier.Option) ([][]float32, tier.Stats, bool, map[string]int) {
	t.Helper()
	return composedDecodeWith(t, m, dev, ids, toks, parts, noGraph, float, nil, extra...)
}

// axesShift states each row's rotary positions as k times the State's and d
// past that: the violation of a served table that turns a row by the
// positions it is told.
type axesShift struct {
	nn.RopeAxesDevice
	k, d int32
	buf  [4]int32
}

func (a *axesShift) SetRopeAxes(pos int, at []int32) {
	for i, v := range at[:min(len(at), 4)] {
		a.buf[i] = a.k*v + a.d
	}
	a.RopeAxesDevice.SetRopeAxes(pos, a.buf[:min(len(at), 4)])
}

// composedTables is how many rotary tables the front end states for m: the
// tables a served step writes.
func composedTables(m *Model) int {
	n := 0
	for _, g := range m.frontGraphs().tables {
		if g != nil {
			n++
		}
	}
	return n
}

// tokenShift promises each row's gather, a decode step's and a prompt's, at
// the token d past its own and the position p past its own: the violation
// of a served gather that reads the rows it is promised.
type tokenShift struct {
	nn.EmbedDevice
	d, vocab int32
	p        int
	ids      []int32
}

func (e *tokenShift) EmbedRows(ids []int32, pos int, dst []float32) bool {
	e.ids = e.ids[:0]
	for _, id := range ids {
		e.ids = append(e.ids, (id+e.d)%e.vocab)
	}
	return e.EmbedDevice.EmbedRows(e.ids, pos+e.p, dst)
}

// shiftTokens has the State promise its device every row's gather d tokens
// past its own.
func shiftTokens(d, vocab int) func(*State) { return shiftEmbed(d, 0, vocab) }

// shiftEmbed has the State promise its device every row's gather d tokens
// and p positions past its own.
func shiftEmbed(d, p, vocab int) func(*State) {
	return func(s *State) {
		if s.ldEmbed != nil {
			s.ldEmbed = &tokenShift{EmbedDevice: s.ldEmbed, d: int32(d), vocab: int32(vocab), p: p}
		}
	}
}

// shiftAxes has the State tell its device every row's rotary positions d
// past their own.
func shiftAxes(d int32) func(*State) { return stretchAxes(1, d) }

// stretchAxes has the State tell its device every row's rotary positions as
// k times their own and d past that.
func stretchAxes(k, d int32) func(*State) {
	return func(s *State) {
		if s.ldRope != nil {
			s.ldRope = &axesShift{RopeAxesDevice: s.ldRope, k: k, d: d}
		}
	}
}

// composedDecodeWith is composedDecode with after applied to the State once
// its device is attached.
func composedDecodeWith(t *testing.T, m *Model, dev string, ids, toks []int32, parts tier.Parts, noGraph, float bool, after func(*State), extra ...tier.Option) ([][]float32, tier.Stats, bool, map[string]int) {
	t.Helper()
	// The prompt is one chunk of a few rows: every arm reserves and warms the
	// widths up to 64, each a form of the composer's where it serves them.
	opts := append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) {
			c.Composed, c.NoGraph, c.ComposedFloat, c.Census, c.PromptRows = parts, noGraph, float, true, 64
		})},
		testTierOpts(t)...)
	opts = append(opts, extra...)
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + len(toks) + 4)
	defer s.Close()
	// A device that fails is an error here, never a sequence finished at home.
	s.SetDeviceFallback(false)
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%s took %d of %d blocks: %s", dev, s.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	if after != nil {
		after(s)
	}
	l, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]float32
	for _, tok := range toks {
		out = append(out, append([]float32(nil), l...))
		if l, err = s.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
	out = append(out, append([]float32(nil), l...))
	if s.DeviceDemotions() != 0 {
		t.Fatalf("the device failed %d time(s) and its blocks moved home", s.DeviceDemotions())
	}
	return out, g.Stats(), s.HeadOnDevice(), g.Census()
}

// servedModels is every checkpoint whose blocks and head the front end states
// as ops and that one card of 16 GiB holds: dense models of the llama, qwen2,
// qwen3, gemma, gemma2, gemma3, olmo2, exaone4, arcee, granite, hunyuan,
// phi3, falcon and apertus classes, the dense latent attention fixtures
// (DeepSeek-V3's block at f32, and under YaRN), and the text models of qwen2vl
// and qwen3vl. Those state their rotaries over several position axes, which
// the served table turns each row by as the State states them; on a text row every axis's
// position is the same, so serving them here says the blocks and the table
// are served at text positions. TestMRopeMatchesTransformers decodes after a
// picture, where the rotary positions leave the cache's. And the mixtures:
// routers softmax, sigmoid, biased, grouped, renormalised or not and scaled,
// gated and ungated shared experts, dense lead blocks, biased banks, and
// latent attention beside them. And the attention features read at a row's
// position or stated per element: Ministral-3-3B's and Pixtral's attention
// temperature on every layer, Llama 4's weightless q/k norm, temperature on
// its unrotated layers and input-weighted top-1 experts, and HunyuanVL's
// XD-RoPE, whose pair halves turn by two coordinates (on a text row, one
// position: TestHunyuanVLTextMatchesTransformers decodes after a picture).
var servedModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf",
	"gemma-2b.gguf", "Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
	"SmolLM2-360M-Instruct-Q8_0.gguf", "stories15M-q8_0.gguf", "SmolVLM-256M-Instruct-Q8_0.gguf",
	"newarch/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf", "newarch/EXAONE-4.0-1.2B-Q4_K_M.gguf", "newarch/AFM-4.5B-Q4_K_M.gguf",
	"granite/granite-3.3-2b-instruct-Q4_K_M.gguf", "granite/granite-4.0-micro-Q4_K_M.gguf",
	"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", "Phi-3.5-mini-instruct-Q4_K_M.gguf",
	"gemma-3-1b-it-Q4_K_M.gguf", "gemma-3-4b-it-Q4_K_M.gguf", "gemma-2-2b-it-Q4_K_M.gguf",
	"falcon/falcon-7b-instruct.Q4_K_M.gguf",
	"Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
	"bitnet-b1.58-2B-4T-i2_s.gguf", "synth-deepseek-dense", "synth-deepseek-yarn.gguf", "synth-deepseek-yarn-af.gguf",
	"synth-apertus.gguf",
	"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", "DeepSeek-V2-Lite.Q4_K_M.gguf",
	"synth-glm4moe.gguf", "synth-qwen2moe.gguf", "synth-ernie45moe.gguf", "synth-hunyuanmoe.gguf",
	"synth-minimaxm2.gguf", "synth-bailingmoe2.gguf", "synth-dots1.gguf", "synth-gptoss.gguf", "synth-deepseek",
	"synth-deepseek-lite", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", "synth-kimilinear@q8",
	"newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "synth-llama4-q8.gguf", "synth-pixtral.gguf", "synth-hunyuanvl.gguf",
	"hunyuanvl/HunyuanOCR-Q8_0.gguf"}

// formedModels is what the front end states and the serving gate does not
// run: a model the tier cannot place (stories260K, unblockedModels),
// float fixtures and checkpoints whose composed logits
// equal the float reference's, which the gate's band cannot measure
// (Qwen3-VL-2B's f16: its blocks and head run their float form, median ratio
// 0), and rnj-1, whose
// composed blocks reach 0.19 from the float reference on CUDA at its 43rd
// teacher-forced step, past four times the tiers' widest (0.0425): open. The
// step reads as a near tie (one block's int8 rounding alone moves it 0.016,
// main's CUDA tier lands at 0.029), and its last block composed alone matches
// the tier's logits to 3e-14 at the median step, but CUDA's split attention,
// which Vulkan does not split at this key bound, is not yet ruled out.
// Qwen3.5's mixture fixture at Q8_0 serves in the band (median 1.17 of the
// tier's distance on CUDA, 1.35 on Vulkan, its prompt's blocks and every
// decode step's served) but the gate cannot discriminate on it: the random
// hybrid is so sensitive that the tier's own backends sit 0.28 and 0.34 from
// the float reference, the arm 0.34 from the tier on Vulkan, and one
// position off reads 2.61, under ten times that. The float fixtures of dbrx,
// mistral3 and Llama 4 are served against the host with each feature taken
// out (TestC6ServesComposed, TestSynthLlama4ServesComposed). LFM2-1.2B serves
// every arm on Vulkan (median 0.974 of the tier's distance) and reads 1.503 on
// CUDA, past the 1.5 band: open. The composed arm sits as far from the float
// reference on both backends (0.142 and 0.143 at its worst); it is CUDA's
// tier that sits closer (0.091, Vulkan's 0.109). No block carries it: with
// the first 4, 8, 12 and 16 blocks composed CUDA reads 1.282, 1.434, 1.505
// and 1.503, the last 4, 8 and 12 1.278, 1.215 and 1.186, the head alone
// 1.000. Shadowed on the tier's own input, every composed block sits at most
// 2.9e-04 from the tier's output (two of them to 5e-13) where the tier
// sits 5.7e-04 to 2.6e-03 from the float reference, alike on both
// backends: a rounding a tenth of the tier's own, amplified through the
// sixteen blocks to a trajectory CUDA's tier happens to keep closer.
var formedModels = []string{"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
	"stories260K.gguf", "synth-ernie45.gguf", "synth-phi4v.gguf", "synth-seedoss.gguf", "synth-smollm3.gguf",
	"synth-olmo2.gguf", "synth-olmo2-gqa.gguf", "newarch/synth-olmo2-q8.gguf", "synth-arcee.gguf", "synth-exaone4.gguf",
	"synth-olmo3.gguf", "synth-phi3", "synth-glm4.gguf", "rnj/EssentialAI_rnj-1-instruct-Q4_K_M.gguf",
	"synth-phi2.gguf", "synth-starcoder.gguf", "synth-stablelm.gguf", "synth-stablelm-par.gguf",
	"synth-starcoder2.gguf", "synth-falcon.gguf", "synth-falcon40.gguf", "synth-nemotron.gguf",
	"synth-commandr.gguf", "synth-cohere2.gguf",
	"synth-qwen2vl.gguf", "synth-qwen25vl.gguf", "synth-qwen3vl.gguf",
	"Qwen3-VL-2B-Instruct-f16.gguf", "synth-bitnet-f16.gguf", "synth-bitnet-i2s.gguf",
	"synth-qwen35moe-hybrid-mtp-q8_0.gguf", "synth-dbrx.gguf", "synth-llama4.gguf", "synth-mistral3.gguf",
	"synth-mistral3-af.gguf", "lfm2/LFM2-1.2B-Q4_K_M.gguf", "synth-gemma4.gguf", "synth-gemma4-kv.gguf", "synth-gemma4-wide.gguf",
	"synth-gemma4-e.gguf", "synth-gemma4-e-ragged.gguf", "synth-gemma4-moe.gguf", "gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf",
	"synth-deepseek32.gguf", "synth-minimaxm3.gguf"}

// mixtureModels are the mixtures the front end states whole and the serving
// gate does not run: tiny-qwen3moe, whose eight-wide rows the tier cannot
// place (unblockedModels), and synth-phimoe, a float fixture whose
// sparsemixer weights each pick by a stated softmax over the logits near it
// (TestMoEFamiliesOnEveryDevice serves it composed against the host).
var mixtureModels = []string{"tiny-qwen3moe-f32.gguf", "synth-phimoe.gguf"}

// unblockedModels are the models whose rows the tier's int8 activation
// blocks (32 wide) do not divide: stories260K's 172-wide MLP and
// tiny-qwen3moe's 8-wide residual. The device refuses their blocks at
// placement, by that name.
var unblockedModels = []string{"stories260K.gguf", "tiny-qwen3moe-f32.gguf"}

// TestTierNamesTheRowsItCannotBlock places each of unblockedModels on every
// device, its blocks composed: the device takes none and names the row.
func TestTierNamesTheRowsItCannotBlock(t *testing.T) {
	ran := 0
	for _, file := range unblockedModels {
		t.Run(file, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(file)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
						tier.WithConfig(func(c *tier.Config) { c.Composed = tier.ComposedStep })}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					s := m.NewState(64)
					defer s.Close()
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if n, why := s.GPULayers(), g.Err(); n != 0 || !strings.Contains(why, "int8 activation blocks of 32") {
						t.Fatalf("%s took %d of %d blocks (%q), want none, refused by the row", dev, n, m.Cfg.NLayer, why)
					}
					t.Logf("%s refuses every block: %s", dev, g.Err())
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}

// declinesOf is every decline the tier counted, the head's and the blocks',
// by name (tier.DeclineName).
func declinesOf(st tier.Stats) string {
	var out []string
	for i := range st.HeadDeclines {
		if k := st.HeadDeclines[i]; k > 0 {
			out = append(out, fmt.Sprintf("head %d: %s", k, tier.DeclineName(i)))
		}
		if k := st.BlockDeclines[i]; k > 0 {
			out = append(out, fmt.Sprintf("blocks %d: %s", k, tier.DeclineName(i)))
		}
		if k := st.StepDeclines[i]; k > 0 {
			out = append(out, fmt.Sprintf("tables and tokens %d: %s", k, tier.DeclineName(i)))
		}
	}
	if len(out) == 0 {
		return "none counted"
	}
	return strings.Join(out, "; ")
}

// composedResidue is what the tier still launches of its own on a one-row
// step whose tables, blocks and head the composer serves: nothing.
var composedResidue []string

// bandRatio and bandWorst hold a composed arm to the spread of the tier's own
// backends: at the median step it may be no further from the float reference
// than bandRatio times the tier, and at its worst no further than bandWorst
// times the widest the tier's backends come at theirs. Both gates that hold an
// arm to a reference take them from here.
const bandRatio, bandWorst = 1.5, 4.0

// ssmServed are the state-space models served (servedModels), their
// prompts a chunk's rows the scan's. A selective update
// with one decay a head beside attention and an MLP (granite-4.0-h, and
// Falcon-H1's mixer beside its attention in one block) and the gated short
// convolution beside attention (LFM2), Mamba-1's per-channel update alone
// (mamba-130m) and beside attention (Jamba), and Mamba-2's mixer layers
// beside attention and MLP layers (Nemotron-Nano-9B-v2), alone
// (mamba-codestral) and beside a mixture (synth-nemotronhmoe-q8), a block
// with no MLP written apart from x (apartModels). falcon-mamba-7B serves
// apart and is not here: its Vulkan arm reads 0.029 at its worst against 4
// times the tier's 0.00291 (median 0.959), one step 30 times its neighbours.
// The step is sensitive, not a block: on CUDA, with the first 48, the last
// 32, the first 32, the first 16 and all 64 blocks composed, the 132nd step
// reads 0.0236, 0.0184, 0.0015, 0.0006 and 0.0007 from the float reference
// (the tier 0.00066), tracking no block and no depth.
var ssmServed = []string{"lfm2/LFM2-350M-Q4_K_M.gguf", "granite/granite-4.0-h-350m-Q8_0.gguf",
	"granite/granite-4.0-h-1b-Q4_K_M.gguf", "falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
	"falconh1/Falcon-H1-1.5B-Instruct-Q8_0.gguf", "mamba/mamba-130m-hf.Q8_0.gguf",
	"jamba/jamba-reasoning-3b-Q4_K_M.gguf", "nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
	"mamba2/mamba-codestral-7b-v0.1-q4_k_m.gguf", "synth-nemotronhmoe-q8.gguf"}

func init() {
	servedModels = append(servedModels, ssmServed...)
}

// apartModels are the state-space checkpoints whose blocks with no MLP end
// in a kernel that adds the projection to x and writes the output: a kernel
// reading x where it writes is not written over it -- its surplus threads
// are clamped to the last element and the lanes storing an element read it,
// some after another's store (RULE 13) -- so the cards write the output
// apart, in a scratch of their own, and the composer's copy puts it over x
// (graph.Copy). Written over x, mamba-130m parted from its float reference
// by 3.3 times the tier's distance and changed under poisoned buffers,
// falcon-mamba-7B by 80, mamba-codestral went non-finite at its 103rd step,
// and Nemotron-Nano-9B-v2 changed under poisoned buffers. synth-nemotronhmoe-q8,
// Nemotron-H's mixture fixture, has such a block a quarter of its steps.
var apartModels = []string{"mamba/mamba-130m-hf.Q8_0.gguf", "falconmamba/falcon-mamba-7B-Q4_K_M.gguf",
	"mamba2/mamba-codestral-7b-v0.1-q4_k_m.gguf", "nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
	"synth-nemotronhmoe-q8.gguf"}

// TestComposedBlocksServeApart decodes each apartModels checkpoint with
// every block and the head asked of the composer on every device: every
// block step served, nothing declined, the blocks whose form reads x where
// it writes served apart and copied over x (a copy served for each), and
// the logits finite and unchanged under poisoned buffers.
func TestComposedBlocksServeApart(t *testing.T) {
	ran := 0
	for _, file := range apartModels {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(path)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
			toks := make([]int32, 24)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					parts := tier.ComposedBlock | tier.ComposedHead
					got, st, _, _ := composedDecode(t, m, dev, ids, toks, parts, true, false)
					served, declined, _ := st.ServedPart(tier.ComposedBlock)
					copies, _, _ := st.ServedPart(tier.PartCopy)
					if served != len(toks)*m.Cfg.NLayer || declined != 0 {
						t.Fatalf("%d block steps served and %d declined of %d: %s",
							served, declined, len(toks)*m.Cfg.NLayer, declinesOf(st))
					}
					if copies == 0 {
						t.Fatalf("no copy over x in %d block steps: the gate selected no block written apart", served)
					}
					poisoned, _, _, _ := composedDecode(t, m, dev, ids, toks, parts, true, false, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))
					for i := range got {
						for _, v := range got[i] {
							if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
								t.Fatalf("step %d: the logits are not finite", i)
							}
						}
						if !slices.Equal(poisoned[i], got[i]) {
							t.Fatalf("step %d: with every buffer poisoned the logits change: a kernel reads what nothing wrote", i)
						}
					}
					t.Logf("%d block steps served, %d of them written apart and copied over x", served, copies)
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}

// trainedToks are n teacher-forced ids after the prompt ids, each the first
// id at or after (ids[i]*7 + i) whose embedding row stores something: a row
// of zeros is a token the model never trained (mamba-codestral's Q4_K_M has
// 277 such rows, 19 and every sixth id after among them), and through a gated
// mixer it zeroes every block's output -- silu of a zero gate -- so every
// arm's logits are exactly zero and no distance between them is defined.
func trainedToks(t *testing.T, m *Model, ids []int32, n int) ([]int32, int) {
	t.Helper()
	row := make([]float32, m.embd.k)
	nonzero := func(v float32) bool { return v != 0 }
	toks, moved := make([]int32, n), 0
	for i := range toks {
		id := (int(ids[i%len(ids)])*7 + i) % m.Vocab.Size()
		for tries := 0; ; tries++ {
			if err := m.embedRow(row, id); err != nil {
				t.Fatal(err)
			}
			if slices.ContainsFunc(row, nonzero) {
				break
			}
			if tries == m.Vocab.Size() {
				t.Fatal("no id stores an embedding")
			}
			if tries == 0 {
				moved++
			}
			id = (id + 1) % m.Vocab.Size()
		}
		toks[i] = int32(id)
	}
	return toks, moved
}

// crossingPrompt is the serving gate's prompt: more than one chunk of the
// rows the arms reserve (64) and several blocks of positions of the prompt's
// attention on the matrix unit, so its rows' running maximum and sum are
// rescaled from block to block -- a prompt inside one block multiplies
// every rescale by zero.
const crossingPrompt = "The clerk counted barrels of salt on the upper floor of the old customs house, " +
	"and the river rose every spring until the bridges were islands. The town kept its boats tied to the " +
	"church railings, and the children learned to row before they could read. When the water finally fell, " +
	"the merchants climbed down from their attics, opened their shutters, and argued about the price of salt " +
	"until the next flood came. The capital of France is"

// flashBlock is the positions a block of the prompt's attention on the
// matrix unit takes (jit's gpu profile, Flash.Keys).
const flashBlock = 32

// TestComposedBlocksServe decodes dense models with each block served whole
// by the composer from its graph (the model's front end's, from the
// container), the prompt's blocks at the chunk's rows, and
// the head, in main's runtime (Config.Composed), against the tier's own
// kernels: every block placed, with and without the captured graph, the
// prompt's last row -- a prompt of two chunks crossing several blocks of
// positions (crossingPrompt) -- and 140 teacher-forced steps (past the
// first key bound),
// and on a model whose sliding window cuts inside its context, steps past
// the window. Each arm is held to a float reference -- the same blocks and
// head formed with float activations, which neither the tier nor the arm
// computes: from the first
// decoded step its distance from it may be no more than bandRatio times the
// tier's at the median step, and no more than bandWorst times the widest the
// tier's own backends come at their worst: a model whose decode passes a
// near tie (OLMo 2's fifth step) sends each implementation to one side of it
// or the other, main's CUDA and Vulkan tiers 0.118 apart there, and the arm
// is held to that spread, not to the side one backend's tier landed on.
// Stats().ServedPart with ReplayedPart is the selection check, every block
// of every step served whether emitted or replayed, a poisoned arm must change
// nothing, and one position off must be ten times the arm's worst: the
// comparison discriminates.
func TestComposedBlocksServe(t *testing.T) {
	type arm struct {
		name  string
		parts tier.Parts
		tile  int
	}
	arms := []arm{{"whole block", tier.ComposedBlock, 0}, {"whole step", tier.ComposedStep, 0}}
	for k := 2; k <= matrixTileCount(); k++ {
		arms = append(arms, arm{fmt.Sprintf("whole step at matrix tile %d", k), tier.ComposedStep, k})
	}
	ran := 0
	for _, file := range servedModels {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(path)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode(crossingPrompt, true)
			if len(ids) <= 2*flashBlock {
				t.Fatalf("the prompt is %d tokens: it does not cross several blocks of %d positions", len(ids), flashBlock)
			}
			n := 140
			// A window the context can outrun is decoded past, so it cuts:
			// the window's last position and windowPast more.
			const windowPast = 64
			w, c := 0, int(m.Cfg.NCtx)
			for li := range m.Cfg.NLayer {
				if m.Cfg.SWA(li) {
					w = m.Cfg.SWAWindow
				}
			}
			if w > 0 && (c <= 0 || w+windowPast+len(ids)+4 <= c) {
				n = max(n, w+windowPast)
			}
			if c > 0 {
				n = min(n, c-len(ids)-4)
			}
			if w > 0 && len(ids)+n > w {
				t.Logf("the decode runs %d positions past the window of %d", len(ids)+n-w, w)
			} else if w > 0 {
				t.Logf("the window of %d cuts nothing in the model's context of %d", w, c)
			}
			toks, moved := trainedToks(t, m, ids, n)
			if moved > 0 {
				t.Logf("%d teacher-forced ids store no embedding and took the next id that does", moved)
			}
			// The tier and the float reference on every device first: the
			// spread of the tier's own backends is the band each arm is held to.
			type tierRun struct{ want, ref [][]float32 }
			runs := map[string]*[2]tierRun{}
			var widest [2]float64
			for _, dev := range stepDevices() {
				// A device the box does not have skips its own subtest, not the model's.
				t.Run(dev+"/reference", func(t *testing.T) {
					r := &[2]tierRun{}
					for g, noGraph := range []bool{false, true} {
						r[g].want, _, _, _ = composedDecode(t, m, dev, ids, toks, 0, noGraph, false)
						r[g].ref, _, _, _ = composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead|tier.ComposedPrompt, noGraph, true)
						for i := 1; i < len(r[g].ref); i++ {
							widest[g] = max(widest[g], logitNMSE(r[g].want[i], r[g].ref[i]))
						}
					}
					runs[dev] = r
				})
			}
			for _, dev := range stepDevices() {
				if runs[dev] == nil {
					continue
				}
				t.Run(dev, func(t *testing.T) {
					for g, noGraph := range []bool{false, true} {
						want, ref := runs[dev][g].want, runs[dev][g].ref
						for _, a := range arms {
							if a.tile > 0 && noGraph {
								continue
							}
							name := a.name
							if noGraph {
								name = "no graph/" + a.name
							}
							t.Run(name, func(t *testing.T) {
								parts := a.parts | tier.ComposedPrompt
								got, st, headOn, census := composedDecode(t, m, dev, ids, toks, parts, noGraph, false,
									tier.WithConfig(func(c *tier.Config) { c.MatrixTile = a.tile }))
								if a.tile > 0 && st.MatrixPart(tier.ComposedPrompt) > 0 && st.MatrixTiled == 0 {
									t.Fatalf("the prompt ran on the matrix unit with no part formed at tile %d", a.tile)
								}
								// Every form was built at the placement or PrepHead.
								if st.FormedAtStep != 0 {
									t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
								}
								// Every part the arm asks for served, and none declined: a
								// head the device does not hold (it did not fit beside the
								// blocks) is named, not demanded.
								served, _, _ := st.ServedPart(parts)
								for _, k := range []tier.Parts{tier.ComposedTable, tier.ComposedBlock, tier.ComposedHead, tier.ComposedPrompt} {
									if parts&k == 0 {
										continue
									}
									n, d, _ := st.ServedPart(k)
									if k == tier.ComposedHead && !headOn {
										t.Logf("the head runs at home here (it does not fit beside the blocks): the composed head has nothing to serve")
										continue
									}
									if n == 0 || d != 0 {
										t.Fatalf("composed parts %#x served %d steps and were declined %d: the arm did not run them (declines: %s; last error %q)", uint8(k), n, d, declinesOf(st), st.LastErr)
									}
								}
								// Every step runs every block, emitted or replayed in a
								// recording, so a block the composer did not serve shows
								// in the count.
								blocks, _, _ := st.ServedPart(tier.ComposedBlock)
								if replayed := st.ReplayedPart(tier.ComposedBlock); blocks+replayed != len(toks)*m.Cfg.NLayer {
									t.Fatalf("the composer ran %d block steps (%d of them replayed) where %d steps of %d blocks are %d",
										blocks+replayed, replayed, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer)
								}
								// Every step writes every table the plan states.
								if a.parts&tier.ComposedTable != 0 {
									nt, _, tableUnasked := st.ServedPart(tier.ComposedTable)
									if replayed := st.ReplayedPart(tier.ComposedTable); nt+replayed != len(toks)*composedTables(m) {
										t.Fatalf("the composer wrote %d tables (%d of them replayed) in %d steps of %d tables; %d capture(s), %d unasked, last error %q",
											nt+replayed, replayed, len(toks), composedTables(m), st.Captures, tableUnasked, st.LastErr)
									}
								}
								// Every step's input row gathered by the composer, from
								// the head where it is the embedding and from the token
								// embedding the device holds beside it where it is not
								// (tinyllama, stories15M), and the prompt's rows too.
								if a.parts&tier.ComposedEmbed != 0 {
									ne, de, _ := st.ServedPart(tier.ComposedEmbed)
									ne += st.ReplayedPart(tier.ComposedEmbed)
									np, dp, _ := st.ServedPart(tier.PromptEmbed)
									switch {
									case ne+de != len(toks):
										t.Fatalf("the input rows of %d steps served %d times and declined %d", len(toks), ne, de)
									case headOn && ne != len(toks):
										t.Fatalf("the composer gathered %d of %d steps' rows (%s)", ne, len(toks), declinesOf(st))
									case headOn && (np == 0 || dp != 0):
										t.Fatalf("the composer gathered the prompt's rows %d times and declined them %d (%s)", np, dp, declinesOf(st))
									}
									t.Logf("composed %s: %d input rows gathered, %d declined, the prompt's %d (%s)", a.name, ne, de, np, declinesOf(st))
								}
								// Every step's head, and the prompt's once.
								wantHeads := len(toks) + 1
								heads, headDeclined, headUnasked := st.ServedPart(tier.ComposedHead)
								if replayed := st.ReplayedPart(tier.ComposedHead); a.parts&tier.ComposedHead != 0 && headOn &&
									heads+replayed != wantHeads {
									t.Fatalf("the composer ran the head %d times (%d of them replayed) in %d steps and the prompt (%d asked); %d declined, %d unasked, %d folds, %d capture(s)",
										heads+replayed, replayed, len(toks), parts&tier.ComposedPrompt, headDeclined, headUnasked, st.HeadFolds, st.Captures)
								}
								if noGraph {
									t.Logf("composed %s: the tier's own launches on the steps it served: %v", a.name, census)
									// With the blocks and the head the composer's, what the tier
									// still launches on a step is the residual list, nothing more.
									if a.parts&tier.ComposedHead != 0 && headOn {
										for k := range census {
											if !slices.Contains(composedResidue, k) {
												t.Fatalf("the tier launched its own %s on a step the composer served whole (%v)", k, census)
											}
										}
									}
									if nb, _, _ := st.ServedPart(tier.ComposedBlock); nb != len(toks)*m.Cfg.NLayer {
										t.Fatalf("the composer served %d block steps where %d steps of %d blocks are %d",
											nb, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer)
									}
									if nh, _, _ := st.ServedPart(tier.ComposedHead); a.parts&tier.ComposedHead != 0 && headOn && nh != wantHeads {
										t.Fatalf("the composer served the head %d times in %d steps and the prompt (%d asked)", nh, len(toks), parts&tier.ComposedPrompt)
									}
								}
								// The prompt's every block step served, in whole chunks.
								if np, dp, up := st.ServedPart(tier.ComposedPrompt); np < m.Cfg.NLayer || np%m.Cfg.NLayer != 0 || dp != 0 || up != 0 {
									t.Fatalf("the composer served %d block steps of the prompt, declined %d and was not asked for %d, where every chunk is %d blocks (%s)",
										np, dp, up, m.Cfg.NLayer, st.LastErr)
								} else {
									t.Logf("composed %s: %d block steps of the prompt served", a.name, np)
								}
								worst := 0.0
								for i := range want {
									e := logitNMSE(got[i], want[i])
									if math.IsNaN(e) {
										t.Errorf("step %d: with the composed %s the logits are not finite", i, a.name)
									}
									worst = max(worst, e)
								}
								off := 0.0
								for i := 1; i < len(want); i++ {
									off = max(off, logitNMSE(got[i], want[i-1]))
								}
								// Step 0 is the prompt's last row: the composer's, held
								// to the band with every decoded step, the band's widest
								// still the tier's backends' over the decoded steps alone.
								// The prompt's input rows, where the composer gathers
								// them, are the same rows as the tier's and the CPU's.
								first := 0
								{
									var ratios []float64
									worstTier, worstGot := 0.0, 0.0
									for i := first; i < len(ref); i++ {
										et, eg := logitNMSE(want[i], ref[i]), logitNMSE(got[i], ref[i])
										if math.IsNaN(et) || math.IsNaN(eg) || !(et > 0) {
											t.Fatalf("step %d: the float reference is %.3g from the tier and %.3g from the arm: a degenerate reference proves nothing", i, et, eg)
										}
										ratios = append(ratios, eg/et)
										worstTier, worstGot = max(worstTier, et), max(worstGot, eg)
									}
									slices.Sort(ratios)
									med := ratios[len(ratios)/2]
									if !(med <= bandRatio) || !(worstGot <= bandWorst*widest[g]) {
										t.Errorf("against the float reference the composed %s is %.3g at the median of the tier's distance and %.3g at its worst, where the tier's backends reach %.3g (bounds %.1f and %.1fx)",
											a.name, med, worstGot, widest[g], bandRatio, bandWorst)
									}
									t.Logf("composed %s against the float reference: median ratio to the tier's distance %.3f (range %.3f..%.3f), worst %.3g against the tier's %.3g here and %.3g on its widest backend",
										a.name, med, ratios[0], ratios[len(ratios)-1], worstGot, worstTier, widest[g])
								}
								{
									var ratios []float64
									worstOff := 0.0
									for i := max(first, 1); i < len(ref); i++ {
										eg := logitNMSE(got[i-1], ref[i])
										ratios = append(ratios, eg/logitNMSE(want[i], ref[i]))
										worstOff = max(worstOff, eg)
									}
									slices.Sort(ratios)
									med := ratios[len(ratios)/2]
									if med <= bandRatio && worstOff <= bandWorst*widest[g] {
										t.Errorf("one position off the composed %s is %.3g at the median of the tier's distance and %.3g at its worst, inside the band (%.1f and %.1fx of %.3g): the gate cannot see a wrong part",
											a.name, med, worstOff, bandRatio, bandWorst, widest[g])
									}
									t.Logf("one position off the composed %s: median ratio %.3g, worst %.3g against the band's %.3g", a.name, med, worstOff, bandWorst*widest[g])
								}
								t.Logf("composed %s: %d part steps served, worst NMSE from the tier %.3g, one position off %.3g", a.name, served, worst, off)
								// A part the composer serves wrong must fall outside the
								// band the arm is held to: the State's statement of the
								// step changed under the served part, k the part.
								violated := func(what string, k tier.Parts, after func(*State)) {
									bad, bst, _, _ := composedDecodeWith(t, m, dev, ids, toks, parts, noGraph, false, after)
									if n, d, _ := bst.ServedPart(k); n == 0 || d != 0 {
										t.Fatalf("the arm with %s served the part %d times and declined it %d", what, n, d)
									}
									var ratios []float64
									worstBad := 0.0
									for i := first; i < len(ref); i++ {
										eg := logitNMSE(bad[i], ref[i])
										ratios = append(ratios, eg/logitNMSE(want[i], ref[i]))
										worstBad = max(worstBad, eg)
									}
									slices.Sort(ratios)
									med := ratios[len(ratios)/2]
									if med <= bandRatio && worstBad <= bandWorst*widest[g] {
										t.Fatalf("with %s the arm is %.3g at the median of the tier's distance and %.3g at its worst, inside the band (%.1f and %.1fx of %.3g): the gate cannot see it",
											what, med, worstBad, bandRatio, bandWorst, widest[g])
									}
									t.Logf("with %s: median ratio %.3g, worst %.3g against the band's %.3g", what, med, worstBad, bandWorst*widest[g])
								}
								// The served tables turn each row by the positions the
								// State states for it; the gathered row is the token's.
								// The positions are not one off: the band is relative to
								// the tier's own distance from the float reference, and
								// on stories15M that distance is wide (0.14 at its
								// worst), wider than one position moves a 288-wide
								// model's logits -- one position off reads a median 1.10
								// and 1.14 of the tier's distance there, inside the 1.5
								// band. Nor are they shifted alike: a rotary turns a
								// query and a key by their difference, and every row
								// decoded here is told its shift, so among the decoded
								// rows a uniform shift moves only the few prompt keys --
								// 16 past read a median 1.22 to 1.36 on Qwen2.5-VL-3B,
								// inside the band. Twice the position and 16 past moves
								// every distance. The M-RoPE gate holds its tables to
								// transformers itself, with no tier between, and one
								// position is seen there. A model whose attention
								// applies no positional encoding (FlagNoPosEnc,
								// Kimi-Linear's latent attention) reads no position:
								// there is nothing for the violation to move.
								if a.parts&tier.ComposedTable != 0 && !m.container.Config().Flags.Has(jlm.FlagNoPosEnc) {
									violated("every row told twice its position and 16 past", tier.ComposedTable, stretchAxes(2, 16))
								}
								if ne, _, _ := st.ServedPart(tier.ComposedEmbed); ne > 0 {
									violated("every row gathered at the next token's", tier.ComposedEmbed, shiftTokens(1, m.Vocab.Size()))
								}
								// Every buffer filled with NaN at allocation: a
								// composed kernel reading what nothing wrote shows
								// as a changed logit.
								if a.parts&tier.ComposedHead != 0 {
									poisoned, _, _, _ := composedDecode(t, m, dev, ids, toks, parts, noGraph, false, tier.WithPoisonScratch(true), tier.WithPoisonKV(true))
									for i := range got {
										if !slices.Equal(poisoned[i], got[i]) {
											t.Fatalf("step %d: with every buffer poisoned the logits change: a composed kernel reads what nothing wrote", i)
										}
									}
								}
								ran++
							})
						}
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}

// TestComposedSharedPoolMatchesTheDoubledOne decodes Qwen3.5-0.8B with its
// prompt and every step's blocks and head served by the composer, its
// recurrent pools in the shared form (Config.SharedRec: one state a slot,
// stepped in place through the composer's own descriptor) and in the doubled
// one, and demands the same logits bit for bit: the scan and the one-row
// rule read a slot and write it after, as they read one half and write the
// other. Stats.RecShared and the served counts are the selection check.
func TestComposedSharedPoolMatchesTheDoubledOne(t *testing.T) {
	const file = "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"
	path := testmodels.Path(file)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and the river rose", true)
	toks := m.Vocab.Encode(" every spring until the bridges were islands and the town kept its boats", false)
	parts := tier.ComposedBlock | tier.ComposedHead | tier.ComposedPrompt
	ran := 0
	for _, dev := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(dev, func(t *testing.T) {
			want, wst, _, _ := composedDecode(t, m, dev, ids, toks, parts, true, false)
			got, st, _, _ := composedDecode(t, m, dev, ids, toks, parts, true, false,
				tier.WithConfig(func(c *tier.Config) { c.SharedRec = true }))
			if wst.RecShared != 0 || st.RecShared == 0 {
				t.Fatalf("the doubled arm grew %d shared state(s) and the shared arm %d: the gate compared one form with itself",
					wst.RecShared, st.RecShared)
			}
			for _, s := range []tier.Stats{wst, st} {
				np, dp, _ := s.ServedPart(tier.ComposedPrompt)
				nb, db, _ := s.ServedPart(tier.ComposedBlock)
				if np < m.Cfg.NLayer || dp != 0 || nb < len(toks)*m.Cfg.NLayer || db != 0 {
					t.Fatalf("the composer served %d prompt block steps (%d declined) and %d decode block steps (%d declined): %s",
						np, dp, nb, db, declinesOf(s))
				}
			}
			for i := range want {
				if !slices.Equal(got[i], want[i]) {
					t.Fatalf("step %d: the shared pool's logits differ from the doubled one's (NMSE %.3g)", i, logitNMSE(got[i], want[i]))
				}
			}
			t.Logf("%d shared state(s); %d steps identical", st.RecShared, len(want))
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}

// TestComposedPromptIsNotServed holds a prompt's calls to the prompt
// (nn.PromptDevice) with the composer serving the blocks and the head. A
// float head does not fold into a batched chunk and a prompt of a few rows
// runs as one ragged step, so either way the prompt's last-row head is a
// one-row call of its own: it must count as unasked, never as a step the
// composer serves. Each decode step after it serves every block and the head
// once. With the State keeping the prompt from the device (the violation)
// the head is served once more than the steps, which the gate must see.
func TestComposedPromptIsNotServed(t *testing.T) {
	const file, steps = "synth-qwen3vl.gguf", 6
	path := testmodels.Path(file)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ran := 0
	for _, dev := range stepDevices() {
		// Three rows run as one ragged step, nine as a chunk whose float head
		// does not fold.
		for _, rows := range []int{3, 9} {
			t.Run(fmt.Sprintf("%s/%d rows", dev, rows), func(t *testing.T) {
				run := func(withhold bool) (heads, unasked, blocks int) {
					// Without a recording every step runs every part, so the
					// counts are the steps'.
					g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff),
						tier.WithConfig(func(c *tier.Config) {
							c.Composed, c.Census, c.NoGraph = tier.ComposedStep, true, true
						})},
						testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					s := m.NewState(rows + steps + 4)
					defer s.Close()
					s.SetDeviceFallback(false)
					if err := s.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if s.GPULayers() != m.Cfg.NLayer || !s.HeadOnDevice() {
						t.Fatalf("%s took %d of %d blocks, head on it %v: %s", dev, s.GPULayers(), m.Cfg.NLayer, s.HeadOnDevice(), g.Err())
					}
					if withhold {
						s.ldPrompt = nil
					}
					ids := make([]int32, rows)
					for i := range ids {
						ids[i] = int32((i*7 + 3) % m.Vocab.Size())
					}
					if _, err := s.Prefill(ids); err != nil {
						t.Fatal(err)
					}
					for i := range steps {
						if _, err := s.Forward(int32((i*11 + 5) % m.Vocab.Size())); err != nil {
							t.Fatal(err)
						}
					}
					st := g.Stats()
					heads, _, unasked = st.ServedPart(tier.ComposedHead)
					blocks, _, _ = st.ServedPart(tier.ComposedBlock)
					if !withhold {
						for k := range g.Census() {
							if !slices.Contains(composedResidue, k) {
								t.Fatalf("the tier launched its own %s on a call it counts as a step (%v)", k, g.Census())
							}
						}
					}
					return heads, unasked, blocks
				}
				heads, unasked, blocks := run(false)
				if heads != steps || unasked < 1 || blocks != steps*m.Cfg.NLayer {
					t.Fatalf("the head was served %d times and left unasked %d in %d steps, the blocks served %d times where %d steps of %d blocks are %d",
						heads, unasked, steps, blocks, steps, m.Cfg.NLayer, steps*m.Cfg.NLayer)
				}
				leaked, _, _ := run(true)
				if leaked != steps+1 {
					t.Fatalf("with the prompt kept from the device the head was served %d times in %d steps: the gate cannot see a prompt's head served", leaked, steps)
				}
				t.Logf("the head served %d times in %d steps, %d of its calls unasked; kept from the device, %d", heads, steps, unasked, leaked)
				ran++
			})
		}
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}
