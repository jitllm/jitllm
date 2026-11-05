package engine

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// composedDecode decodes toks teacher-forced after ids on a fresh tier of
// dev serving the parts, every block placed, and returns
// every step's logits, the tier's stats and its census of the launches of
// its own kernels on the steps the composer was asked for.
func composedDecode(t *testing.T, m *model.Model, io []Option, dev string, ids, toks []int32, parts nn.Parts, noGraph, float bool, extra ...gpu.Option) ([][]float32, gpu.Stats, bool) {
	t.Helper()
	return composedDecodeWith(t, m, io, dev, ids, toks, parts, noGraph, float, nil, extra...)
}

// unsplit forms a reference's sums with no split (Config.ComposedUnsplit):
// a split an arm forms wrong is not the reference's too.
func unsplit() gpu.Option { return gpu.WithConfig(func(c *gpu.Config) { c.ComposedUnsplit = true }) }

// unsplitRef is the float reference of parts on dev, formed with no split:
// its blocks all served, and every split's last segment dropped leaves it
// bit for bit, so a split an arm forms wrong is not the reference's too.
func unsplitRef(t *testing.T, m *model.Model, dev string, ids, toks []int32, parts nn.Parts, noGraph bool, io ...Option) [][]float32 {
	t.Helper()
	ref, st, _ := composedDecode(t, m, io, dev, ids, toks, parts, noGraph, true, unsplit())
	if n, d := st.ServedPart(nn.ComposedBlock); n == 0 || d != 0 {
		t.Fatalf("the float reference served its blocks %d times and declined them %d (%s)", n, d, declinesOf(st))
	}
	cut, _, _ := composedDecode(t, m, io, dev, ids, toks, parts, noGraph, true, unsplit(), gpu.WithSplitFault(true))
	for i := range cut {
		if !slices.Equal(cut[i], ref[i]) {
			t.Fatalf("step %d: with every split's last segment dropped the float reference moves (NMSE %.3g): it forms through the composer's splits", i, logitNMSE(cut[i], ref[i]))
		}
	}
	return ref
}

// composedTables is how many rotary tables the front end states for m: the
// tables a served step writes.
func composedTables(m *model.Model) int {
	n := 0
	for _, g := range m.FrontGraphs().Tables {
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

// shiftTokens has the instance promise its device every row's gather d tokens
// past its own.
func shiftTokens(d, vocab int) func(*Instance) { return shiftEmbed(d, 0, vocab) }

// shiftEmbed has the instance promise its device every row's gather d tokens
// and p positions past its own.
func shiftEmbed(d, p, vocab int) func(*Instance) {
	return func(s *Instance) {
		if s.ldEmbed != nil {
			s.ldEmbed = &tokenShift{EmbedDevice: s.ldEmbed, d: int32(d), vocab: int32(vocab), p: p}
		}
	}
}

// composedDecodeWith is composedDecode with after applied to the instance once
// its device is attached.
func composedDecodeWith(t *testing.T, m *model.Model, io []Option, dev string, ids, toks []int32, parts nn.Parts, noGraph, float bool, after func(*Instance), extra ...gpu.Option) ([][]float32, gpu.Stats, bool) {
	t.Helper()
	// The prompt is one chunk of a few rows: every arm reserves and warms the
	// widths up to 64, each a form of the composer's where it serves them.
	opts := append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) {
			c.NoGraph, c.ComposedFloat, c.PromptRows = noGraph, float, 64
		})},
		testTierOpts(t)...)
	opts = append(opts, extra...)
	g, err := openGPU(t, opts...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	s := testInstance(t, m, len(ids)+len(toks)+4, io...)
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
	return out, g.Stats(), s.HeadOnDevice()
}

// servedSince is the steps of the parts of k the composer served or a
// recording replayed between snapshots a and b.
func servedSince(a, b gpu.Stats, k nn.Parts) int {
	n0, _ := a.ServedPart(k)
	n1, _ := b.ServedPart(k)
	return n1 - n0 + b.ReplayedPart(k) - a.ReplayedPart(k)
}

// cpuDecode is the composed CPU's logits for the prompt ids and then each of
// toks, as composedDecode returns the card's: the reference a card's arm is
// held to.
func cpuDecode(t *testing.T, m *model.Model, ids, toks []int32, io ...Option) [][]float32 {
	t.Helper()
	s := testInstance(t, m, len(ids)+len(toks)+4, io...)
	defer s.Close()
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
	return append(out, append([]float32(nil), l...))
}

// servedModels is every checkpoint whose blocks and head the front end states
// as ops and that one card of 16 GiB holds: dense models of the llama, qwen2,
// qwen3, gemma, gemma2, gemma3, olmo2, exaone4, arcee, granite, hunyuan,
// phi3, falcon and apertus classes, the dense latent attention fixtures
// (DeepSeek-V3's block at f32, and under YaRN), and the text models of qwen2vl
// and qwen3vl. Those state their rotaries over several position axes, which
// the served table turns each row by as the instance states them; on a text row every axis's
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
func servedBaseModels() []string {
	return []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf", "tinyllama-1.1b-q3_K_M.gguf",
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
		"hunyuanvl/HunyuanOCR-Q8_0.gguf", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "gpt-oss-20b-MXFP4.gguf"}
}

// budgetElsewhere is the fixtures whose experts are too small for any budget
// to tell streamed from resident, each with the served model whose budget arm
// stands for its family's.
func budgetElsewhere() map[string]string {
	return map[string]string{"synth-gptoss.gguf": "gpt-oss-20b-MXFP4.gguf"}
}

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
func formedModels() []string {
	return []string{"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
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
}

// mixtureModels are the mixtures the front end states whole and the serving
// gate does not run: tiny-qwen3moe, whose eight-wide rows the tier cannot
// place (unblockedModels), and synth-phimoe, a float fixture whose
// sparsemixer weights each pick by a stated softmax over the logits near it
// (TestMoEFamiliesOnEveryDevice serves it composed against the host).
func mixtureModels() []string { return []string{"tiny-qwen3moe-f32.gguf", "synth-phimoe.gguf"} }

// unblockedModels are the models whose rows the tier's int8 activation
// blocks (32 wide) do not divide: stories260K's 172-wide MLP and
// tiny-qwen3moe's 8-wide residual. The device refuses their blocks at
// placement, by that name.
func unblockedModels() []string { return []string{"stories260K.gguf", "tiny-qwen3moe-f32.gguf"} }

// TestTierNamesTheRowsItCannotBlock places each of unblockedModels on every
// device, its blocks composed: the device takes none and names the row.
func TestTierNamesTheRowsItCannotBlock(t *testing.T) {
	ran := 0
	for _, file := range unblockedModels() {
		t.Run(file, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(file)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					s := testInstance(t, m, 64)
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

// declinesOf is every decline the tier counted of the blocks, by name
// (gpu.DeclineName).
func declinesOf(st gpu.Stats) string {
	var out []string
	for i := range st.BlockDeclines() {
		if k := st.BlockDeclines()[i]; k > 0 {
			out = append(out, fmt.Sprintf("blocks %d: %s", k, gpu.DeclineName(i)))
		}
		if k := st.StepDeclines()[i]; k > 0 {
			out = append(out, fmt.Sprintf("tables and tokens %d: %s", k, gpu.DeclineName(i)))
		}
	}
	if len(out) == 0 {
		return "none counted"
	}
	return strings.Join(out, "; ")
}

// bandRatio and bandWorst hold a composed arm to a reference implementation's
// distance from the float reference: at the median step no further than
// bandRatio times the reference's, and at its worst no further than bandWorst
// times the widest the reference comes. The serving gate's reference is
// llama.cpp: each step the wider of its CUDA and CPU arms, teacher-forced on
// the panel; main's tier where llama.cpp does not read the model (the user's
// decision). Both gates that hold an arm to a reference take them from here.
const bandRatio, bandWorst = 1.5, 4.0

// bandFloor is the least distance from the float reference a model of
// layers blocks is held to, where its tier sits closer: what binary16
// attention operands alone reach. A block's attention rounds four operands
// (q, k, the exponentials, v), each a relative variance of u^2/3 (u = 2^-11)
// at its output, and the blocks add through the residual independently:
// layers*4*u^2/3, times the attention gate's tail factor of 6
// (jit's flashTail).
func bandFloor(layers int) float64 { return 6 * 4 * float64(layers) / 3 / (1 << 22) }

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
func ssmServed() []string {
	return []string{"lfm2/LFM2-350M-Q4_K_M.gguf", "granite/granite-4.0-h-350m-Q8_0.gguf",
		"granite/granite-4.0-h-1b-Q4_K_M.gguf", "falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
		"falconh1/Falcon-H1-1.5B-Instruct-Q8_0.gguf", "mamba/mamba-130m-hf.Q8_0.gguf",
		"jamba/jamba-reasoning-3b-Q4_K_M.gguf", "nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
		"mamba2/mamba-codestral-7b-v0.1-q4_k_m.gguf", "synth-nemotronhmoe-q8.gguf"}
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
func apartModels() []string {
	return []string{"mamba/mamba-130m-hf.Q8_0.gguf", "falconmamba/falcon-mamba-7B-Q4_K_M.gguf",
		"mamba2/mamba-codestral-7b-v0.1-q4_k_m.gguf", "nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
		"synth-nemotronhmoe-q8.gguf"}
}

// TestComposedBlocksServeApart decodes each apartModels checkpoint with
// every block and the head asked of the composer on every device: every
// block step served, nothing declined, the blocks whose form reads x where
// it writes served apart and copied over x (a copy served for each), and
// the logits finite and unchanged under poisoned buffers.
func TestComposedBlocksServeApart(t *testing.T) {
	ran := 0
	for _, file := range apartModels() {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(path)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := model.Open(jlmOf(t, path))
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
					parts := nn.ComposedBlock | nn.ComposedHead
					got, st, _ := composedDecode(t, m, nil, dev, ids, toks, parts, true, false)
					served, declined := st.ServedPart(nn.ComposedBlock)
					copies, _ := st.ServedPart(nn.CopiedOut)
					if served != len(toks)*m.Cfg.NLayer || declined != 0 {
						t.Fatalf("%d block steps served and %d declined of %d: %s",
							served, declined, len(toks)*m.Cfg.NLayer, declinesOf(st))
					}
					if copies == 0 {
						t.Fatalf("no copy over x in %d block steps: the gate selected no block written apart", served)
					}
					poisoned, _, _ := composedDecode(t, m, nil, dev, ids, toks, parts, true, false, gpu.WithPoisonScratch(true), gpu.WithPoisonKV(true))
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
func trainedToks(t *testing.T, m *model.Model, ids []int32, n int) ([]int32, int) {
	t.Helper()
	row := make([]float32, m.Embd.K)
	nonzero := func(v float32) bool { return v != 0 }
	toks, moved := make([]int32, n), 0
	for i := range toks {
		id := (int(ids[i%len(ids)])*7 + i) % m.Vocab.Size()
		for tries := 0; ; tries++ {
			if err := embedRowRef(m, row, id); err != nil {
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
// the head, in main's runtime, against the tier's own
// kernels: every block placed, with and without the captured graph, on each
// prompt of servePanel (each of two chunks crossing several blocks of
// positions) its last row and teacher-forced steps: 140 on crossingPrompt
// (past the first key bound, and on a model whose sliding window cuts inside
// its context, past the window) and panelSteps on the others. Each arm is
// held to a float reference -- the same blocks and head formed with float
// activations, which neither the tier nor the arm computes: its distance from
// it may be no more than bandRatio times the tier's at the median prompt's
// median step, and no more than bandWorst times the widest the
// tier's own backends and llama.cpp's come at their worst: a model whose decode passes a
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
		parts nn.Parts
		tile  int
		// stream puts a mixture's routed experts in pages on the card.
		stream streamArm
	}
	arms := []arm{{"whole step", nn.ComposedStep, 0, 0}}
	for k := 2; k <= matrixTileCount(); k++ {
		arms = append(arms, arm{fmt.Sprintf("whole step at matrix tile %d", k), nn.ComposedStep, k, 0})
	}
	arms = append(arms, arm{"whole step, experts streamed under a budget", nn.ComposedStep, 0, streamBudget},
		arm{"whole step, experts cached in pages", nn.ComposedStep, 0, streamCache})
	for f, by := range budgetElsewhere() {
		if !slices.Contains(servedModels(), by) {
			t.Fatalf("%s's budget arm runs on %s, which the gate does not serve", f, by)
		}
	}
	ran := 0
	for _, file := range servedModels() {
		t.Run(file, func(t *testing.T) {
			path := testmodels.Path(file)
			if _, err := os.Stat(fixtureSource(path)); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			// The budget arm streams blocks the card does not hold: the trial
			// against the CPU would move them mid-decode.
			instOpts := []Option{WithStreamTrial(false)}
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			// A window the context can outrun is decoded past on the first
			// prompt, so it cuts: the window's last position and windowPast more.
			const windowPast = 64
			w, c := 0, int(m.Cfg.NCtx)
			for li := range m.Cfg.NLayer {
				if m.Cfg.SWA(li) {
					w = m.Cfg.SWAWindow
				}
			}
			type prompt struct{ ids, toks []int32 }
			var panel []prompt
			for pi, text := range servePanel() {
				ids := m.Vocab.Encode(text, true)
				// A context the prompt and the panel's steps overrun keeps the
				// prompt's head, still across several blocks.
				if c > 0 && len(ids)+panelSteps+4 > c {
					keep := max(2*flashBlock+1, c-panelSteps-4)
					t.Logf("prompt %d: %d of its %d tokens in the model's context of %d", pi, keep, len(ids), c)
					ids = ids[:min(keep, len(ids))]
				}
				if len(ids) <= 2*flashBlock {
					t.Fatalf("prompt %d is %d tokens: it does not cross several blocks of %d positions", pi, len(ids), flashBlock)
				}
				n := panelSteps
				if pi == 0 {
					n = 140
					if w > 0 && (c <= 0 || w+windowPast+len(ids)+4 <= c) {
						n = max(n, w+windowPast)
					}
				}
				if c > 0 {
					n = min(n, c-len(ids)-4)
				}
				if n < 1 {
					t.Fatalf("prompt %d of %d tokens leaves no step in the model's context of %d", pi, len(ids), c)
				}
				if pi == 0 && w > 0 && len(ids)+n > w {
					t.Logf("the decode runs %d positions past the window of %d", len(ids)+n-w, w)
				} else if pi == 0 && w > 0 {
					t.Logf("the window of %d cuts nothing in the model's context of %d", w, c)
				}
				toks, moved := trainedToks(t, m, ids, n)
				if moved > 0 {
					t.Logf("prompt %d: %d teacher-forced ids store no embedding and took the next id that does", pi, moved)
				}
				panel = append(panel, prompt{ids, toks})
			}
			// The band's reference first: llama.cpp's cuda arm against the float
			// reference on every device, or the tier's where llama.cpp does not
			// read the model (the user's decision).
			lcppDir := os.Getenv("JITLLM_LCPP")
			if lcppDir == "" {
				t.Fatal("JITLLM_LCPP is unset: it names a CUDA build of llama.cpp holding scripts/lcpplogits, the worst-step bound's oracle")
			}
			var lcpp [][][]float32
			for _, ngl := range []int{99, 0} {
				for _, p := range panel {
					lg, ok := lcppSteps(t, lcppDir, fixtureSource(path), p.ids, p.toks, ngl)
					if !ok {
						t.Logf("llama.cpp does not read %s: the band is held to the tier's backends", file)
						lcpp = nil
						break
					}
					lcpp = append(lcpp, lg)
				}
				if lcpp == nil {
					break
				}
			}
			type tierRun struct{ want, cpu, ref [][]float32 }
			refDist := func(r tierRun, i int) float64 {
				e := logitNMSE(r.want[i], r.ref[i])
				if r.cpu != nil {
					e = max(e, logitNMSE(r.cpu[i], r.ref[i]))
				}
				return e
			}
			runs := map[string]*[2][]tierRun{}
			var widest [2]float64
			var widestLcpp [2]float64
			var lcppAt [2]string
			for _, dev := range stepDevices() {
				// A device the box does not have skips its own subtest, not the model's.
				t.Run(dev+"/reference", func(t *testing.T) {
					r := &[2][]tierRun{}
					for g, noGraph := range []bool{false, true} {
						for _, p := range panel {
							var tr tierRun
							pi := len(r[g])
							tr.ref = unsplitRef(t, m, dev, p.ids, p.toks, nn.ComposedStep, noGraph, instOpts...)
							// Where llama.cpp does not read the model the band is
							// the floor: the reference stands for itself.
							tr.want = tr.ref
							if lcpp != nil {
								tr.want = lcpp[pi]
								if len(lcpp) >= 2*len(panel) {
									tr.cpu = lcpp[len(panel)+pi]
								}
							}
							for li := pi; li < len(lcpp); li += len(panel) {
								if len(lcpp[li][0]) != len(tr.ref[0]) {
									t.Fatalf("llama.cpp's logits are %d wide where the model's are %d", len(lcpp[li][0]), len(tr.ref[0]))
								}
								for i := 1; i < len(tr.ref); i++ {
									if e := logitNMSE(lcpp[li][i], tr.ref[i]); e > widestLcpp[g] {
										widestLcpp[g], lcppAt[g] = e, fmt.Sprintf("%s, prompt %d step %d", [2]string{"cuda", "cpu"}[li/len(panel)], pi, i)
									}
								}
							}
							widest[g] = widestLcpp[g]
							r[g] = append(r[g], tr)
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
						floor := bandFloor(m.Cfg.NLayer)
						wide := max(widest[g], floor)
						if lcpp != nil {
							t.Logf("graph %v: the band's reference is the wider of llama.cpp's cuda and cpu arms, which reach %.3g at their worst step (%s)", !noGraph, widestLcpp[g], lcppAt[g])
						} else {
							t.Logf("graph %v: llama.cpp does not read this model: the band is the floor, %.3g", !noGraph, floor)
						}
						for _, a := range arms {
							if a.tile > 0 && noGraph || a.stream != 0 && (noGraph || m.Cfg.NExpert == 0) {
								continue
							}
							if a.stream == streamBudget && budgetElsewhere()[file] != "" {
								t.Logf("%s: its budget arm runs on %s", a.name, budgetElsewhere()[file])
								continue
							}
							name := a.name
							if noGraph {
								name = "no graph/" + a.name
							}
							t.Run(name, func(t *testing.T) {
								parts := a.parts | nn.ComposedPrompt
								tile := gpu.WithConfig(func(c *gpu.Config) { c.MatrixTile = a.tile })
								ctx := 0
								for _, p := range panel {
									ctx = max(ctx, len(p.ids)+len(p.toks)+4)
								}
								sopts := streamOpts(t, m, dev, a.stream, parts, noGraph, ctx, instOpts...)
								var band, offBand panelBand
								var st0 gpu.Stats
								var got0 [][]float32
								for pi, p := range panel {
									ids, toks := p.ids, p.toks
									want, ref := runs[dev][g][pi].want, runs[dev][g][pi].ref
									got, st, headOn := composedDecode(t, m, instOpts, dev, ids, toks, parts, noGraph, false, append([]gpu.Option{tile}, sopts...)...)
									if pi == 0 {
										st0, got0 = st, got
									}
									if a.stream != 0 {
										streamSelected(t, st, a.stream)
									}
									if a.tile > 0 && st.MatrixPart(nn.ComposedPrompt) > 0 && st.MatrixTiled == 0 {
										t.Fatalf("the prompt ran on the matrix unit with no part formed at tile %d", a.tile)
									}
									// Every form was built at the placement or PrepHead.
									if st.FormedAtStep != 0 {
										t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
									}
									// Every part the arm asks for served, and none declined: a
									// head the device does not hold (it did not fit beside the
									// blocks) is named, not demanded.
									for _, k := range []nn.Parts{nn.ComposedTable, nn.ComposedBlock, nn.ComposedHead, nn.ComposedPrompt} {
										if parts&k == 0 {
											continue
										}
										n, d := st.ServedPart(k)
										if k == nn.ComposedHead && !headOn {
											if pi == 0 {
												t.Logf("the head runs at home here (it does not fit beside the blocks): the composed head has nothing to serve")
											}
											continue
										}
										if n == 0 || d != 0 {
											t.Fatalf("composed parts %#x served %d steps and were declined %d: the arm did not run them (declines: %s; last error %q)", uint8(k), n, d, declinesOf(st), st.LastErr)
										}
									}
									// Every step runs every block, emitted or replayed in a
									// recording, so a block the composer did not serve shows
									// in the count.
									blocks, _ := st.ServedPart(nn.ComposedBlock)
									if replayed := st.ReplayedPart(nn.ComposedBlock); blocks+replayed != len(toks)*m.Cfg.NLayer {
										t.Fatalf("the composer ran %d block steps (%d of them replayed) where %d steps of %d blocks are %d",
											blocks+replayed, replayed, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer)
									}
									// Every step and every prompt chunk writes every table
									// the plan states, none declined; a
									// prompt's tail too short for a chunk runs as a step and
									// writes its own.
									if a.parts&nn.ComposedTable != 0 {
										nt, tableDeclined := st.ServedPart(nn.ComposedTable)
										np, _ := st.ServedPart(nn.ComposedPrompt)
										calls := len(toks) + np/m.Cfg.NLayer
										if replayed := st.ReplayedPart(nn.ComposedTable); tableDeclined != 0 || nt+replayed < calls*composedTables(m) {
											t.Fatalf("the composer wrote %d tables (%d of them replayed) in %d steps and %d prompt chunks of %d tables; %d declined, %d capture(s), last error %q",
												nt+replayed, replayed, len(toks), np/m.Cfg.NLayer, composedTables(m), tableDeclined, st.Captures, st.LastErr)
										}
									}
									// Every step's input row gathered by the composer, from
									// the head where it is the embedding and from the token
									// embedding the device holds beside it where it is not
									// (tinyllama, stories15M), and the prompt's rows too.
									if a.parts&nn.ComposedEmbed != 0 {
										ne, de := st.ServedPart(nn.ComposedEmbed)
										ne += st.ReplayedPart(nn.ComposedEmbed)
										np, dp := st.ServedPart(nn.PromptEmbed)
										switch {
										case ne+de != len(toks):
											t.Fatalf("the input rows of %d steps served %d times and declined %d", len(toks), ne, de)
										case headOn && ne != len(toks):
											t.Fatalf("the composer gathered %d of %d steps' rows (%s)", ne, len(toks), declinesOf(st))
										case headOn && (np == 0 || dp != 0):
											t.Fatalf("the composer gathered the prompt's rows %d times and declined them %d (%s)", np, dp, declinesOf(st))
										}
										if pi == 0 {
											t.Logf("composed %s: %d input rows gathered, %d declined, the prompt's %d (%s)", a.name, ne, de, np, declinesOf(st))
										}
									}
									// Every step's head, and the prompt's once.
									wantHeads := len(toks) + 1
									heads, headDeclined := st.ServedPart(nn.ComposedHead)
									if replayed := st.ReplayedPart(nn.ComposedHead); a.parts&nn.ComposedHead != 0 && headOn &&
										heads+replayed != wantHeads {
										t.Fatalf("the composer ran the head %d times (%d of them replayed) in %d steps and the prompt (%d asked); %d declined, %d folds, %d capture(s)",
											heads+replayed, replayed, len(toks), parts&nn.ComposedPrompt, headDeclined, st.HeadFolds, st.Captures)
									}
									if noGraph {
										if nb, _ := st.ServedPart(nn.ComposedBlock); nb != len(toks)*m.Cfg.NLayer {
											t.Fatalf("the composer served %d block steps where %d steps of %d blocks are %d",
												nb, len(toks), m.Cfg.NLayer, len(toks)*m.Cfg.NLayer)
										}
										if nh, _ := st.ServedPart(nn.ComposedHead); a.parts&nn.ComposedHead != 0 && headOn && nh != wantHeads {
											t.Fatalf("the composer served the head %d times in %d steps and the prompt (%d asked)", nh, len(toks), parts&nn.ComposedPrompt)
										}
									}
									// The prompt's every block step served, in whole chunks.
									if np, dp := st.ServedPart(nn.ComposedPrompt); np < m.Cfg.NLayer || np%m.Cfg.NLayer != 0 || dp != 0 {
										t.Fatalf("the composer served %d block steps of the prompt and declined %d, where every chunk is %d blocks (%s)",
											np, dp, m.Cfg.NLayer, st.LastErr)
									}
									for i := range want {
										if math.IsNaN(logitNMSE(got[i], want[i])) {
											t.Errorf("prompt %d step %d: with the composed %s the logits are not finite", pi, i, a.name)
										}
									}
									// Step 0 is the prompt's last row: the composer's, held
									// to the band with every decoded step, the band's widest
									// still the tier's backends' over the decoded steps alone.
									var ratios []float64
									worstGot := 0.0
									for i := range ref {
										et, eg := refDist(runs[dev][g][pi], i), logitNMSE(got[i], ref[i])
										if math.IsNaN(et) || math.IsNaN(eg) || !(et > 0) {
											t.Fatalf("prompt %d step %d: the float reference is %.3g from the tier and %.3g from the arm: a degenerate reference proves nothing", pi, i, et, eg)
										}
										ratios = append(ratios, eg/max(et, floor))
										worstGot = max(worstGot, eg)
									}
									band.add(ratios, worstGot)
									ratios, worstOff := nil, 0.0
									for i := 1; i < len(ref); i++ {
										eg := logitNMSE(got[i-1], ref[i])
										ratios = append(ratios, eg/max(refDist(runs[dev][g][pi], i), floor))
										worstOff = max(worstOff, eg)
									}
									offBand.add(ratios, worstOff)
								}
								if !band.inside(wide) {
									t.Errorf("against the float reference the composed %s is %.3g at the median prompt's median of the reference's distance and %.3g at its worst, where the reference reaches %.3g (bounds %.1f and %.1fx)",
										a.name, band.median(), band.worst, wide, bandRatio, bandWorst)
								}
								t.Logf("composed %s against the float reference: median ratio to the reference's distance %.3f (per prompt %.3f), worst %.3g against %.3g, the reference's widest",
									a.name, band.median(), band.meds, band.worst, wide)

								if offBand.inside(wide) {
									t.Errorf("one position off the composed %s is %.3g at the median prompt's median of the reference's distance and %.3g at its worst, inside the band (%.1f and %.1fx of %.3g): the gate cannot see a wrong part",
										a.name, offBand.median(), offBand.worst, bandRatio, bandWorst, wide)
								}
								t.Logf("one position off the composed %s: median ratio %.3g (per prompt %.3g), worst %.3g against the band's %.3g", a.name, offBand.median(), offBand.meds, offBand.worst, bandWorst*wide)
								// A part the composer serves wrong must fall outside the
								// band the arm is held to on the panel: the instance's
								// statement of the step changed under the served part,
								// k the part.
								violated := func(what string, k nn.Parts, after func(*Instance), extra ...gpu.Option) {
									var bad panelBand
									for pi, p := range panel {
										ref := runs[dev][g][pi].ref
										out, bst, _ := composedDecodeWith(t, m, instOpts, dev, p.ids, p.toks, parts, noGraph, false, after, slices.Concat(sopts, extra)...)
										if n, d := bst.ServedPart(k); n == 0 || d != 0 {
											t.Fatalf("the arm with %s served the part %d times and declined it %d", what, n, d)
										}
										var ratios []float64
										worstBad := 0.0
										for i := range ref {
											eg := logitNMSE(out[i], ref[i])
											ratios = append(ratios, eg/max(refDist(runs[dev][g][pi], i), floor))
											worstBad = max(worstBad, eg)
										}
										bad.add(ratios, worstBad)
									}
									if bad.inside(wide) {
										t.Fatalf("with %s the arm is %.3g at the median prompt's median of the reference's distance and %.3g at its worst, inside the band (%.1f and %.1fx of %.3g): the gate cannot see it",
											what, bad.median(), bad.worst, bandRatio, bandWorst, wide)
									}
									t.Logf("with %s: median ratio %.3g (per prompt %.3g), worst %.3g against the band's %.3g", what, bad.median(), bad.meds, bad.worst, bandWorst*wide)
								}
								// The served tables turn each row by the positions the
								// Instance states for it; the gathered row is the token's.
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
								if a.parts&nn.ComposedTable != 0 && !m.Container.Config().Flags.Has(jlm.FlagNoPosEnc) {
									violated("every row's table at twice its position and 16 past", nn.ComposedTable, nil, gpu.WithRopeFault(2, 16))
								}
								if ne, _ := st0.ServedPart(nn.ComposedEmbed); ne > 0 {
									violated("every row gathered at the next token's", nn.ComposedEmbed, shiftTokens(1, m.Vocab.Size()))
								}
								violated("every split's last segment dropped", nn.ComposedBlock, nil, tile, gpu.WithSplitFault(true))
								if a.stream != 0 {
									violated("every page table swapping two experts", nn.ComposedBlock, nil, gpu.WithPageFault(gpu.PageFaultSwap))
									violated("every block's pages left as the last call filled them", nn.ComposedBlock, nil, gpu.WithPageFault(gpu.PageFaultStale))
								}
								// Every buffer filled with NaN at allocation: a
								// composed kernel reading what nothing wrote shows
								// as a changed logit.
								if a.parts&nn.ComposedHead != 0 {
									poisoned, _, _ := composedDecode(t, m, instOpts, dev, panel[0].ids, panel[0].toks, parts, noGraph, false, slices.Concat([]gpu.Option{tile, gpu.WithPoisonScratch(true), gpu.WithPoisonKV(true)}, sopts)...)
									for i := range got0 {
										if !slices.Equal(poisoned[i], got0[i]) {
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
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and the river rose", true)
	toks := m.Vocab.Encode(" every spring until the bridges were islands and the town kept its boats", false)
	parts := nn.ComposedBlock | nn.ComposedHead | nn.ComposedPrompt
	ran := 0
	for _, dev := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(dev, func(t *testing.T) {
			want, wst, _ := composedDecode(t, m, nil, dev, ids, toks, parts, true, false)
			got, st, _ := composedDecode(t, m, nil, dev, ids, toks, parts, true, false,
				gpu.WithConfig(func(c *gpu.Config) { c.SharedRec = true }))
			if wst.RecShared != 0 || st.RecShared == 0 {
				t.Fatalf("the doubled arm grew %d shared state(s) and the shared arm %d: the gate compared one form with itself",
					wst.RecShared, st.RecShared)
			}
			for _, s := range []gpu.Stats{wst, st} {
				np, dp := s.ServedPart(nn.ComposedPrompt)
				nb, db := s.ServedPart(nn.ComposedBlock)
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
// one-row call of its own, which the composed head serves: no call is left
// declined. Each decode step after it serves every block once; the
// prompt reaches the device as a prompt, never as steps. With the instance not
// telling the device the calls are a prompt's (the violation) a ragged prompt
// runs as a step, which the gate must see; a chunk is a prompt by its shape,
// and its run must not change.
func TestComposedPromptIsNotServed(t *testing.T) {
	const file, steps = "synth-qwen3vl.gguf", 6
	path := testmodels.Path(file)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ran := 0
	for _, dev := range stepDevices() {
		// Three rows run as one ragged step, nine as a chunk whose float head
		// does not fold.
		for _, rows := range []int{3, 9} {
			ragged := rows == 3
			t.Run(fmt.Sprintf("%s/%d rows", dev, rows), func(t *testing.T) {
				run := func(withhold bool) (heads, left, blocks, prompt int) {
					// Without a recording every step runs every part, so the
					// counts are the steps'.
					g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
						gpu.WithConfig(func(c *gpu.Config) {
							c.NoGraph = true
						})},
						testTierOpts(t)...)...)
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					s := testInstance(t, m, rows+steps+4)
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
					heads, declined := st.ServedPart(nn.ComposedHead)
					left = declined
					blocks, _ = st.ServedPart(nn.ComposedBlock)
					var pd int
					prompt, pd = st.ServedPart(nn.ComposedPrompt)
					left += pd
					return heads, left, blocks, prompt
				}
				asPrompt := func(blocks, prompt int) bool { return blocks == steps*m.Cfg.NLayer && prompt > 0 }
				heads, left, blocks, prompt := run(false)
				if heads != steps+1 || left != 0 || !asPrompt(blocks, prompt) {
					t.Fatalf("the head was served %d times and the head or prompt left to the tier %d in %d steps and the prompt; the blocks served %d times where %d steps of %d blocks are %d, and the prompt part %d times",
						heads, left, steps, blocks, steps, m.Cfg.NLayer, steps*m.Cfg.NLayer, prompt)
				}
				_, _, wb, wp := run(true)
				switch {
				case ragged && asPrompt(wb, wp):
					t.Fatalf("with the prompt's calls not marked the blocks were served %d times in %d steps of %d blocks and the prompt part %d times: the gate cannot see a ragged prompt run as a step",
						wb, steps, m.Cfg.NLayer, wp)
				case !ragged && (wb != blocks || wp != prompt):
					t.Fatalf("with the chunk's calls not marked the blocks were served %d times and the prompt part %d, against %d and %d: a chunk's run depends on the mark",
						wb, wp, blocks, prompt)
				}
				t.Logf("the head served %d times in %d steps and the prompt, the blocks %d and the prompt part %d; unmarked, %d and %d", heads, steps, blocks, prompt, wb, wp)
				ran++
			})
		}
	}
	if ran == 0 {
		t.Fatal("no device ran")
	}
}
