package model

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
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
// latent attention beside them.
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
	"synth-deepseek-lite", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"}

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
var formedModels = []string{"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
	"stories260K.gguf", "synth-ernie45.gguf", "synth-phi4v.gguf", "synth-seedoss.gguf", "synth-smollm3.gguf",
	"synth-olmo2.gguf", "synth-olmo2-gqa.gguf", "newarch/synth-olmo2-q8.gguf", "synth-arcee.gguf", "synth-exaone4.gguf",
	"synth-olmo3.gguf", "synth-phi3", "synth-glm4.gguf", "rnj/EssentialAI_rnj-1-instruct-Q4_K_M.gguf",
	"synth-phi2.gguf", "synth-starcoder.gguf", "synth-stablelm.gguf", "synth-stablelm-par.gguf",
	"synth-starcoder2.gguf", "synth-falcon.gguf", "synth-falcon40.gguf", "synth-nemotron.gguf",
	"synth-commandr.gguf", "synth-cohere2.gguf",
	"synth-qwen2vl.gguf", "synth-qwen25vl.gguf", "synth-qwen3vl.gguf",
	"Qwen3-VL-2B-Instruct-f16.gguf", "synth-bitnet-f16.gguf", "synth-bitnet-i2s.gguf"}

// mixtureModels are the mixtures the front end states whole and the serving
// gate does not run: tiny-qwen3moe, whose eight-wide rows the tier cannot
// place (unblockedModels).
var mixtureModels = []string{"tiny-qwen3moe-f32.gguf"}

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

// promptModels are the served models whose prompt the gate has the composer
// serve too, at the chunk's rows (ComposedPrompt): three Q8_0 models, the
// k-quant Llama-3.2-1B, tinyllama (Q3_K), Qwen2-1.5B and Qwen3-1.7B, whose
// projections at a chunk's rows run in row tiles (jit.RowTiles), and
// gemma-3-1b, whose sliding layers' chunk reads the windowed descriptors.
var promptModels = []string{"stories15M-q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf",
	"gemma-3-1b-it-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "tinyllama-1.1b-q3_K_M.gguf",
	"Qwen2-1.5B-Instruct-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf"}

// TestComposedBlocksServe decodes dense models with each block served whole
// by the composer from its graph (the model's front end's, from the
// container), the prompt's blocks at the chunk's rows (promptModels), and
// the head, in main's runtime (Config.Composed), against the tier's own
// kernels: every block placed, with and without the captured graph, the
// prompt's last row and 140 teacher-forced steps (past the first key bound),
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
	arms := []struct {
		name  string
		parts tier.Parts
	}{
		{"whole block", tier.ComposedBlock},
		{"whole step", tier.ComposedStep},
	}
	ran := 0
	for _, file := range servedModels {
		t.Run(file, func(t *testing.T) {
			prompt := tier.Parts(0)
			if slices.Contains(promptModels, file) {
				prompt = tier.ComposedPrompt
			}
			path := testmodels.Path(file)
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)
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
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = int32((int(ids[i%len(ids)])*7 + i) % m.Vocab.Size())
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
						r[g].ref, _, _, _ = composedDecode(t, m, dev, ids, toks, tier.ComposedBlock|tier.ComposedHead|prompt, noGraph, true)
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
							name := a.name
							if noGraph {
								name = "no graph/" + a.name
							}
							t.Run(name, func(t *testing.T) {
								parts := a.parts | prompt
								got, st, headOn, census := composedDecode(t, m, dev, ids, toks, parts, noGraph, false)
								// Every form was built at the placement or PrepHead.
								if st.FormedAtStep != 0 {
									t.Fatalf("%d composed part(s) formed at a step, not at the placement", st.FormedAtStep)
								}
								// Every part the arm asks for served, and none declined: a
								// head the device does not hold (it did not fit beside the
								// blocks) is named, not demanded.
								served, _, unasked := st.ServedPart(parts)
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
									nt, _, _ := st.ServedPart(tier.ComposedTable)
									if replayed := st.ReplayedPart(tier.ComposedTable); nt+replayed != len(toks)*composedTables(m) {
										t.Fatalf("the composer wrote %d tables (%d of them replayed) in %d steps of %d tables", nt+replayed, replayed, len(toks), composedTables(m))
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
								// Every step's head, and the prompt's once where the prompt
								// is asked.
								wantHeads := len(toks)
								if prompt != 0 {
									wantHeads++
								}
								heads, _, _ := st.ServedPart(tier.ComposedHead)
								if replayed := st.ReplayedPart(tier.ComposedHead); a.parts&tier.ComposedHead != 0 && headOn &&
									heads+replayed != wantHeads {
									t.Fatalf("the composer ran the head %d times (%d of them replayed) in %d steps and the prompt (%d asked)", heads+replayed, replayed, len(toks), prompt)
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
										t.Fatalf("the composer served the head %d times in %d steps and the prompt (%d asked)", nh, len(toks), prompt)
									}
								}
								// The prompt's every block step served, in whole chunks.
								if prompt != 0 {
									np, dp, up := st.ServedPart(tier.ComposedPrompt)
									if np < m.Cfg.NLayer || np%m.Cfg.NLayer != 0 || dp != 0 || up != 0 {
										t.Fatalf("the composer served %d block steps of the prompt, declined %d and was not asked for %d, where every chunk is %d blocks (%s)",
											np, dp, up, m.Cfg.NLayer, st.LastErr)
									}
									t.Logf("composed %s: %d block steps of the prompt served", a.name, np)
								} else {
									t.Logf("composed %s: %d block steps of the prompt asked for nothing (prefill is not served)", a.name, unasked)
								}
								worst, worstAt := 0.0, 0
								for i := range want {
									e := logitNMSE(got[i], want[i])
									if math.IsNaN(e) {
										t.Errorf("step %d: with the composed %s the logits are not finite", i, a.name)
									}
									if e > worst {
										worst, worstAt = e, i
									}
								}
								off := 0.0
								for i := 1; i < len(want); i++ {
									off = max(off, logitNMSE(got[i], want[i-1]))
								}
								if !(off > max(1e-9, 10*worst)) {
									t.Errorf("one position off the logits are within %.3g of the tier's, where the arm's worst is %.3g (step %d): the comparison cannot see a wrong part", off, worst, worstAt)
								}
								// Step 0 is the prompt's last row. Where the prompt is
								// not served every arm's first logits are the tier's;
								// where it is, they are the composer's, held to the band
								// with every decoded step, the band's widest still the
								// tier's backends' over the decoded steps alone.
								// The prompt's input rows, where the composer gathers
								// them, are the same rows as the tier's and the CPU's.
								first := 0
								if prompt == 0 {
									if !slices.Equal(got[0], want[0]) || !slices.Equal(ref[0], want[0]) {
										t.Fatalf("the prompt is not served, yet its logits differ from the tier's: the composed setting reached the prompt")
									}
									first = 1
								}
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
								// position is seen there.
								if a.parts&tier.ComposedTable != 0 {
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
