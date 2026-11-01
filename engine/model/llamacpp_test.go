package model_test

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// tieMargin is how close the top two logits must be for a disagreement with
// llama.cpp to be acceptable. It duplicates the constant in forward_test.go,
// which is in the internal test package.
const tieMargin = 0.5

// partedEarly reports whether jitllm's chain parted from llama.cpp's ids at
// token i-1 although the text still matched there: llama.cpp's token extends
// jitllm's as text (" then" against " the"), so the text comparison sees the
// parting one token late. want is llama.cpp's own ids for the continuation.
func partedEarly(m *model.Model, out, want []int32, i int) bool {
	if i == 0 || i > len(want) || !slices.Equal(out[:i-1], want[:i-1]) || out[i-1] == want[i-1] {
		return false
	}
	a, b := m.Vocab.Decode(out[i-1:i]), m.Vocab.Decode(want[i-1:i])
	return len(b) > len(a) && strings.HasPrefix(b, a)
}

// sidesWithTheSessionRun reports whether jitllm's chain, where it left
// llama.cpp's plain run's text at token i, is token for token the run that
// saved the session (want) through token i. The greedy chain is compared as
// text against the plain run, and the session run is a second llama.cpp answer
// whose reduction order differs; where the two part, llama.cpp disagrees with
// itself and a chain that follows one of them has no oracle to fail against.
// google_gemma-3-4b-it-Q4_K_M at "The capital of France is", token 14: the
// plain CPU run says "," (and on to " on the Seine River."), the session run
// and llama.cpp's CUDA run say ".", jitllm "." by 0.879.
func sidesWithTheSessionRun(out, want []int32, i int) bool {
	return i < len(out) && i < len(want) && slices.Equal(out[:i+1], want[:i+1])
}

// recordedNearTies are teacher-forced positions where no two engines agree,
// so a margin there says how one host rounded rather than whether the graph
// is right. Apertus-8B's token 11 after "The capital of France is Paris,
// which is also the country's largest city.": llama.cpp " region" on an x86
// Xeon and " urban" on the M4, transformers' float32 checkpoint " not" with
// " Paris" outside its top five, jitllm " thus" at a 0.19 margin on the Xeon
// (its f16 cache) and " Paris" at 0.80 on the M4 (its f32 cache), " best" at
// 0.87 with the M4's cache at f16. Its residual stream reaches 2.6e4, and the
// two hosts' 7e-4 apart after layer 0's MLP are 16% apart at the output norm
// (docs/engineering-history/model-correctness.md, "Apertus").
var recordedNearTies = map[string]map[string]int{
	"swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M.gguf": {"The capital of France is": 11},
}

// TestGreedyMatchesLlamaCpp decodes the same prompt on jitllm and on llama.cpp
// and requires the text to agree. It is the test that catches a wrong graph:
// the internal reference is an oracle for kernels, not for the architecture,
// and only an independent implementation of the model is. It decodes 20 tokens
// because a wrong graph can get the first few right.
func TestGreedyMatchesLlamaCpp(t *testing.T) {
	lcpp := os.Getenv("JITLLM_LCPP")
	if lcpp == "" {
		t.Skip("set JITLLM_LCPP to a llama.cpp build directory holding llama-completion")
	}
	bin := lcppBin(lcpp, "llama-completion")
	if _, err := os.Stat(bin); err != nil {
		t.Skipf("llama-completion is not in JITLLM_LCPP (%s)", lcpp)
	}
	// The oracle (llama-completion) reads GGUF, so a host that keeps only
	// containers cannot run the comparison; say so once rather than per model.
	if g := testmodels.Glob("*.gguf"); len(g) == 0 {
		t.Skip("no GGUF on this host and llama-completion reads GGUF: the oracle " +
			"comparison belongs on a box that keeps the converter's input (RULE 7m)")
	}
	paths := languageModels(t)
	// JITLLM_GREEDY_PAIR=<gguf>=<container> holds a container converted from
	// safetensors (e.g. Kimi-Linear) against a GGUF used only as the oracle.
	var pair map[string]string
	if p := os.Getenv("JITLLM_GREEDY_PAIR"); p != "" {
		g, c, ok := strings.Cut(p, "=")
		if !ok {
			t.Fatalf("JITLLM_GREEDY_PAIR=%q: want <gguf>=<container>", p)
		}
		g, c = testmodels.Resolve(g), testmodels.Resolve(c)
		paths, pair = []string{g}, map[string]string{g: c}
	}
	// Several prompts, because a tie at token 0 compares nothing and one prompt
	// cannot avoid that for every model (gemma-2-2b-it ties " Paris" against
	// ":" by 0.045). A token-0 tie moves to the next prompt; the gate fails only
	// when every prompt ties there.
	prompts := []string{
		"The capital of France is",
		"Once upon a time",
		"The largest planet in our solar system is",
	}
	const n = 20

	ran := 0
	for _, path := range paths {
		name := filepath.Base(path)
		if strings.Contains(name, "stories") || strings.Contains(name, "260K") {
			continue // toy models babble; there is nothing to agree about
		}
		// synth-llama4's chunk and temperature floor are 4, which llama.cpp
		// hardcodes at 8192, so llama.cpp runs a different model. Its oracle is
		// transformers (TestSynthLlama4MatchesTransformers).
		if strings.HasPrefix(name, "synth-llama4") {
			continue
		}
		// synth-exaone4 slides a window on three layers in four, as the 32B
		// does; llama.cpp's exaone4 turns sliding off for any model that is not
		// 64 layers deep, so on four layers it runs every layer global and
		// rotated -- a different model. Its oracle is transformers
		// (TestC6MatchesTransformers); the real 1.2B, which slides nothing,
		// is held to llama.cpp here.
		if strings.HasPrefix(name, "synth-exaone4") {
			continue
		}
		// A phimoe model is PhimoeForCausalLM, whose LayerNorm llama.cpp runs
		// as an RMSNorm plus a bias and whose sparsemixer router it runs as a
		// renormalised softmax top-2 -- a different model (RULE 7m, chosen:
		// the class). synth-phimoe's oracle is transformers
		// (TestMoEFamiliesMatchTransformers); the real checkpoints' is
		// transformers on the GGUF's own weights (scripts/phimoestream.py).
		// Read off the file, not the name: Phi-mini-MoE carries no "phimoe"
		// in its name and parted from llama.cpp at a 2.7 margin.
		if f, err := gguf.Open(path); err == nil {
			arch, _ := f.KV["general.architecture"].String()
			f.Close()
			if arch == "phimoe" {
				continue
			}
		}
		// The -af fixtures state YaRN's attention_factor, which llama.cpp's
		// converter writes as rope.scaling.yarn_attn_factor and its runtime
		// never reads -- a different model (RULE 7m). Their oracle is
		// transformers (TestC6MatchesTransformers).
		if strings.HasPrefix(name, "synth-") && strings.HasSuffix(name, "-af.gguf") {
			continue
		}
		if testing.Short() && !strings.Contains(name, "Qwen") {
			continue
		}
		// Models over 2 GiB are skipped by default so the package fits go
		// test's timeout. JITLLM_SLOW=1 runs everything; JITLLM_GREEDY_MODEL
		// names one model (e.g. deepseek2, whose smallest checkpoint is large).
		if pick := os.Getenv("JITLLM_GREEDY_MODEL"); pick != "" {
			if !strings.Contains(name, pick) {
				continue
			}
		} else if fi, err := os.Stat(path); pair == nil && err == nil && fi.Size() > 2<<30 && os.Getenv("JITLLM_SLOW") == "" {
			t.Logf("skipping %s (%.1f GiB); set JITLLM_SLOW=1 (or JITLLM_GREEDY_MODEL=%s)",
				name, float64(fi.Size())/(1<<30), name[:min(len(name), 12)])
			continue
		}
		t.Run(name, func(t *testing.T) {
			// jlmOf skips on convert.ErrNotImplemented (an out-of-scope GGUF)
			// and fails on any other conversion error; `ran` stops an all-skip
			// run from reading as a pass.
			c := pair[path]
			if c == "" {
				c = jlmOf(t, path)
			}
			// Compare with the exact GEMM arithmetic: the integer super-block
			// form (arm64, pre-VNNI amd64) drifts prefill logits enough to flip
			// ties, and the ratchet below exists to catch drift. The integer
			// form is held to this one by TestPrefillMatchesForward and
			// TestBatchMatchesForward.
			// Tuning off, so the ratchet below measures the engine and not a
			// per-process kernel choice: on arm64 a timed choice once
			// flipped a near-tie.
			opts := []model.Option{model.WithJITOptions(nn.WithGEMMExact(true), nn.WithTune(nn.TuneOff))}
			// Hunyuan-A13B runs on an f32 KV cache. At "The capital of
			// France is"'s tenth position its 64-expert router holds the 8th
			// and 9th experts 1e-4 apart in most blocks, and which side the
			// cascade falls on is decided by precision, not graph: transformers
			// on the file's own weights says "<th", as llama.cpp's default
			// (flash attention) and jitllm's f32 cache do; jitllm's f16 cache
			// and llama.cpp without flash attention say " capital", and the
			// final margin, 10.5, is no measure of how near the tie sat
			// upstream (model-correctness.md, "hunyuan"). The oracle runs its
			// default, so jitllm runs the width that agrees with the reference
			// there; every other position, and the f16 cache on every other
			// model, is held as before.
			if strings.HasPrefix(name, "Hunyuan-A13B") {
				opts = append(opts, model.WithKVF16(false))
			}
			m, err := model.Open(c, opts...)
			if err != nil {
				t.Skipf("open: %v", err)
			}
			defer m.Close()
			if m.Vocab == nil {
				t.Skipf("no tokenizer: %v", m.TokErr)
			}
			// One prompt is enough as soon as it compares a token; a tie at
			// position 0 compares none, so fall through to the next.
			var ties []string
			tie := func(p, why string) { ties = append(ties, fmt.Sprintf("%q: %s", p, why)) }
			for _, prompt := range prompts {
				ids := m.Vocab.Encode(prompt, true)
				st := m.NewState(len(ids) + n + 1)
				defer st.Close()
				logits, err := st.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				out := make([]int32, 0, n)
				margins := make([]float32, 0, n)
				for i := 0; i < n; i++ {
					best, second, bi := float32(math.Inf(-1)), float32(math.Inf(-1)), int32(0)
					for j, v := range logits {
						if v > best {
							best, second, bi = v, best, int32(j)
						} else if v > second {
							second = v
						}
					}
					if bi == m.Vocab.EOS {
						break
					}
					out = append(out, bi)
					margins = append(margins, best-second)
					if logits, err = st.Forward(bi); err != nil {
						t.Fatal(err)
					}
				}

				// Offload llama.cpp when the model fits the card; its CPU and
				// CUDA paths agree on the graph, and offload is faster.
				ngl := "99"
				if fi, err := os.Stat(path); err == nil && fi.Size() > 3<<30/2 {
					ngl = "0"
				}
				// -c 512 is required: otherwise llama.cpp allocates the training
				// context (131072 for some models), tens of GB of KV cache. A
				// failed offload falls back to the CPU. sess != "" saves the ids
				// to that file (--prompt-cache-all).
				//
				// llama.cpp runs MiniMax Sparse Attention only under flash
				// attention and otherwise attends to every block -- a different
				// model. Its default, -fa auto, turns flash attention off when
				// the layer's device has no kernel for the head width (CUDA
				// at synth-minimaxm3's 16), and says so only in its log, so an
				// MSA model forces it on and a log that still reports it off
				// is refused.
				msa := m.Cfg.MSA()
				llama := func(sess string) []byte {
					for _, g := range []string{ngl, "0"} {
						args := []string{"-m", path, "-p", prompt, "-n", itoa(n),
							"--temp", "0", "-no-cnv", "--no-warmup", "-ngl", g, "-c", "512"}
						if msa {
							args = append(args, "-fa", "on")
						}
						if sess != "" {
							os.Remove(sess) // a stale session would be LOADED, not overwritten
							args = append(args, "--prompt-cache", sess, "--prompt-cache-all")
						}
						cmd := exec.Command(bin, args...)
						cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+filepath.Dir(bin)+string(os.PathListSeparator)+os.Getenv("LD_LIBRARY_PATH"))
						var log strings.Builder
						cmd.Stderr = &log
						if out, e := cmd.Output(); e == nil {
							if msa && strings.Contains(log.String(), "Flash Attention not supported, set to disabled") {
								t.Fatalf("llama.cpp turned flash attention off (-ngl %s), so it ran "+
									"MiniMax Sparse Attention as dense attention: not this model", g)
							}
							return out
						}
					}
					t.Skip("llama-completion would not run this model")
					return nil
				}
				// Its stdout is the prompt plus the completion, wrapped over any
				// number of lines, and it appends "[end of text]" on EOS -- jitllm
				// stops there silently, so that marker is a difference in reporting.
				text := strings.TrimSpace(strings.ReplaceAll(string(llama("")), "[end of text]", ""))
				if !strings.HasPrefix(text, prompt) {
					tie(prompt, "llama-completion did not echo the prompt")
					continue
				}

				// The teacher-forced golden is llama.cpp's own token ids, read
				// back from its session file. Re-tokenizing its text is not the
				// inverse of decoding (a model may emit " JO"+"B" where the
				// canonical split is " JOB"; random-weight fixtures do often).
				//
				// It is a second run because saving a session changes the first
				// logits' reduction order; the teacher-forced arm tolerates the
				// resulting ties, the greedy ratchet below does not, so the
				// ratchet reads the ordinary run.
				sess := filepath.Join(t.TempDir(), "session.bin")
				llama(sess)
				lids, err := sessionTokens(sess)
				if err != nil {
					t.Fatalf("llama.cpp's session file: %v", err)
				}
				// The prompt halves must match, or a tokenizer defect would be
				// reported as a graph one.
				if len(lids) < len(ids) || !slices.Equal(lids[:len(ids)], ids) {
					t.Errorf("%q: llama.cpp tokenized the prompt as %v, jitllm as %v",
						prompt, lids[:min(len(ids), len(lids))], ids)
					return
				}
				want := lids[len(ids):]

				// Teacher-forced: feed llama.cpp's own ids and compare at every
				// position. The greedy prefix below stops at the first near-tie,
				// so on a chaotic model it measures tie luck; here both engines
				// see identical input and a tie costs one position. It compares
				// argmax with a margin because only llama.cpp's ids, not its
				// logits, are available.
				if len(want) > 0 {
					fst := m.NewState(len(ids) + len(want) + 1)
					fl, ferr := fst.Prefill(ids)
					if ferr != nil {
						t.Fatal(ferr)
					}
					agreed := 0
					for i := 0; i < len(want); i++ {
						bi, best, second := int32(0), float32(math.Inf(-1)), float32(math.Inf(-1))
						for j, v := range fl {
							if v > best {
								best, second, bi = v, best, int32(j)
							} else if v > second {
								second = v
							}
						}
						pos, recorded := recordedNearTies[name][prompt]
						switch {
						case bi == want[i]:
							agreed++
						case recorded && pos == i:
							t.Logf("teacher-forced position %d: jitllm %q (margin %.3f), llama.cpp %q -- a "+
								"recorded near-tie (recordedNearTies)", i, m.Vocab.Text(bi), best-second, m.Vocab.Text(want[i]))
						case best-second > tieMargin:
							t.Errorf("teacher-forced position %d: jitllm %q (margin %.3f), "+
								"llama.cpp %q -- fed llama.cpp's own ids, so both engines saw "+
								"an identical input here and a CONFIDENT disagreement is a wrong "+
								"answer rather than a chain that parted earlier",
								i, m.Vocab.Text(bi), best-second, m.Vocab.Text(want[i]))
						}
						if fl, ferr = fst.Forward(want[i]); ferr != nil {
							t.Fatal(ferr)
						}
					}
					fst.Close()
					t.Logf("teacher-forced: %d of %d positions agree with llama.cpp", agreed, len(want))
				}

				// The greedy chain is compared as text against the ordinary run,
				// agreement being a prefix relationship.
				covered := 0
				for i := 0; i < len(out); i++ {
					got := prompt + m.Vocab.Decode(out[:i+1])
					if strings.HasPrefix(text, got) {
						covered++
						continue
					}
					if strings.HasPrefix(got, text) {
						break // llama.cpp stopped (EOS) or its tail was trimmed inside this token
					}
					rest := text[min(len(text), len(prompt+m.Vocab.Decode(out[:i]))):]
					rest = rest[:min(len(rest), 24)]
					// The text matched through token i-1 only as a prefix where
					// llama.cpp's token extends jitllm's: ERNIE's " the" against
					// llama.cpp's " then" passes as text and the parting surfaces
					// one token late, at "following" by 3.6, where the decision
					// was " the" against " then" at 0.40. The verdict is the token
					// that parted, read off llama.cpp's ids.
					if partedEarly(m, out, want, i) {
						i--
						covered = i
						rest = text[min(len(text), len(prompt+m.Vocab.Decode(out[:i]))):]
						rest = rest[:min(len(rest), 24)]
					}
					// A disagreement is only a bug if the margin is wide: the
					// engines quantize and sum differently, so near-ties may go
					// either way.
					if pos, ok := recordedNearTies[name][prompt]; ok && pos == i && i > 0 {
						t.Logf("%q: diverged at token %d, a recorded near-tie (%q at %.3f vs llama.cpp's %q...)",
							prompt, i, m.Vocab.Text(out[i]), margins[i], rest)
						break
					}
					if sidesWithTheSessionRun(out, want, i) {
						t.Logf("%q: token %d: llama.cpp's plain run says %q... where its session run says %q, as jitllm "+
							"does by %.3f: llama.cpp disagrees with itself here; compared up to there",
							prompt, i, rest, m.Vocab.Text(out[i]), margins[i])
						covered = i
						break
					}
					if margins[i] <= tieMargin {
						// A tie at position 0 compares nothing, so it counts as
						// a tie for this prompt rather than a pass.
						if i == 0 {
							tie(prompt, fmt.Sprintf("tied at token 0 by %.3f (%q vs llama.cpp's %q...)",
								margins[i], m.Vocab.Text(out[i]), rest))
							covered = 0
							break
						}
						t.Logf("%q: diverged at token %d on a %.3f tie (%q vs llama.cpp's %q...); not a bug",
							prompt, i, margins[i], m.Vocab.Text(out[i]), rest)
						break
					}
					t.Errorf("token %d: jitllm %q (margin %.3f), llama.cpp %q...\n jitllm:       %q\n llama.cpp: %q",
						i, m.Vocab.Text(out[i]), margins[i], rest,
						prompt+m.Vocab.Decode(out), text)
					covered = -1
					break
				}
				if covered < 0 {
					return // a confident mismatch; another prompt cannot unsay it
				}
				if covered > 0 {
					// The verdict is the teacher-forced arm and a chain that parts
					// only at a tie, both above. Where the chain parts is not: a
					// tie goes whichever way the host's reduction order sends it,
					// so the same binary can part at token 3 on one host and walk
					// all 20 on another, with every position teacher-forced in
					// agreement on both. The length is reported beside it as a
					// lead, never as a floor.
					if rec, ok := llamaCppCoverage[name]; ok && covered != rec {
						t.Logf("agreed for %d tokens; the laptop's record is %d (a different tie, not a gate)",
							covered, rec)
					}
					ran++
					return
				}
			}
			t.Errorf("every prompt compared nothing:\n  %s", strings.Join(ties, "\n  "))
		})
	}
	if ran == 0 {
		t.Fatal("GGUFs are present and not one model reached llama.cpp; this gate " +
			"proved nothing and a skip here would hide that (RULE 10)")
	}
}

// llamaCppCoverage is how many tokens each model walked with llama.cpp before
// the first near-tie, measured against llama-completion. It is a
// record, not a floor: where a greedy chain meets a tie, and which way the tie
// goes, depends on the host's reduction order (a one-ulp change in the rotary
// table moved four rows, two each way), so another host parts elsewhere with
// every position teacher-forced in agreement. The teacher-forced sweep and the
// tie rule are what guard every position.
var llamaCppCoverage = map[string]int{
	"Llama-3.2-1B-Instruct-Q4_K_M.gguf": 20,
	"Qwen2-1.5B-Instruct-Q4_K_M.gguf":   1, // one-ulp table, 0.085 tie at token 1
	"Qwen2-VL-2B-Instruct-Q4_K_M.gguf":  20,
	"Qwen3-1.7B-Q4_K_M.gguf":            5, // one-ulp table, 0.035 tie at token 5
	"Qwen3-MOE-4x0.6B-Q4_K_M.gguf":      20,
	"SmolLM2-360M-Instruct-Q8_0.gguf":   20,
	"SmolVLM-256M-Instruct-Q8_0.gguf":   20,
	"gemma-2-2b-it-Q4_K_M.gguf":         18, // arm64 untuned (TuneOff), 0.089 tie at token 18
	"gemma-2b.gguf":                     17,
	"gemma-3-1b-it-Q4_K_M.gguf":         7,
	// Held against the container streamed from safetensors, through
	// JITLLM_GREEDY_PAIR: jitllm takes Kimi-Linear only from safetensors.
	"moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00001-of-00002.gguf": 20,
	// A GPU server, host tier, against llama.cpp's CPU path: 20 of 20
	// teacher-forced as well.
	"Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf": 20,
	"synth-gptoss.gguf": 7,
	// The C6 fixtures.
	"synth-cohere2.gguf":               20,
	"synth-dbrx.gguf":                  20,
	"synth-commandr.gguf":              20,
	"synth-falcon.gguf":                20,
	"synth-falcon40.gguf":              20,
	"synth-nemotron.gguf":              20,
	"synth-phi2.gguf":                  20,
	"synth-phi2-q8.gguf":               20,
	"synth-stablelm.gguf":              20,
	"synth-stablelm-par.gguf":          20,
	"synth-starcoder.gguf":             20,
	"synth-starcoder2.gguf":            20,
	"google_gemma-3-1b-it-Q4_K_M.gguf": 20,
	// Held through JITLLM_GREEDY_PAIR (it is over the 2 GiB default): the
	// first K-quant falcon, whose attn_qkv is Q5_1. llama.cpp ends at EOS
	// after 11 tokens, all 11 teacher-forced positions agreeing.
	"falcon-7b-instruct.Q4_K_M.gguf": 10,
	"tiny-qwen3moe-f32.gguf":         19,
	"tinyllama-1.1b-q3_K_M.gguf":     7,
	// The flagships, on a GPU server's host tier against llama.cpp's
	// CPU path (JITLLM_GREEDY_MODEL): every teacher-forced position agrees.
	// The 26B compared 12 teacher-forced positions, all agreeing.
	"gemma-4-31B-it-Q4_K_M.gguf":            20,
	"google_gemma-4-26B-A4B-it-Q4_K_M.gguf": 15,
	"google_gemma-4-E2B-it-Q4_K_M.gguf":     5,
	"google_gemma-4-E4B-it-Q4_K_M.gguf":     14,
	"MiniMax-M2-Q3_K_M-00001-of-00003.gguf": 20,
	// The dense llama-family fixtures, and the real models of the group.
	"synth-smollm3.gguf":                       20,
	"synth-arcee.gguf":                         20,
	"synth-seedoss.gguf":                       20,
	"synth-olmo2.gguf":                         20,
	"synth-olmo2-gqa.gguf":                     20,
	"synth-olmo3.gguf":                         20,
	"synth-mistral3.gguf":                      20,
	"OLMo-2-0425-1B-Instruct-Q4_K_M.gguf":      20,
	"SmolLM3-3B-Q4_K_M.gguf":                   20,
	"Ministral-3-3B-Instruct-2512-Q4_K_M.gguf": 20,
	"AFM-4.5B-Q4_K_M.gguf":                     20,
	// llama.cpp ends at EOS after 8 tokens, all 8 teacher-forced positions
	// agreeing.
	"EXAONE-4.0-1.2B-Q4_K_M.gguf": 7,
	// Against llama.cpp's MiniMax Sparse Attention, i.e. with flash attention
	// on; its dense fallback agrees at 8 of 20 teacher-forced positions.
	"synth-minimaxm3.gguf": 20,
}

// sessionTokens reads the ids llama-completion's --prompt-cache-all saved: a
// u32 magic ('ggsn'), a u32 version, a u32 count and then the ids. The state
// that follows them is not read.
func sessionTokens(path string) ([]int32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 12 || binary.LittleEndian.Uint32(b) != 0x6767736e {
		return nil, errors.New("not a llama.cpp session file")
	}
	n := int(binary.LittleEndian.Uint32(b[8:]))
	if n > (len(b)-12)/4 {
		return nil, fmt.Errorf("session file holds %d bytes and claims %d tokens", len(b), n)
	}
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(b[12+4*i:]))
	}
	return ids, nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestSidesWithTheSessionRunIsNarrow holds the excuse above to a chain that is
// llama.cpp's session run token for token: a token of neither run, or a chain
// that left the session run earlier, is still a parting the gate judges.
func TestSidesWithTheSessionRunIsNarrow(t *testing.T) {
	want := []int32{5, 6, 7, 8}
	for _, c := range []struct {
		out  []int32
		i    int
		want bool
	}{
		{[]int32{5, 6, 7}, 2, true},        // the session run's token where the plain run parts
		{[]int32{5, 6, 9}, 2, false},       // a token of neither run
		{[]int32{5, 9, 7}, 2, false},       // left the session run before
		{[]int32{5, 6, 7, 8, 1}, 4, false}, // past the session run's ids
	} {
		if got := sidesWithTheSessionRun(c.out, want, c.i); got != c.want {
			t.Errorf("out %v at %d: %v, want %v", c.out, c.i, got, c.want)
		}
	}
}
