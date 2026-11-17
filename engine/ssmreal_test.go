package engine_test

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// ssmRealModels are the real state-space hybrids held to the GGUF oracle, each
// relative to JITLLM_MODELS; JITLLM_SSM_REAL replaces the list
// (comma-separated). A missing file is skipped by name.
//
// Jamba is held at F16, where transformers on the file's own weights, jitllm
// and both of the oracle's arms agree at every position. At Q4_K_M the story
// prompt parts from every reference, because the 8-bit activations both engines
// quantize are amplified by 26 recurrences; no engine's greedy answer there is
// an oracle for another's (docs/engineering-history/model-correctness.md,
// Mamba-1).
func ssmRealModels() []string {
	return []string{
		"granite/granite-4.0-h-350m-Q8_0.gguf",
		"granite/granite-4.0-h-1b-Q4_K_M.gguf",
		"granite/granite-4.0-h-tiny-Q4_K_M.gguf",
		"nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
		"falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M.gguf",
		"falconh1/Falcon-H1-1.5B-Instruct-Q8_0.gguf",
		"lfm2/LFM2-350M-Q4_K_M.gguf",
		"lfm2/LFM2-1.2B-Q4_K_M.gguf",
		"lfm2/LFM2-8B-A1B-Q4_K_M.gguf",
		"jamba/jamba-reasoning-3b-F16.gguf",
		"mamba/mamba-130m-hf.Q8_0.gguf",
		"falconmamba/falcon-mamba-7B-Q4_K_M.gguf",
		"mamba2/mamba-codestral-7b-v0.1-q4_k_m.gguf",
	}
}

// TestSSMRealModelsTeacherForced feeds jitllm the GGUF oracle's own greedy
// tokens on real Mamba-2 hybrids and compares the argmax at every position:
// three prompts of 32 tokens each, past EOS (--ignore-eos), so an instruct
// model's early stop does not leave nothing to compare. --ignore-eos biases
// every end-of-generation token to -inf, so jitllm's argmax skips them too.
// Both engines see an identical input at every position, so a CONFIDENT
// disagreement (margin over tieMargin) is a wrong answer; a tie is neither
// engine's fault. JITLLM_SSM_NGL sets the GGUF oracle's offload (its CPU arm is
// 0).
func TestSSMRealModelsTeacherForced(t *testing.T) {
	lcpp := os.Getenv("JITLLM_LCPP")
	if lcpp == "" {
		t.Skip("set JITLLM_LCPP to a GGUF oracle build directory holding llama-completion")
	}
	bin := lcppBin(lcpp, "llama-completion")
	if _, err := os.Stat(bin); err != nil {
		t.Skipf("llama-completion is not in JITLLM_LCPP (%s)", lcpp)
	}
	names := ssmRealModels()
	if v := os.Getenv("JITLLM_SSM_REAL"); v != "" {
		names = strings.Split(v, ",")
	}
	prompts := []string{"The capital of France is", "Once upon a time",
		"def fibonacci(n):\n    "}
	const n = 32
	ran := 0
	for _, name := range names {
		path := testmodels.Path(name)
		if _, err := os.Stat(path); err != nil {
			t.Logf("%s: not here", name)
			continue
		}
		t.Run(filepath.Base(name), func(t *testing.T) {
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			total, agree := 0, 0
			for _, prompt := range prompts {
				ids := m.Vocab.Encode(prompt, true)
				sess := filepath.Join(t.TempDir(), "session.bin")
				// The GGUF oracle's CPU arm is the oracle every SSM is held to:
				// its card arm's reduction order moves with the cards it sees
				// (granite-4.0-h-1b and LFM2-8B-A1B read another text on one
				// card than on two), and its CUDA build offloads a long batch
				// to a visible card even at -ngl 0, so no card is visible to it
				// unless JITLLM_SSM_NGL asks for another split.
				ngl, hide := "0", true
				if v := os.Getenv("JITLLM_SSM_NGL"); v != "" {
					ngl, hide = v, false
				}
				oracleEnv := func() []string {
					env := append(os.Environ(), "LD_LIBRARY_PATH="+filepath.Dir(bin)+string(os.PathListSeparator)+os.Getenv("LD_LIBRARY_PATH"))
					if hide {
						env = append(env, "CUDA_VISIBLE_DEVICES=")
					}
					return env
				}
				run := func(ngl string) ([]byte, error) {
					cmd := exec.Command(bin, "-m", path, "-p", prompt, "-n", itoa(n), "--temp", "0", "-no-cnv",
						"--no-warmup", "--ignore-eos", "-ngl", ngl, "-c", "512", "--prompt-cache", sess, "--prompt-cache-all")
					cmd.Env = oracleEnv()
					return cmd.CombinedOutput()
				}
				out, err := run(ngl)
				if err != nil {
					t.Fatalf("llama-completion: %v\n%s", err, out)
				}
				lids, err := sessionTokens(sess)
				if err != nil {
					t.Fatal(err)
				}
				if len(lids) < len(ids) || !slices.Equal(lids[:len(ids)], ids) {
					t.Fatalf("%q: the GGUF oracle tokenized the prompt as %v, jitllm as %v", prompt,
						lids[:min(len(ids), len(lids))], ids)
				}
				want := lids[len(ids):]
				// The ids come from a run that saves a session, and saving one
				// changes the GGUF oracle's reduction order: on the M4
				// LFM2-8B's session run said " Seine" at token 30 where its
				// plain run, on its CPU or not, says " Louvre" -- the GGUF
				// oracle disagreeing with itself, at a position jitllm
				// (" Louvre", margin 1.38) was failed on. The plain run's text
				// is the GGUF oracle's answer, so the ids are compared only up
				// to where the session run leaves it.
				plain := exec.Command(bin, "-m", path, "-p", prompt, "-n", itoa(n), "--temp", "0", "-no-cnv",
					"--no-warmup", "--ignore-eos", "-ngl", ngl, "-c", "512")
				plain.Env = oracleEnv()
				if pout, err := plain.Output(); err == nil {
					text := strings.TrimSpace(strings.ReplaceAll(string(pout), "[end of text]", ""))
					for i := range want {
						if s := prompt + m.Vocab.Decode(want[:i+1]); !strings.HasPrefix(text, s) && !strings.HasPrefix(s, text) {
							t.Logf("%q: the GGUF oracle's session run leaves its plain run at token %d; compared up to there", prompt, i)
							want = want[:i]
							break
						}
					}
				}
				st := testInstance(t, m, len(ids)+len(want)+1)
				lg, err := st.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				got := 0
				for i, w := range want {
					bi, best, second := int32(0), float32(math.Inf(-1)), float32(math.Inf(-1))
					for j, v := range lg {
						if m.Vocab.IsEOG(int32(j)) {
							continue
						}
						if v > best {
							best, second, bi = v, best, int32(j)
						} else if v > second {
							second = v
						}
					}
					switch {
					case bi == w:
						got++
					case best-second > tieMargin:
						t.Errorf("%q position %d: jitllm %q (margin %.3f), the GGUF oracle %q", prompt, i,
							m.Vocab.Text(bi), best-second, m.Vocab.Text(w))
					default:
						t.Logf("%q position %d: a %.3f tie (%q against the GGUF oracle's %q)", prompt, i,
							best-second, m.Vocab.Text(bi), m.Vocab.Text(w))
					}
					if lg, err = st.Forward(w); err != nil {
						t.Fatal(err)
					}
				}
				st.Close()
				t.Logf("%q: %d of %d positions agree: %q", prompt, got, len(want), m.Vocab.Decode(want))
				total, agree = total+len(want), agree+got
			}
			t.Logf("teacher-forced: %d of %d positions agree with the GGUF oracle", agree, total)
		})
		ran++
	}
	if ran == 0 {
		t.Skip("no real state-space hybrid here (JITLLM_SSM_REAL)")
	}
}
