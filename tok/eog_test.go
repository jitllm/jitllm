package tok_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/tok"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStopSetMatchesLlamaCpp holds the stop set built from a GGUF to the one
// llama.cpp prints at load (llama-vocab.cpp, special_eog_ids): EOS, the
// file's eot, eom and FIM ids, and the names it matches, with its harmony and
// gemma4 exceptions. The models are picked for what each exercises: a stated
// eot no name covers (HunyuanOCR), a literal "</s>" and FIM tokens in a BPE
// vocabulary (Qwen2.5, stories260K-infill), harmony (gpt-oss), <|end|>
// (phi3), and the families' own turn ends.
func TestStopSetMatchesLlamaCpp(t *testing.T) {
	lcpp := os.Getenv("JITLLM_LCPP")
	if lcpp == "" {
		t.Skip("set JITLLM_LCPP to a llama.cpp build directory holding llama-tokenize -- this gate proved nothing")
	}
	bin := filepath.Join(lcpp, "llama-tokenize")
	eogLine := regexp.MustCompile(`EOG token\s+=\s+(\d+)`)
	for _, name := range []string{
		"hunyuanvl/HunyuanOCR-Q8_0", "qwen2.5-1.5b-instruct-q4_k_m", "stories260K-infill",
		"gpt-oss-20b-Q4_K_M", "Phi-3.5-mini-instruct-Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M",
		"gemma-3-1b-it-Q4_K_M", "SmolLM2-360M-Instruct-Q8_0", "DeepSeek-V2-Lite.Q4_K_M",
		"Mistral-7B-Instruct-v0.3-Q4_K_M", "Qwen3-0.6B-Q8_0", "phi-4-Q4_K_M",
	} {
		t.Run(filepath.Base(name), func(t *testing.T) {
			path := testmodels.Path(name + ".gguf")
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
			}
			cmd := exec.Command(bin, "-m", path, "-p", "x", "--ids", "--log-verbose")
			cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+lcpp)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("llama-tokenize: %v\n%s", err, out)
			}
			var want []int32
			for _, m := range eogLine.FindAllStringSubmatch(string(out), -1) {
				n, _ := strconv.Atoi(m[1])
				if !slices.Contains(want, int32(n)) {
					want = append(want, int32(n))
				}
			}
			if len(want) == 0 {
				t.Fatalf("llama-tokenize printed no EOG token -- the oracle said nothing:\n%s", out)
			}
			s := openSrc(t, testmodels.Path(name))
			if s == nil {
				testmodels.Missing(t, "MODEL MISSING: %s", name)
			}
			v, err := tok.New(s.vocab(t))
			s.Close()
			if err != nil {
				t.Fatal(err)
			}
			got := slices.Clone(v.EOG())
			slices.Sort(got)
			slices.Sort(want)
			if !slices.Equal(got, want) {
				t.Errorf("stop set %v, llama.cpp %v", got, want)
			} else {
				t.Logf("%d stop ids agree: %v", len(got), got)
			}
		})
	}
}

// TestStatedStopsEndAGeneration: an id the container states ends a generation
// though no name llama.cpp matches covers it. HunyuanOCR closes a turn with
// <｜hy_Assistant｜>, which reaches the container only as the GGUF's
// eot_token_id; with the stated set dropped -- the state before the
// container carried it -- the reply runs on past its turn.
func TestStatedStopsEndAGeneration(t *testing.T) {
	name := testmodels.Path("hunyuanvl/HunyuanOCR-Q8_0")
	s := openSrc(t, name)
	if s == nil {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+name+" (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)")
	}
	vc := s.vocab(t)
	s.Close()
	v, err := tok.New(vc)
	if err != nil {
		t.Fatal(err)
	}
	id, ok := v.ID("<｜hy_Assistant｜>")
	if !ok || id != 120007 {
		t.Fatalf("<｜hy_Assistant｜> is %d (%v), want 120007", id, ok)
	}
	if !v.IsEOG(id) || !v.IsEOG(v.EOS) {
		t.Errorf("IsEOG(<｜hy_Assistant｜>) = %v, IsEOG(eos %d) = %v; want both", v.IsEOG(id), v.EOS, v.IsEOG(v.EOS))
	}
	dropped := *vc
	dropped.Stop = nil
	if w, err := tok.New(&dropped); err != nil {
		t.Fatal(err)
	} else if w.IsEOG(id) {
		t.Error("with the stated stops dropped <｜hy_Assistant｜> still ends a generation: a name covers it, and this gate cannot see the container's set")
	}
}

// TestHarmonyContinuesPastEnd holds gpt-oss's stop set to its harmony format
// and phi3's to its own, on the real vocabularies. In harmony <|end|> closes a
// message -- the reasoning is one -- and the turn ends at <|return|> or
// <|call|>; stopping on <|end|> cut every reply off before its answer. phi3's
// <|end|> is its end of turn, and must stay one.
func TestHarmonyContinuesPastEnd(t *testing.T) {
	for _, c := range []struct {
		stems       []string
		eog, notEOG []string
	}{
		{[]string{testmodels.Path("gpt-oss-20b-Q4_K_M")}, []string{"<|return|>", "<|call|>"}, []string{"<|end|>"}},
		{[]string{testmodels.Path("Phi-3.5-mini-instruct-Q4_K_M")}, []string{"<|end|>"}, nil},
	} {
		s := openSrc(t, c.stems...)
		if s == nil {
			testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", c.stems)
		}
		v, err := tok.New(s.vocab(t))
		s.Close()
		if err != nil {
			t.Fatal(err)
		}
		check := func(name string, want bool) {
			id, ok := v.ID(name)
			if !ok {
				t.Fatalf("%s: vocabulary has no %s", c.stems[0], name)
			}
			if v.IsEOG(id) != want {
				t.Errorf("%s: IsEOG(%s) = %v, want %v", c.stems[0], name, !want, want)
			}
		}
		for _, n := range c.eog {
			check(n, true)
		}
		for _, n := range c.notEOG {
			check(n, false)
		}
	}
}

// TestDecodeChatRendersHarmonyChannels: gpt-oss's reasoning must reach a
// reader as a <think> block and its answer as text, with the role and channel
// headers gone -- Decode drops the special tokens and leaves the header words
// glued on ("analysisThe user asks..."). A turn cut off mid-reasoning, which is
// what a stream shows, is an open block. Any other vocabulary is Decode.
func TestDecodeChatRendersHarmonyChannels(t *testing.T) {
	s := openSrc(t, testmodels.Path("gpt-oss-20b-Q4_K_M"))
	if s == nil {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("gpt-oss-20b-Q4_K_M")+" (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)")
	}
	v, err := tok.New(s.vocab(t))
	s.Close()
	if err != nil {
		t.Fatal(err)
	}
	sp := func(name string) int32 {
		id, ok := v.ID(name)
		if !ok {
			t.Fatalf("no %s", name)
		}
		return id
	}
	var ids []int32
	add := func(x ...any) {
		for _, e := range x {
			switch e := e.(type) {
			case string:
				ids = append(ids, v.Encode(e, false)...)
			case int32:
				ids = append(ids, e)
			}
		}
	}
	add(sp("<|channel|>"), "analysis", sp("<|message|>"), "The user asks for 17*23.")
	if got, want := v.DecodeChat(ids), "<think>The user asks for 17*23."; got != want {
		t.Errorf("mid-reasoning: %q, want %q", got, want)
	}
	add(sp("<|end|>"), sp("<|start|>"), "assistant", sp("<|channel|>"), "final", sp("<|message|>"),
		"It is 391.", sp("<|return|>"))
	if got, want := v.DecodeChat(ids), "<think>The user asks for 17*23.</think>It is 391."; got != want {
		t.Errorf("full turn: %q, want %q", got, want)
	}

	p := openSrc(t, testmodels.Path("Phi-3.5-mini-instruct-Q4_K_M"))
	if p == nil {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("Phi-3.5-mini-instruct-Q4_K_M")+" (set JITLLM_MODELS to the model directory) -- the non-harmony arm would prove nothing")
	}
	pv, err := tok.New(p.vocab(t))
	p.Close()
	if err != nil {
		t.Fatal(err)
	}
	x := pv.Encode("The capital of France is Paris.", false)
	if pv.DecodeChat(x) != pv.Decode(x) {
		t.Errorf("phi3: DecodeChat %q differs from Decode %q", pv.DecodeChat(x), pv.Decode(x))
	}
}
