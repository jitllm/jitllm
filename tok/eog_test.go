package tok_test

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/tok"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStopSetIsTheModelsOwn holds the stop set built from a GGUF to the
// model's own: its generation_config.json's eos_token_id, copied below from
// the published file, which a GGUF carries as its eos and eot ids and its
// family's library statement's stop pieces (convert's mjStates.Stop). No
// piece is a stop by its name alone: SmolLM2's <|endoftext|>, Qwen's FIM
// pieces and </s>, and gpt-oss's <|end|> end no generation, since their
// references do not stop on them. stories260K-infill has no reference
// generation_config, and stops on the eos its file states alone, its FIM
// ids no stops. rnj-1's file states <|end_of_text|> as its eos beside the
// <|eot_id|> its generation_config lists.
func TestStopSetIsTheModelsOwn(t *testing.T) {
	for _, c := range []struct {
		name string
		want []int32
	}{
		{"hunyuanvl/HunyuanOCR-Q8_0", []int32{120007, 120020}},
		{"qwen2.5-1.5b-instruct-q4_k_m", []int32{151643, 151645}},
		{"stories260K-infill", []int32{2}},
		{"gpt-oss-20b-Q4_K_M", []int32{199999, 200002, 200012}},
		{"Phi-3.5-mini-instruct-Q4_K_M", []int32{32000, 32001, 32007}},
		{"Llama-3.2-1B-Instruct-Q4_K_M", []int32{128001, 128008, 128009}},
		{"Meta-Llama-3.1-8B-Instruct-Q4_K_M", []int32{128001, 128008, 128009}},
		{"gemma-2-2b-it-Q4_K_M", []int32{1, 107}},
		{"gemma-3-1b-it-Q4_K_M", []int32{1, 106}},
		{"gemma4/gemma-4-31B-it-Q4_K_M", []int32{1, 50, 106}},
		{"SmolLM2-360M-Instruct-Q8_0", []int32{2}},
		{"DeepSeek-V2-Lite.Q4_K_M", []int32{100001}},
		{"Mistral-7B-Instruct-v0.3-Q4_K_M", []int32{2}},
		{"Qwen3-0.6B-Q8_0", []int32{151643, 151645}},
		{"phi-4-Q4_K_M", []int32{100257, 100265}},
		{"falconh1/Falcon-H1-0.5B-Instruct-Q4_K_M", []int32{11, 228}},
		{"lfm2/LFM2-350M-Q4_K_M", []int32{7}},
		{"granite/granite-3.3-2b-instruct-Q4_K_M", []int32{0}},
		{"newarch/EXAONE-4.0-1.2B-Q4_K_M", []int32{361}},
		{"realck/Phi-mini-MoE-instruct-Q4_K_M", []int32{32000, 32001, 32007}},
		{"realck/Phi-tiny-MoE-instruct-Q8_0", []int32{32000, 32001, 32007}},
		{"GLM-4-9B-0414-Q4_K_M", []int32{151329, 151336, 151338}},
		{"library/Llama-4-Scout-17B-16E-Instruct-Q3_K_S", []int32{200001, 200007, 200008}},
		{"moe2/inclusionAI_Ling-mini-2.0-Q4_K_M", []int32{156892, 156895}},
		{"nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M", []int32{2, 11, 12}},
		{"nemotron/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M", []int32{2, 11}},
		{"swiss-ai_Apertus-8B-Instruct-2509-Q4_K_M", []int32{2, 68, 72}},
		{"minicpmv/MiniCPM-V-4-Q4_K_M", []int32{2, 73440}},
		{"rnj/EssentialAI_rnj-1-instruct-Q4_K_M", []int32{128001, 128009}},
	} {
		t.Run(filepath.Base(c.name), func(t *testing.T) {
			path := testmodels.Path(c.name + ".gguf")
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
			}
			s := openSrc(t, testmodels.Path(c.name))
			if s == nil {
				testmodels.Missing(t, "MODEL MISSING: %s", c.name)
			}
			v, err := tok.New(s.vocab(t))
			s.Close()
			if err != nil {
				t.Fatal(err)
			}
			got := slices.Clone(v.EOG())
			slices.Sort(got)
			if !slices.Equal(got, c.want) {
				t.Errorf("stop set %v, the model's generation_config states %v", got, c.want)
			} else {
				t.Logf("%d stop ids agree: %v", len(got), got)
			}
		})
	}
}

// TestStatedStopsEndAGeneration: an id the container states ends a
// generation, and one it does not state does not. HunyuanOCR closes a turn with
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
		t.Error("with the stated stops dropped <｜hy_Assistant｜> still ends a generation: something besides the container's set decides it")
	}
}

// TestAnUnstatedContainerStopsByName: a container written before it stated
// its stop set (WithNamedStops, which model.Open passes for a container with
// no generation tail) stops where it did -- Phi-3.5's <|end|> and </s> --
// and harmony's <|end|> still does not end gpt-oss's reply; the same
// vocabulary without the option stops on EOS alone.
func TestAnUnstatedContainerStopsByName(t *testing.T) {
	for _, c := range []struct {
		name     string
		eog, not []string
		unstated []string
	}{
		{"Phi-3.5-mini-instruct-Q4_K_M", []string{"<|end|>", "</s>"}, nil, []string{"<|end|>"}},
		{"gpt-oss-20b-Q4_K_M", []string{"<|return|>", "<|call|>"}, []string{"<|end|>"}, []string{"<|call|>"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := openSrc(t, testmodels.Path(c.name))
			if s == nil {
				testmodels.Missing(t, "MODEL MISSING: %s", c.name)
			}
			vc := *s.vocab(t)
			s.Close()
			vc.Stop = nil
			named, err := tok.New(&vc, tok.WithNamedStops())
			if err != nil {
				t.Fatal(err)
			}
			bare, err := tok.New(&vc)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range append(append(slices.Clone(c.eog), c.not...), c.unstated...) {
				id, ok := named.ID(p)
				if !ok {
					t.Fatalf("%s is not in the vocabulary", p)
				}
				switch {
				case slices.Contains(c.eog, p) && !named.IsEOG(id):
					t.Errorf("%s does not end a generation by name", p)
				case slices.Contains(c.not, p) && named.IsEOG(id):
					t.Errorf("%s ends a generation by name", p)
				case slices.Contains(c.unstated, p) && bare.IsEOG(id):
					t.Errorf("%s ends a generation with no stops stated and no names", p)
				}
			}
		})
	}
}

// TestHarmonyContinuesPastEnd holds gpt-oss's stop set to its harmony format
// and phi3's to its own, on the real vocabularies, each its library
// statement's. In harmony <|end|> closes a message -- the reasoning is one --
// and the turn ends at <|return|> or <|call|>; stopping on <|end|> cut every
// reply off before its answer. phi3's <|end|> is its end of turn, and must
// stay one.
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
