package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGenerationIsTheModelsOwn holds what a conversion states a model
// generates with to the model's own generation_config.json, the real files:
// the draw (do_sample, temperature, top-k, top-p, and GenerationConfig's
// default for each one the file leaves out) and the stop set (its
// eos_token_id, one id or a list, with the tokenizer's eos). The values below
// are copied from the published files; SmolLM2's, HunyuanOCR's, Pythia's and
// Phi-tiny-MoE's state no sampling field, so they are greedy.
func TestGenerationIsTheModelsOwn(t *testing.T) {
	ref := jlm.ReferenceGeneration()
	sampled := func(temp float64, topK int, topP float64) jlm.Generation {
		g := ref
		g.DoSample, g.Temperature, g.TopP = true, temp, topP
		if topK != 0 {
			g.TopK = topK
		}
		return g
	}
	for _, c := range []struct {
		dir  string
		gen  jlm.Generation
		stop []int32
	}{
		{"SmolLM2-360M-Instruct", ref, []int32{2}},
		{"qwen3", sampled(0.6, 20, 0.95), []int32{151643, 151645}},
		{"qwen3-moe", sampled(0.6, 20, 0.95), []int32{151643, 151645}},
		{"qwen3-next-moe", sampled(0.7, 20, 0.8), []int32{151643, 151645}},
		{"bitnet-b1.58-2B-4T", sampled(0.6, 0, 0.9), []int32{128001, 128009}},
		{"DeepSeek-V2-Lite", sampled(0.3, 0, 0.95), []int32{100001}},
		{"hunyuanvl/hf/HunyuanOCR", ref, []int32{120007, 120020}},
		{"realck/Phi-tiny-MoE-instruct-hf", ref, []int32{32000, 32001, 32007}},
		{"pythia-160m", ref, []int32{0}},
	} {
		t.Run(filepath.Base(c.dir), func(t *testing.T) {
			p := testmodels.Path(c.dir)
			if _, err := os.Stat(filepath.Join(p, "generation_config.json")); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
			}
			dir := dirFiles(p)
			g, err := hfGeneration(dir)
			if err != nil {
				t.Fatal(err)
			}
			if *g != c.gen {
				t.Errorf("generation %+v, the file states %+v", *g, c.gen)
			}
			b, err := os.ReadFile(filepath.Join(p, "config.json"))
			if err != nil {
				t.Fatal(err)
			}
			var cfg struct {
				Vocab uint32 `json:"vocab_size"`
				Text  struct {
					Vocab uint32 `json:"vocab_size"`
				} `json:"text_config"`
			}
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Fatal(err)
			}
			v, err := hfVocabOf(dir, max(cfg.Vocab, cfg.Text.Vocab))
			if err != nil {
				t.Fatal(err)
			}
			if v.Stop, err = hfStops(dir, v); err != nil {
				t.Fatal(err)
			}
			got := append([]int32{v.EOS}, v.Stop...)
			slices.Sort(got)
			got = slices.Compact(got)
			if !slices.Equal(got, c.stop) {
				t.Errorf("stop set %v (eos %d, stated %v), the file states %v", got, v.EOS, v.Stop, c.stop)
			}
		})
	}
}

// TestGGUFGenerationReadsItsSamplingKeys: a GGUF's general.sampling keys are
// the draw its container states, and one stating none is greedy. HunyuanOCR's
// file states temperature 0, top-k 1 and top-p 1, the greedy draw;
// Gemma 4's states temperature 1, top-k 64 and top-p 0.95, which reach the
// container as the decimals written, not their f32 neighbours; Llama 3.2's
// states none. synth-sampling's states every key the converter writes, min-p
// and the repetition penalty among them, as scripts/samplinggold.py's
// generation_config.json states them.
func TestGGUFGenerationReadsItsSamplingKeys(t *testing.T) {
	ref := jlm.ReferenceGeneration()
	gemma := ref
	gemma.DoSample, gemma.Temperature, gemma.TopK, gemma.TopP = true, 1, 64, 0.95
	hy := ref
	hy.Temperature, hy.TopK = 0, 1
	every := ref
	every.DoSample, every.Temperature, every.TopK, every.TopP = true, 0.6, 20, 0.95
	every.MinP, every.RepetitionPenalty = 0.05, 1.1
	for _, c := range []struct {
		name string
		want jlm.Generation
	}{
		{"hunyuanvl/HunyuanOCR-Q8_0", hy},
		{"gemma4/gemma-4-31B-it-Q4_K_M", gemma},
		{"Llama-3.2-1B-Instruct-Q4_K_M", ref},
		{"synth-sampling", every},
	} {
		t.Run(filepath.Base(c.name), func(t *testing.T) {
			p := testmodels.Path(c.name + ".gguf")
			if _, err := os.Stat(p); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
			}
			f, err := gguf.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			g, err := ggufGeneration(f, nil)
			if err != nil {
				t.Fatal(err)
			}
			if *g != c.want {
				t.Errorf("generation %+v, want %+v", *g, c.want)
			}
		})
	}
}

// TestDoSampleIsGenerationConfigs: a generation_config.json that states a
// temperature and top-p but no do_sample (Seed-OSS's) draws greedily, as
// GenerationConfig's default does, keeping the values it states; one stating
// do_sample true samples with them. Seed-OSS's GGUF, whose file has the keys
// and no do_sample, draws as its family's statement says (seed_oss: greedy),
// so the two conversions agree.
func TestDoSampleIsGenerationConfigs(t *testing.T) {
	want := jlm.ReferenceGeneration()
	want.Temperature, want.TopP = 1.1, 0.95
	for _, c := range []struct {
		file string
		do   bool
	}{
		{`{"temperature": 1.1, "top_p": 0.95}`, false},
		{`{"do_sample": true, "temperature": 1.1, "top_p": 0.95}`, true},
	} {
		d := t.TempDir()
		if err := os.WriteFile(filepath.Join(d, "generation_config.json"), []byte(c.file), 0o644); err != nil {
			t.Fatal(err)
		}
		g, err := hfGeneration(dirFiles(d))
		if err != nil {
			t.Fatal(err)
		}
		w := want
		w.DoSample = c.do
		if *g != w {
			t.Errorf("%s: generation %+v, want %+v", c.file, *g, w)
		}
	}
	p := testmodels.Path("synth-seedoss.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
	}
	f, err := gguf.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	g, err := ggufGeneration(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	if *g != want {
		t.Errorf("synth-seedoss.gguf: generation %+v, want %+v", *g, want)
	}
}

// TestThePublishersGenerationConfigIsTheDraw: a GGUF converted with its
// publisher's directory (WithModelConfig, the directory or its config.json)
// states the draw that directory's generation_config.json states, over what
// its keys gave; with no generation_config.json there the keys' draw stays.
func TestThePublishersGenerationConfigIsTheDraw(t *testing.T) {
	keys := jlm.ReferenceGeneration()
	keys.Temperature, keys.TopK = 0, 1
	want := jlm.ReferenceGeneration()
	want.DoSample, want.Temperature, want.TopP = true, 0.6, 0.9
	d := t.TempDir()
	if err := os.WriteFile(filepath.Join(d, "config.json"), []byte(`{"vocab_size": 8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(d, "generation_config.json"),
		[]byte(`{"do_sample": true, "temperature": 0.6, "top_p": 0.9}`), 0o644); err != nil {
		t.Fatal(err)
	}
	bare := t.TempDir()
	if err := os.WriteFile(filepath.Join(bare, "config.json"), []byte(`{"vocab_size": 8}`), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		config string
		want   jlm.Generation
	}{
		{d, want},
		{filepath.Join(d, "config.json"), want},
		{bare, keys},
		{"", keys},
	} {
		g := keys
		s := &jlm.Source{Config: &jlm.Config{Generation: &g}}
		o := &options{modelConfig: c.config}
		if err := o.applyGenerationConfig(s); err != nil {
			t.Fatal(err)
		}
		if *s.Config.Generation != c.want {
			t.Errorf("%q: generation %+v, want %+v", c.config, *s.Config.Generation, c.want)
		}
	}
}
