package convert

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"testing/fstest"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestStopSetRoundTrips: the ids a model's file states end a generation reach
// the container from either input. HunyuanVL closes a turn with
// <｜hy_Assistant｜> (120007): llama.cpp's converter writes it to a GGUF as
// tokenizer.ggml.eot_token_id beside an eos of its own, and transformers keeps
// it in generation_config.json's eos_token_id list. Dropped at conversion, a
// reply ran on past its turn. Both fixtures are converted, written, reopened
// and the stop set read back; the real HunyuanOCR files are read for the same
// set without writing their weights.
func TestStopSetRoundTrips(t *testing.T) {
	const assistant = 120007
	stops := func(v *jlm.Vocab) []int32 {
		set := append([]int32{v.EOS}, v.Stop...)
		slices.Sort(set)
		return set
	}
	need := func(t *testing.T, p string) {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", err)
		}
	}
	for _, c := range []struct {
		name, src string
		gguf      bool
		want      []int32
	}{
		// The fixture's GGUF states eos 3, which its converter took off the
		// fixture's tokenizer; the eot is what matters.
		{"gguf", "synth-hunyuanvl.gguf", true, []int32{3, assistant}},
		// HunYuanVL's class does not convert from safetensors; Apertus's
		// generation_config lists </s> and <|tools_suffix|> beside its
		// tokenizer's <|assistant_end|>, and no name llama.cpp matches covers
		// the last.
		{"safetensors", "synth-apertus-hf", false, []int32{2, 68, 72}},
	} {
		t.Run(c.name, func(t *testing.T) {
			src := testmodels.Path(c.src)
			need(t, src)
			dst := filepath.Join(t.TempDir(), "out"+jlm.Ext)
			var err error
			if c.gguf {
				_, err = FromGGUF(src, dst, jlm.Fingerprint{Host: "test"})
			} else {
				_, err = FromSafetensors(src, dst, jlm.Fingerprint{Host: "test"})
			}
			if err != nil {
				t.Fatal(err)
			}
			ct, err := jlm.Open(dst)
			if err != nil {
				t.Fatal(err)
			}
			defer ct.Close()
			v := ct.Vocab()
			if got := stops(v); !slices.Equal(got, c.want) {
				t.Errorf("stop set %v (eos %d, stated %v), want %v", got, v.EOS, v.Stop, c.want)
			}
			if slices.Contains(v.Stop, v.EOS) {
				t.Errorf("stated stops %v repeat the eos %d", v.Stop, v.EOS)
			}
		})
	}

	// The real model, both inputs: the same two ids either way, whichever of
	// them each side calls its eos.
	t.Run("HunyuanOCR", func(t *testing.T) {
		want := []int32{assistant, 120020}
		gp := testmodels.Path("hunyuanvl/HunyuanOCR-Q8_0.gguf")
		need(t, gp)
		f, err := gguf.Open(gp)
		if err != nil {
			t.Fatal(err)
		}
		gv, err := VocabOf(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if got := stops(gv); !slices.Equal(got, want) {
			t.Errorf("gguf: stop set %v, want %v", got, want)
		}
		dir := testmodels.Path("hunyuanvl/hf/HunyuanOCR")
		need(t, dir)
		hv, err := hfVocabOf(dirFiles(dir), uint32(gv.Tokens.Len()))
		if err != nil {
			t.Fatal(err)
		}
		if hv.Stop, err = hfStops(dirFiles(dir), hv); err != nil {
			t.Fatal(err)
		}
		if got := stops(hv); !slices.Equal(got, want) {
			t.Errorf("safetensors: stop set %v, want %v", got, want)
		}
	})
}

// TestStopSetRefusesAnIdOutsideTheVocabulary: a stated stop the token table
// does not hold is a broken file, refused where a person is in front of it;
// and the two other shapes transformers accepts are read.
func TestStopSetRefusesAnIdOutsideTheVocabulary(t *testing.T) {
	v := &jlm.Vocab{Tokens: jlm.NewStrings([]string{"a", "b", "c"}), EOS: 2}
	stated := func(files map[string]string) ([]int32, error) {
		m := fstest.MapFS{}
		for name, body := range files {
			m[name] = &fstest.MapFile{Data: []byte(body)}
		}
		return hfStops(hfDir{m, "test"}, v)
	}
	if _, err := stated(map[string]string{"generation_config.json": `{"eos_token_id": [1, 7]}`}); err == nil {
		t.Error("eos_token_id 7 in a 3-token vocabulary was accepted")
	}
	// One id, not a list, and the eos itself left out of the stated set.
	if got, err := stated(map[string]string{"generation_config.json": `{"eos_token_id": 2}`}); err != nil || len(got) != 0 {
		t.Errorf("a lone eos_token_id equal to the eos: stated %v, %v; want none", got, err)
	}
	// No generation_config.json: config.json's, under text_config for a
	// vision-language model.
	if got, err := stated(map[string]string{"config.json": `{"text_config": {"eos_token_id": [2, 0]}}`}); err != nil || !slices.Equal(got, []int32{0}) {
		t.Errorf("config.json's text_config: stated %v, %v; want [0]", got, err)
	}
}
