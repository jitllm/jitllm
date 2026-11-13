package tok_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok"
)

// The oracle is llama.cpp's own tokenizer (llama-tokenize --ids), not a round
// trip: a tokenizer that emits one token per byte round-trips perfectly and
// ruins the model.
func llamaTokenize(t *testing.T, model, text string, parseSpecial bool) ([]int32, bool) {
	t.Helper()
	// Without JITLLM_LCPP (a llama.cpp build directory holding llama-tokenize)
	// there is no oracle and the caller skips.
	lcpp := os.Getenv("JITLLM_LCPP")
	if lcpp == "" {
		return nil, false
	}
	bin := filepath.Join(lcpp, "llama-tokenize")
	if _, err := os.Stat(bin); err != nil {
		return nil, false
	}
	args := []string{"-m", model, "-p", text, "--ids"}
	if !parseSpecial {
		args = append(args, "--no-parse-special")
	}
	cmd := exec.Command(bin, args...)
	cmd.Env = append(os.Environ(),
		"LD_LIBRARY_PATH="+filepath.Dir(bin))
	out, err := cmd.Output()
	if err != nil {
		// It exits non-zero on an empty prompt, which is a case worth keeping.
		return nil, false
	}
	line := ""
	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(strings.TrimSpace(l), "[") {
			line = strings.TrimSpace(l)
		}
	}
	if line == "" {
		return nil, false
	}
	var ids []int32
	if err := json.Unmarshal([]byte(line), &ids); err != nil {
		t.Logf("parsing %q: %v", line, err)
		return nil, false
	}
	return ids, true
}

// The cases are chosen for the parts of the pre-tokenizer that are easy to get
// subtly wrong: contractions, digit grouping in threes, runs of spaces (where
// the last one belongs to the next word), newlines, CJK, emoji and punctuation
// runs.
func bpeCases() []string {
	return []string{
		"The capital of France is Paris.",
		"hello world",
		"  leading and   internal   spaces  ",
		"Isn't it? I'd say we've all got 'em, y'all.",
		"1 12 123 1234 12345 007 3.14159",
		"line one\nline two\n\n\nline three\r\ntrailing",
		"tabs\tand\t\tmore",
		"CamelCaseIdentifier snake_case_name kebab-case-name",
		"func main() { fmt.Println(\"hi\") } // comment",
		"日本語のテキストです。中文也可以。한국어도.",
		"emoji 🚀🔥 and combining é vs e\u0301",
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		"a  b   c    d     e",
		"$1,234.56 + €99 = ???",
		"",
		" ",
		"\n",
		"x",
	}
}

func TestBPEAgainstLlamaCpp(t *testing.T) {
	models := testmodels.Glob("*.gguf")
	if len(models) == 0 {
		testmodels.Missing(t, "%s", "MODEL MISSING: no *.gguf in "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	checked := 0
	for _, path := range models {
		f, err := gguf.Open(path)
		if err != nil {
			continue
		}
		kind, _ := f.KV["tokenizer.ggml.model"].String()
		jv := vocabOfMaybe(t, f)
		if jv == nil {
			f.Close()
			continue // a projector GGUF has no tokenizer section
		}
		v, verr := tok.New(jv)
		f.Close()
		if kind != "gpt2" {
			continue
		}
		if verr != nil {
			t.Errorf("%s: tok.New: %v", filepath.Base(path), verr)
			continue
		}
		name := filepath.Base(path)
		// llama-tokenize parses specials unless told not to: EncodeSpecial is
		// the one, Encode the other, a control token's text characters there.
		for _, text := range bpeCases() {
			for _, parse := range []bool{true, false} {
				want, ok := llamaTokenize(t, path, text, parse)
				if !ok {
					continue // llama-tokenize declined this one (it rejects "")
				}
				got := v.Encode(text, true)
				if parse {
					got = v.EncodeSpecial(text, true)
				}
				if !slices.Equal(got, want) {
					t.Errorf("%s %q (parse_special %v):\n got  %v\n want %v", name, text, parse, got, want)
					continue
				}
				checked++
			}
		}
		// Decode must invert Encode -- except that it deliberately does not
		// render control tokens, which is what a caller printing model output
		// wants and what llama.cpp's detokenizer does by default. So the round
		// trip holds for every case that contains none.
		for _, text := range bpeCases() {
			if strings.Contains(text, "<|") {
				continue
			}
			if got, want := v.Decode(v.Encode(text, false)), tok.Representable(v, text); got != want {
				t.Errorf("%s round trip %q -> %q, want %q", name, text, got, want)
			}
		}
		// Only meaningful where those tokens exist; on a vocabulary without
		// them the text is literal and round-trips, which is also correct.
		if ids := v.EncodeSpecial("<|im_start|>hi<|im_end|>", false); len(ids) == 3 {
			if got := v.Decode(ids); got != "hi" {
				t.Errorf("%s: control tokens should not render, got %q", name, got)
			}
		}
	}
	if checked == 0 {
		t.Skip("no gpt2-tokenizer model, or llama-tokenize is not installed")
	}
	t.Logf("%d cases identical to llama.cpp", checked)
}
