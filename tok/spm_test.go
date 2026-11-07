package tok_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/internal/testmodels"
)

type goldenCase struct {
	Text  string  `json:"text"`
	BOS   []int32 `json:"bos"`
	NoBOS []int32 `json:"nobos"`
}

func models() map[string]string {
	return map[string]string{
		"stories260K": "../testdata/models/stories260K.gguf",
		"stories15M":  testmodels.Path("stories15M-q4_0.gguf"),
		"tinyllama":   testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"),
	}
}

// TestGoldens is the tokenizer gate: exact token ids, against llama.cpp's own
// tokenizer, on every model. There is no "close enough" here — one wrong id
// shifts every embedding lookup after it and the model still produces fluent
// text, so a tolerance would defeat the entire test.
func TestGoldens(t *testing.T) {
	for name, path := range models() {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "tokens", name+".json"))
			if err != nil {
				t.Skipf("no goldens (run scripts/tokgold.py): %v", err)
			}
			var cases []goldenCase
			if err := json.Unmarshal(raw, &cases); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(path); err != nil {
				testmodels.Missing(t, "model not present: %s (a model-directory file follows JITLLM_MODELS)", path)
			}
			f, err := gguf.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			v, err := tok.New(vocabOf(t, f))
			if err != nil {
				t.Fatal(err)
			}

			// slices.Equal, not reflect.DeepEqual: an empty token list is
			// legitimately nil from Encode and [] from JSON, and they mean the
			// same thing.
			for _, c := range cases {
				if got := v.Encode(c.Text, true); !slices.Equal(got, c.BOS) {
					t.Errorf("Encode(%q, bos)\n got %v\nwant %v", c.Text, got, c.BOS)
				}
				if got := v.Encode(c.Text, false); !slices.Equal(got, c.NoBOS) {
					t.Errorf("Encode(%q, nobos)\n got %v\nwant %v", c.Text, got, c.NoBOS)
				}
			}
			t.Logf("%d cases exact", len(cases))
		})
	}
}

// stories260K is the repository's own fixture, or the container converted from
// it where the fixture is not shipped (a cross-compiled test binary ships with
// its package's testdata, not the repository's).
func stories260K(t *testing.T) *src {
	t.Helper()
	f := openSrc(t, strings.TrimSuffix(models()["stories260K"], ".gguf"), testmodels.Path("stories260K"))
	if f == nil {
		t.Fatal("stories260K is neither the testdata GGUF nor a container under " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory)")
	}
	return f
}

// TestRoundTrip: decoding what we encoded must give the text back. The leading
// space that AddSpacePrefix inserts is real and comes back, so it is stripped
// here rather than pretended away.
func TestRoundTrip(t *testing.T) {
	f := stories260K(t)
	defer f.Close()
	v, err := tok.New(f.vocab(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"Hello world", "one two three", "a", "日本", "tab\there"} {
		ids := v.Encode(s, false)
		got := v.Decode(ids)
		if len(got) > 0 && got[0] == ' ' {
			got = got[1:]
		}
		if got != s {
			t.Errorf("round trip %q -> %v -> %q", s, ids, got)
		}
	}
}

func TestVocabShape(t *testing.T) {
	f := stories260K(t)
	defer f.Close()
	v, err := tok.New(f.vocab(t))
	if err != nil {
		t.Fatal(err)
	}
	if v.BOS != 1 || v.EOS != 2 || v.Unk != 0 {
		t.Errorf("bos/eos/unk = %d/%d/%d, want 1/2/0", v.BOS, v.EOS, v.Unk)
	}
	// All 256 byte-fallback tokens must exist, or the first non-ASCII input
	// silently becomes <unk> instead of round-tripping.
	for b := 0; b < 256; b++ {
		if v.ByteToken(b) < 0 {
			t.Fatalf("byte 0x%02x has no fallback token", b)
		}
	}
}
