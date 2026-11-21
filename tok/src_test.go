package tok_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// src is a model's tokenizer as a test reads it: from the GGUF through the
// converter where the GGUF is on this host, and from the container's own
// tokenizer section where only the container is.
//
// The container is not a weaker input: its section is exactly what conversion
// wrote, and a host may hold only containers.
type src struct {
	name string
	f    *meta.File // the GGUF, or nil
	jv   *jlm.Vocab // the container's section, when f is nil
	pre_ string     // a pre name set by renamePre on the container arm
}

// openSrc opens the first of stems (paths without an extension) that exists
// as a GGUF or, failing that, as a container.
func openSrc(t testing.TB, stems ...string) *src {
	t.Helper()
	for _, stem := range stems {
		if f, err := gguf.Open(stem + ".gguf"); err == nil {
			return &src{name: filepath.Base(stem) + ".gguf", f: f}
		}
		if c, err := jlm.Open(stem + ".jlm"); err == nil {
			jv := c.Vocab()
			c.Close()
			if jv != nil {
				return &src{name: filepath.Base(stem) + ".jlm", jv: jv}
			}
		}
	}
	return nil
}

// stem is the model's file name without its extension, for pinning a measured
// number on the model it was taken from whichever form was opened.
func (s *src) stem() string { return strings.TrimSuffix(s.name, filepath.Ext(s.name)) }

// vocab is a fresh tokenizer section: a new conversion of the GGUF, or a copy of
// the container's, so a test may build several vocabularies from one source.
func (s *src) vocab(t testing.TB) *jlm.Vocab {
	t.Helper()
	if s.f != nil {
		return vocabOf(t, s.f)
	}
	cp := *s.jv
	if s.pre_ != "" {
		cp.PreName, cp.Pre = s.pre_, nil
	}
	return &cp
}

// kind is the tokenizer family in GGUF's words: "gpt2" or "llama".
func (s *src) kind() string {
	if s.f != nil {
		k, _ := s.f.KV["tokenizer.ggml.model"].String()
		return k
	}
	switch s.jv.Kind {
	case jlm.VocabBPE:
		return "gpt2"
	case jlm.VocabSPM:
		return "llama"
	}
	return ""
}

func (s *src) pre() string {
	if s.f != nil {
		p, _ := s.f.KV["tokenizer.ggml.pre"].String()
		return p
	}
	if s.pre_ != "" {
		return s.pre_
	}
	return s.jv.PreName
}

// renamePre makes the pre-tokenizer a name no table carries, the state a
// converter that wrote such a name leaves: in the GGUF's header, or in the
// container's section as conversion stores it -- the name with no stages.
func (s *src) renamePre(name string) {
	if s.f != nil {
		s.f.KV["tokenizer.ggml.pre"] = meta.Value{Type: meta.String, Raw: []byte(name)}
		return
	}
	s.pre_ = name
}

func (s *src) Close() {
	if s.f != nil {
		s.f.Close()
	}
}

// modelStems is every model in the model directory by stem, whichever form it is in.
func modelStems() []string {
	seen := map[string]bool{}
	var out []string
	for _, ext := range []string{".gguf", ".jlm"} {
		paths := testmodels.Glob("*" + ext)
		for _, p := range paths {
			if s := strings.TrimSuffix(p, ext); !seen[s] {
				seen[s] = true
				out = append(out, s)
			}
		}
	}
	return out
}

// checkedInSource reads a file of this repository's source that a test compares
// the generated tables against.
//
// It skips only for a shipped binary (JITLLM_SHIPPED_BINARY, set in a
// cross-compiled binary's environment on another host), whose neighbouring
// source is not the one it was built from; the build tree runs the comparison
// every time.
func checkedInSource(t *testing.T, path string) []byte {
	t.Helper()
	if os.Getenv("JITLLM_SHIPPED_BINARY") != "" {
		t.Skipf("LOUD SKIP -- NOT A PASS: %s is source, and this is a shipped test binary running "+
			"away from the tree it was built from. That tree runs this check.", path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
