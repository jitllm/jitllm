package engine_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// testInstance is the external test package's twin of the internal one:
// engine.NewInstance for a gate, a refusal failing the test.
func testInstance(tb testing.TB, m *model.Model, maxSeq int, opts ...engine.Option) *engine.Instance {
	tb.Helper()
	s, err := engine.NewInstance(m, maxSeq, opts...)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if d := s.DeviceDemotions(); d > 0 && !engine.FaultsInjected {
			tb.Errorf("the device failed %d time(s) and every block moved to the CPU device: %+v", d, s.Evacuations())
		}
	})
	return s
}

// jlmOf is the external test package's twin of the internal jlmOf (see
// jlm_helper_test.go): convert a GGUF to a container beside it, failing
// rather than skipping when conversion fails.
func jlmOf(t testing.TB, path string) string {
	t.Helper()
	if strings.HasSuffix(path, jlm.Ext) {
		return path
	}
	dst := strings.TrimSuffix(path, filepath.Ext(path)) + jlm.Ext
	if _, err := os.Stat(path); err != nil {
		// The container is the model and its source is disposable; see
		// containerAlone in jlm_helper_test.go, which this mirrors.
		c, err := jlm.Open(dst)
		if err != nil {
			testmodels.Missing(t, "MODEL MISSING: %s, and its container %s does not open (%v) (set JITLLM_MODELS to the model directory) -- this "+
				"gate proved nothing", path, dst, err)
		}
		c.Close()
		return dst
	}
	if di, err := os.Stat(dst); err == nil {
		if si, err := os.Stat(path); err == nil && di.ModTime().After(si.ModTime()) {
			if c, err := jlm.Open(dst); err == nil {
				stale := c.FP.Writer != engine.BuildIDForTest()
				c.Close()
				if !stale {
					return dst
				}
			}
		}
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), strings.TrimSuffix(filepath.Base(dst), jlm.Ext)+".tmp-*"+jlm.Ext)
	if err != nil {
		t.Fatalf("convert %s: %v", path, err)
	}
	tmp.Close()
	os.Remove(tmp.Name())
	var opts []convert.Option
	if c := engine.ModelConfigForTest(path); c != "" {
		opts = append(opts, convert.WithModelConfig(c))
	}
	if _, err := convert.FromGGUF(path, tmp.Name(), jlm.Fingerprint{Host: "test", Writer: engine.BuildIDForTest()}, opts...); err != nil {
		os.Remove(tmp.Name())
		if errors.Is(err, convert.ErrPart) {
			t.Skipf("PART OF A MODEL: %s -- %v", filepath.Base(path), err)
		}
		if errors.Is(err, convert.ErrBadInput) {
			t.Skipf("A BAD INPUT: %s -- %v", filepath.Base(path), err)
		}
		if errors.Is(err, convert.ErrNotImplemented) {
			t.Skipf("NOT IMPLEMENTED: %s -- %v (RULE 7 scope)", filepath.Base(path), err)
		}
		t.Fatalf("convert %s: %v", path, err)
	}
	if err := os.Rename(tmp.Name(), dst); err != nil {
		os.Remove(tmp.Name())
		t.Fatalf("convert %s: %v", path, err)
	}
	return dst
}

// languageModels is the internal helper's twin for the external test package:
// the language models available, projectors left out by
// general.architecture.
func languageModels(t testing.TB) []string {
	t.Helper()
	paths := testmodels.Glob("*.gguf")
	if len(paths) == 0 {
		// No GGUF is not "no models": a host may hold only containers.
		return containerModels(t)
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		f, err := gguf.Open(p)
		if err != nil {
			continue
		}
		arch, _ := f.KV["general.architecture"].String()
		f.Close()
		if arch == "clip" {
			continue
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		t.Fatal("every GGUF on this box is a projector; this sweep would prove nothing")
	}
	return out
}

// containerModels is languageModels' arm for a host that holds .jlm and no GGUF.
// The paths it returns are containers, which jlmOf passes straight through.
func containerModels(t testing.TB) []string {
	t.Helper()
	paths := testmodels.Glob("*.jlm")
	if len(paths) == 0 {
		testmodels.Missing(t, "%s", "no models under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- RULE 11: fetch some, do not record the absence")
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		f, err := jlm.Open(p)
		if err != nil {
			// A container this build cannot read is stale; say so.
			t.Logf("skipping %s: %v", filepath.Base(p), err)
			continue
		}
		c := f.Config()
		lm := c != nil && c.NLayer > 0 && c.NVocab > 0
		f.Close()
		if lm {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		t.Fatal("every container on this box is a projector or unreadable; this sweep would prove nothing")
	}
	return out
}
