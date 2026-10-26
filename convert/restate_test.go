package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRestatedIsAFreshConversion holds a container written before the
// converter stated its activations -- the config record without its
// thirteenth tail, as an older writer laid it -- to what `jitllm upgrade`
// makes of it (jlm.Restate with StateStored): byte for byte the container a
// fresh conversion writes, and left alone when upgraded again. The models
// cover a gated SiLU, an ungated squared ReLU, xIELU's stored numbers and
// Gemma 3n's layers of two activations.
func TestRestatedIsAFreshConversion(t *testing.T) {
	fp := jlm.Fingerprint{Host: "test", Writer: "test"}
	for _, name := range []string{"stories15M-q8_0.gguf", "synth-nemotron.gguf", "synth-apertus.gguf", "synth-gemma3n.gguf"} {
		t.Run(name, func(t *testing.T) {
			src := testmodels.Path(name)
			if _, err := os.Stat(src); err != nil {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", src)
			}
			dir := t.TempDir()
			fresh := filepath.Join(dir, "fresh"+jlm.Ext)
			if _, err := FromGGUF(src, fresh, fp); err != nil {
				t.Fatal(err)
			}
			want, err := os.ReadFile(fresh)
			if err != nil {
				t.Fatal(err)
			}
			f, err := gguf.Open(src)
			if err != nil {
				t.Fatal(err)
			}
			s, err := buildGGUF(f, nil, src, "", options{})
			if err != nil {
				f.Close()
				t.Fatal(err)
			}
			acts := len(s.Config.Acts)
			s.Config.Acts, s.Config.ActLayers = nil, nil
			old := filepath.Join(dir, "old"+jlm.Ext)
			_, err = jlm.Write(old, s, fp)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if c, err := jlm.Open(old); err != nil {
				t.Fatal(err)
			} else {
				stated := c.Config().ActAt(0) != nil
				c.Close()
				if stated {
					t.Fatal("the container written without its activations states one")
				}
			}
			if changed, err := jlm.Restate(old, StateStored); err != nil || !changed {
				t.Fatalf("restating: changed %v, %v", changed, err)
			}
			got, err := os.ReadFile(old)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("the restated container (%d bytes) is not the fresh one (%d bytes)", len(got), len(want))
			}
			if changed, err := jlm.Restate(old, StateStored); err != nil || changed {
				t.Fatalf("restating a container that states its activations: changed %v, %v", changed, err)
			}
			t.Logf("%d activations stated, %d bytes as a fresh conversion writes them", acts, len(got))
		})
	}
}
