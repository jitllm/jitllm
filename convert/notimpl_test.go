package convert_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestScopeRefusalIsDistinguishableFromABreak is the gate on a gate. The
// sweeping test helpers skip on convert.ErrNotImplemented and fail on
// everything else, which is only safe while the two classes stay apart: an
// over-broad skip predicate would turn every oracle in
// model.TestGreedyMatchesLlamaCpp into a no-op (RULE 10). So both directions
// are asserted.
func TestScopeRefusalIsDistinguishableFromABreak(t *testing.T) {
	dst := filepath.Join(t.TempDir(), "out"+jlm.Ext)

	t.Run("a break is NOT a scope refusal", func(t *testing.T) {
		// Not a GGUF at all. This is the class the helpers must still fail on.
		bad := filepath.Join(t.TempDir(), "notagguf.gguf")
		if err := os.WriteFile(bad, []byte("this is not a GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := convert.FromGGUF(bad, dst, jlm.Fingerprint{Host: "test"})
		if err == nil {
			t.Fatal("a 18-byte text file converted; the gate below proves nothing")
		}
		if errors.Is(err, convert.ErrNotImplemented) {
			t.Fatalf("a corrupt file reads as a SCOPE refusal, so every sweeping "+
				"gate would skip a real break: %v", err)
		}
	})

	t.Run("an unimplemented projector IS a scope refusal", func(t *testing.T) {
		// The violation is constructed, not borrowed: renaming the projector
		// in a copy of a real mmproj stays a violation however the implemented
		// list grows (pointing at a real unimplemented projector rotted when it
		// was implemented). The rename keeps the byte length, so the only thing
		// the converter can object to is the name.
		mm := testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
		if _, err := os.Stat(mm); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this half proved nothing", mm, err)
		}
		raw, err := os.ReadFile(mm)
		if err != nil {
			t.Fatal(err)
		}
		// The value is a length-prefixed string in the KV block; find the
		// literal and overwrite it in place.
		const was, now = "idefics3", "notaproj"
		i := bytes.Index(raw, []byte("clip.projector_type"))
		if i < 0 {
			t.Fatalf("%s has no clip.projector_type", mm)
		}
		j := bytes.Index(raw[i:i+256], []byte(was))
		if j < 0 {
			t.Fatalf("clip.projector_type is not %q in %s", was, mm)
		}
		copy(raw[i+j:], now)
		bad := filepath.Join(t.TempDir(), "renamed.gguf")
		if err := os.WriteFile(bad, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		_, err = convert.FromGGUF(bad, dst, jlm.Fingerprint{Host: "test"})
		if err == nil {
			t.Fatalf("projector_type %q converted; this engine implements a LIST", now)
		}
		if !errors.Is(err, convert.ErrNotImplemented) {
			t.Fatalf("an unimplemented projector does not carry the sentinel, so "+
				"every sweeping gate goes RED for a RULE 7 scope decision: %v", err)
		}
	})
}
