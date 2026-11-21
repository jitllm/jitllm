package convert

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestConvertRefusesAMismatchedPair: a projector whose rows are not the text
// model's width is refused at conversion, where there is a file in front of
// a person, the refusal naming both files; and a tower given as the text
// model is refused as one. The projector beside its own text model converts
// (TestQwen2VLStatesItsTower).
func TestConvertRefusesAMismatchedPair(t *testing.T) {
	text := testmodels.Path("stories15M-q8_0.gguf")
	mm := testmodels.Path("mmproj-synth-qwen2vl.gguf")
	for _, p := range []string{text, mm} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
		}
	}
	dst := filepath.Join(t.TempDir(), "pair.jlm")
	_, err := FromGGUFs(text, mm, dst, jlm.Fingerprint{Writer: "gate"})
	if err == nil {
		t.Fatal("a projector of another width converted beside stories15M")
	}
	if msg := err.Error(); !strings.Contains(msg, "do not belong together") || !strings.Contains(msg, mm) || !strings.Contains(msg, text) {
		t.Fatalf("the refusal does not name the mismatch and both files: %v", err)
	}
	t.Logf("refused: %v", err)
	if _, err := FromGGUFs(mm, mm, dst, jlm.Fingerprint{Writer: "gate"}); err == nil || !strings.Contains(err.Error(), "is itself a vision tower") {
		t.Fatalf("a tower given as the text model: %v", err)
	}
}
