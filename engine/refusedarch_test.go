package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRefusedArchitecturesSayWhy: a container whose graph has no composed
// form stated from it is refused at Open, naming what it carries, rather
// than opened and run on code of its own.
func TestRefusedArchitecturesSayWhy(t *testing.T) {
	for _, c := range []struct{ name, why string }{
		{"synth-deepseek4.gguf", "compressed history"},
	} {
		t.Run(c.name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(c.name))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(c.name))
			}
			m, err := model.Open(jlmOf(t, src))
			if err == nil {
				m.Close()
				t.Fatalf("Open took a %s container", c.name)
			}
			if !strings.Contains(err.Error(), c.why) {
				t.Fatalf("Open: %v, want a refusal naming %q", err, c.why)
			}
			t.Logf("refused: %v", err)
		})
	}
}
