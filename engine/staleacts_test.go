package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOpenRefusesAContainerThatStatesNoGraphs holds Open to refusing a
// container written before the converter stated its blocks' graphs -- one
// written before it stated its activations is one -- naming the conversion
// that states them, and to refusing it still once `jitllm upgrade`'s
// restatement has run, which states no graph.
func TestOpenRefusesAContainerThatStatesNoGraphs(t *testing.T) {
	b, err := os.ReadFile(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	old := filepath.Join(t.TempDir(), "old"+model.Ext)
	if err := os.WriteFile(old, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(old, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		c.Acts, c.ActLayers, c.Blocks = nil, nil, jlm.BlockGraphs{}
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("writing the container without its graphs: changed %v, %v", changed, err)
	}
	refused := func(what string) {
		t.Helper()
		m, err := model.Open(old)
		if err == nil {
			m.Close()
			t.Fatalf("%s that states no graphs opened", what)
		}
		if !strings.Contains(err.Error(), "jitllm convert") {
			t.Fatalf("%s is refused naming no conversion: %v", what, err)
		}
	}
	refused("a container")
	if _, err := jlm.Restate(old, convert.StateStored); err != nil {
		t.Fatalf("restating: %v", err)
	}
	refused("the upgraded container")
}
