package engine

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOpenRefusesARotaryLayoutItDoesNotState holds Open to refusing a
// container with rotary sections written before revision 3, which states no
// layout for them (jlm.FlagRopeInterleaved, jlm.FlagRopeXD) and would turn
// Qwen3-VL's interleaved split as contiguous runs, with the upgrade command;
// and to opening it, the split interleaved, once `jitllm upgrade`'s
// restatement states the layout.
func TestOpenRefusesARotaryLayoutItDoesNotState(t *testing.T) {
	path := testmodels.Path("synth-qwen3vl.gguf")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS)", path)
	}
	b, err := os.ReadFile(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	old := filepath.Join(t.TempDir(), "old"+model.Ext)
	if err := os.WriteFile(old, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(old, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		if c.RopeSections == [4]uint32{} || !c.Flags.Has(jlm.FlagRopeInterleaved) {
			t.Fatalf("the fixture states sections %v, interleaved %v: it tests nothing", c.RopeSections, c.Flags.Has(jlm.FlagRopeInterleaved))
		}
		c.Flags &^= jlm.FlagRopeInterleaved | jlm.FlagRopeXD
		c.Revision = 2
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("writing the container at revision 2: changed %v, %v", changed, err)
	}
	m, err := model.Open(old)
	var up *model.UpgradeRequiredError
	if !errors.As(err, &up) {
		if m != nil {
			m.Close()
		}
		t.Fatalf("a container with sections and no stated layout opened (err %v)", err)
	}
	if !strings.Contains(err.Error(), "jitllm upgrade "+old) {
		t.Fatalf("the refusal names no upgrade command: %v", err)
	}
	if changed, err := jlm.Restate(old, convert.StateStored); err != nil || !changed {
		t.Fatalf("restating: changed %v, %v", changed, err)
	}
	m, err = model.Open(old)
	if err != nil {
		t.Fatalf("the upgraded container: %v", err)
	}
	defer m.Close()
	if !m.Cfg.RopeInterleaved {
		t.Fatal("the upgraded container's split is not interleaved")
	}
}
