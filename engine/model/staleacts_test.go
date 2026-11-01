package model

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestOpenRefusesAContainerThatStatesNoActivations holds Open, where the
// composer serves, to refusing a container written before the converter
// stated its activations -- every block of it would run on the tier's own
// kernels, unannounced -- with the upgrade command; and to opening it once
// `jitllm upgrade`'s restatement states them. Without the composer serving,
// main's kernels read it as before.
func TestOpenRefusesAContainerThatStatesNoActivations(t *testing.T) {
	b, err := os.ReadFile(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	old := filepath.Join(t.TempDir(), "old"+Ext)
	if err := os.WriteFile(old, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(old, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		c.Acts, c.ActLayers = nil, nil
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("writing the container without its activations: changed %v, %v", changed, err)
	}
	m, err := Open(old, noTune)
	var up *UpgradeRequiredError
	if !errors.As(err, &up) {
		if m != nil {
			m.Close()
		}
		t.Fatalf("a container that states no activations opened where the composer serves (err %v)", err)
	}
	if !strings.Contains(err.Error(), "jitllm upgrade "+old) {
		t.Fatalf("the refusal names no upgrade command: %v", err)
	}
	if !jit.ComposedByDefault {
		m, err := Open(old, noTune)
		if err != nil {
			t.Fatalf("main's kernels refused the container: %v", err)
		}
		m.Close()
	}
	if changed, err := jlm.Restate(old, convert.StateStored); err != nil || !changed {
		t.Fatalf("restating: changed %v, %v", changed, err)
	}
	m, err = Open(old, noTune)
	if err != nil {
		t.Fatalf("the upgraded container: %v", err)
	}
	m.Close()
}
