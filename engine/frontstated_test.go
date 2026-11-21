package engine

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAContainerStatingNoFrontIsUpgraded holds the front end to the
// container's statements outside the blocks: a container that states none
// (a v33 container's record, stepped to v34) is refused naming jitllm
// upgrade, and the upgrade's restatement (convert.StateStored) brings it back
// to the fresh conversion byte for byte, and a model that draws to its
// greedy tokens. The models cover a plain head and input rows, an embedding
// model's readout, a marker head's, the head over several streams with
// per-layer inputs and a softcap, and prediction blocks.
func TestAContainerStatingNoFrontIsUpgraded(t *testing.T) {
	draws := map[string]bool{"stories15M-q8_0.gguf": true, "synth-gemma3n.gguf": true, "synth-qwen35-hybrid-mtp.gguf": true}
	ran := 0
	for _, name := range []string{"stories15M-q8_0.gguf", "embed/MiniLM-L6-v2.Q8_0.gguf", "laya/fixture/laya-fixture.gguf",
		"synth-gemma3n.gguf", "synth-qwen35-hybrid-mtp.gguf"} {
		t.Run(name, func(t *testing.T) {
			src := testmodels.Path(name)
			if _, err := os.Stat(src); err != nil {
				testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS)", src)
			}
			fresh := jlmOf(t, src)
			old := filepath.Join(t.TempDir(), "old"+jlm.Ext)
			b, err := os.ReadFile(fresh)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(old, b, 0o644); err != nil {
				t.Fatal(err)
			}
			stated := 0
			if _, err := jlm.Restate(old, func(_ *jlm.File, c *jlm.Config) (bool, error) {
				stated = len(c.Front.Names)
				c.Front = jlm.FrontGraphs{}
				return true, nil
			}); err != nil {
				t.Fatal(err)
			}
			if stated == 0 {
				t.Fatal("the fresh conversion states no part outside its blocks: the case proves nothing")
			}
			if m, err := model.Open(old); err == nil {
				m.Close()
				t.Fatal("a container stating no part outside its blocks opened")
			} else if !strings.Contains(err.Error(), "jitllm upgrade") {
				t.Fatalf("the refusal %q does not name jitllm upgrade", err)
			}
			if changed, err := jlm.Restate(old, convert.StateStored); err != nil || !changed {
				t.Fatalf("upgrade's restatement: changed %v, %v", changed, err)
			}
			if mb, err := os.ReadFile(old); err != nil {
				t.Fatal(err)
			} else if !slices.Equal(mb, b) {
				t.Fatal("the restated container is not the fresh conversion byte for byte")
			}
			ran++
			if !draws[name] {
				t.Logf("%d parts stated again", stated)
				return
			}
			want, _, _ := greedy(t, fresh, "The capital of France is", 8)
			got, _, _ := greedy(t, old, "The capital of France is", 8)
			if !slices.Equal(got, want) {
				t.Fatalf("the restated container draws %v, the fresh one %v", got, want)
			}
			t.Logf("%d parts stated again, %d greedy tokens alike", stated, len(got))
		})
	}
	if ran != 5 {
		testmodels.Missing(t, "%d of 5 models ran", ran)
	}
}
