package convert

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestMarkerHeadActivationsAreStated converts Laya's fixture and holds its
// statement to the head's graph: every head block after the trunk has the
// ReLU feed-forward Laya's head blocks run (max(x, 0), ungated), and the
// trunk keeps its own GELU-erf GeGLU. A head block left unstated is the
// violation: the front end has no activation to build it from.
func TestMarkerHeadActivationsAreStated(t *testing.T) {
	src := testmodels.Path("laya/fixture/laya-fixture.gguf")
	if !fileExists(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/layagold.py makes it) -- this gate proved nothing", src)
	}
	dst := filepath.Join(t.TempDir(), "laya.jlm")
	if _, err := FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c := f.Config()
	if c.DecisionBlocks == 0 {
		t.Fatalf("the fixture states no head blocks: the gate would check nothing")
	}
	var e expr
	e.Fn(jlm.FnMax, operand(0), num(0))
	relu := e.T
	for li := range int(c.NLayer + c.DecisionBlocks) {
		a := c.ActAt(li)
		head := li >= int(c.NLayer)
		switch {
		case a == nil:
			t.Errorf("block %d of %d (%d head blocks) states no activation", li, c.NLayer+c.DecisionBlocks, c.DecisionBlocks)
		case head && !slices.Equal(a.Terms, relu):
			t.Errorf("head block %d states %v, want ReLU %v", li, a.Terms, relu)
		case !head && slices.Equal(a.Terms, relu):
			t.Errorf("trunk block %d states the head's ReLU", li)
		}
	}
	if c.Revision != jlm.StatementRevision {
		t.Errorf("stated at revision %d, want %d", c.Revision, jlm.StatementRevision)
	}
}
