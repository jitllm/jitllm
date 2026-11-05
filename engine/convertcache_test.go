package engine

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestCachedContainerWasWrittenByThisBuild is the gate on the test suite's own
// converter cache. Reusing a container because it is newer than its GGUF never
// expires anything (a GGUF never changes), and a stale container from an
// earlier build once made a fresh converter look broken on arm64. It checks the
// result, not the predicate: whatever path jlmOf took, the container must carry
// this binary's identity.
func TestCachedContainerWasWrittenByThisBuild(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	if _, err := os.Stat(src); err != nil {
		t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", src)
	}
	dst := jlmOfPair(t, src, "")

	// The violation, run first: a readable container of the current version,
	// written by somebody else.
	f, err := jlm.Open(dst)
	if err != nil {
		t.Fatalf("open %s: %v", dst, err)
	}
	f.Close()
	foreignPath := filepath.Join(t.TempDir(), "foreign"+jlm.Ext)
	if _, err := convert.FromGGUF(src, foreignPath, jlm.Fingerprint{Host: "test", Writer: "go1.0.0 deadbeefcafe"}); err != nil {
		t.Fatalf("write the foreign container: %v", err)
	}
	g, err := jlm.Open(foreignPath)
	if err != nil {
		t.Fatalf("open the foreign container: %v -- the violation must be a READABLE "+
			"container of the current version, or this gate is testing jlm.Open", err)
	}
	foreign := g.FP.Writer
	g.Close()
	if foreign == BuildIDForTest() {
		t.Fatalf("the violation is not one: a container stamped %q reads back as this build", foreign)
	}

	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatalf("open %s: %v", dst, err)
	}
	defer c.Close()
	if got, want := c.FP.Writer, BuildIDForTest(); got != want {
		t.Fatalf("the cache served a container written by %q, this build is %q -- "+
			"every gate reading it is measuring a converter nobody is running", got, want)
	}
	t.Logf("%s: Writer %q, this build", dst, c.FP.Writer)
}
