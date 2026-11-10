package srcgate

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTheDeclinesOnlyShrink holds nn's decline reasons to the list below. A
// device may decline a part only for a reason the engine allows -- the part
// served, the caller keeping it, its buffers not fitting, a CPU below the
// emitter's baseline, and the composition's seal: a step that meets a part
// placement did not form, or a block not composed for the device, may not
// compose it (TestComposesOnlyWithAnInstance) -- or for one still open: a refusal to be implemented,
// never one to add. A reason added fails the gate; one implemented away
// leaves the list with it. The violation, a reason the list does not hold,
// is seen.
func TestTheDeclinesOnlyShrink(t *testing.T) {
	testmodels.SourceTree(t)
	b, err := os.ReadFile(filepath.Join("..", "..", "nn", "decline.go"))
	if err != nil {
		t.Fatal(err)
	}
	allowed := []string{"DeclineServes", "DeclineCaller", "DeclineNoRoom", "DeclineISA", "DeclineNotFormed", "DeclineUnplaced"}
	open := []string{
		"DeclineNoGraph",
		"DeclineNoPool",
		"DeclineWindow",
		"DeclineHeadShape",
		"DeclineTableHeld",
		"DeclineFeeds",
		"DeclineRecurrent",
		"DeclineNoWeight",
		"DeclineExperts",
	}
	got := declineReasons(string(b))
	if len(got) < len(allowed) {
		t.Fatalf("read %d decline reasons: the gate read nothing", len(got))
	}
	for _, n := range got {
		if !slices.Contains(allowed, n) && !slices.Contains(open, n) {
			t.Errorf("nn declares %s: a new reason to refuse a part, where the engine composes it", n)
		}
	}
	for _, n := range open {
		if !slices.Contains(got, n) {
			t.Errorf("%s is no longer declared: take it off the open list", n)
		}
	}
	t.Logf("%d decline reasons: %d the engine allows, %d open", len(got), len(allowed), len(open))
	if v := declineReasons("const (\n\tDeclineServes Decline = iota\n\tDeclineSomethingNew\n)"); len(v) != 2 ||
		slices.Contains(allowed, v[1]) || slices.Contains(open, v[1]) {
		t.Errorf("the violation, a reason the list does not hold, was not seen (%v)", v)
	}
}

// declineReasons is every Decline the const block of src declares.
func declineReasons(src string) []string {
	i := strings.Index(src, "const (")
	if i < 0 {
		return nil
	}
	block := src[i:]
	if j := strings.Index(block, "\n)"); j >= 0 {
		block = block[:j]
	}
	var out []string
	for _, m := range regexp.MustCompile(`(?m)^\t(Decline[A-Za-z]+)`).FindAllStringSubmatch(block, -1) {
		out = append(out, m[1])
	}
	return out
}
