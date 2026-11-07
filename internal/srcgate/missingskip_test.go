package srcgate

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestAMissingModelFailsTheGate holds every test file to testmodels.Missing
// for a model or fixture that is not there: it fails the gate, and skips
// only in a run that asked for no models. A bare t.Skip on an absent model
// is a green line that proved nothing (RULE 10), and a gate that never ran
// on a box reads as passed there. The pattern must find the shape it
// refuses.
func TestAMissingModelFailsTheGate(t *testing.T) {
	testmodels.SourceTree(t)
	bare := regexp.MustCompile(`t\.Skipf?\("(MODEL MISSING|model not present|fixture not present)`)
	if !bare.MatchString(`		t.Skipf("model not present: %s", path)`) {
		t.Fatal("the pattern does not see a bare skip on an absent model")
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	read := 0
	var found []string
	for _, f := range trackedFiles(t, root) {
		if !strings.HasSuffix(f, "_test.go") || f == "internal/srcgate/missingskip_test.go" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, f))
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		read++
		for i, line := range strings.Split(string(b), "\n") {
			if bare.MatchString(line) {
				found = append(found, f+":"+strconv.Itoa(i+1))
			}
		}
	}
	if read < 300 {
		t.Fatalf("read %d test files under %s: the tree was not found, so this gate proved nothing", read, root)
	}
	if len(found) > 0 {
		t.Fatalf("%d test(s) skip on an absent model instead of failing through testmodels.Missing:\n%s",
			len(found), strings.Join(found, "\n"))
	}
}
