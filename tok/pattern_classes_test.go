package tok

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/uniprop"
	"github.com/jitllm/jitllm/tok/pretok"
)

// TestShippedSplitIsPinned drives the anti-revert pins of
// TestPinnedUnicodeTables through the pipeline that ships, llama-bpe's
// pattern on the engine: a letter Go's newer Unicode tables know and 15.1.0
// leaves unassigned is no letter here, so "a<U+1C89>b" is two pieces.
func TestShippedSplitIsPinned(t *testing.T) {
	p, err := buildPipeline(pretok.Table()["llama-bpe"])
	if err != nil {
		t.Fatal(err)
	}
	for text, want := range map[string]int{"aᲉb": 2, "a࢏b": 2, "aꟋb": 2, "ab": 1} {
		if got := len(p.apply(text)); got != want {
			t.Errorf("%q splits into %d pieces, want %d: the engine is not reading the pinned %s tables", text, got, want, uniprop.Version)
		}
	}
}

// TestClassesAreTheLibrarys holds each class the patterns use -- \w, \d, \s
// and the properties -- to the tokenizers library's own regex over every code
// point (testdata/pattern_classes.txt, scripts/patternclasses.py). A code
// point the pinned Unicode leaves unassigned may differ, since the library
// reads a later Unicode; every assigned one must be classed alike.
func TestClassesAreTheLibrarys(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "pattern_classes.txt"))
	if err != nil {
		t.Fatal(err)
	}
	cn, err := uniprop.Spans(uniprop.Key("", "Cn"))
	if err != nil {
		t.Fatal(err)
	}
	unassigned := spanSet(cn)
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) < 15 {
		t.Fatalf("the fixture holds %d classes", len(lines))
	}
	for _, line := range lines {
		f := strings.Fields(line)
		var lib spanSet
		for _, r := range f[2:] {
			lo, hi, _ := strings.Cut(r, "-")
			a, errA := strconv.ParseUint(lo, 16, 32)
			z, errZ := strconv.ParseUint(hi, 16, 32)
			if errA != nil || errZ != nil {
				t.Fatalf("%s: range %q", f[0], r)
			}
			lib = append(lib, uniprop.Span{Lo: rune(a), Hi: rune(z)})
		}
		sp, err := compileSplit(f[0], "Isolated", false)
		if err != nil {
			t.Fatal(err)
		}
		c := &sp.sets[sp.prog[0].set]
		var wrong []rune
		later := 0
		for r := rune(0); r <= 0x10FFFF; r++ {
			if r >= 0xD800 && r <= 0xDFFF || c.has(r) == lib.has(r) {
				continue
			}
			if unassigned.has(r) {
				later++
				continue
			}
			wrong = append(wrong, r)
		}
		if len(wrong) > 0 {
			t.Errorf("%s: %d assigned code points classed unlike the library, e.g. %U", f[0], len(wrong), wrong[:min(len(wrong), 12)])
		}
		t.Logf("%s: every assigned code point as the library; %d assigned after %s", f[0], later, uniprop.Version)
	}
}
