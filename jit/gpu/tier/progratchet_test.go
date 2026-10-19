package tier

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The device program's deviation from its design, counted in its source:
// launches of kernels a scratch built for an architecture, references to a
// family's mechanisms, kernels chosen by the slot a value lands in, and
// slots named for a family. Every op is to run a kernel generated from
// itself (genop.go), so each count only falls; a count above its ceiling
// fails. Lower a ceiling to the count in the same change that lowers it.
var progCeilings = map[string]int{
	"scratch kernel launches": 189,
	"family references":       297,
	"kernels chosen by slot":  54,
	"family-named slots":      68,
}

var progCounters = map[string]*regexp.Regexp{
	"scratch kernel launches": regexp.MustCompile(`\b(lc\.la|lc\.launch|lg|laRows|ds4Rows)\((bs|pk|ml|mg|x|kd|ix|ms|qs)\.`),
	"family references":       regexp.MustCompile(`\b(bs\.(ds4|k3|q4e)|pk\.(msa|qsa|idx)|l\.(ds4|k3|q4e)|p\.(DS4|Q4E)|(ms|qs|ix)\.|K3Fault\w*|DS4Fault\w*)\b`),
	"kernels chosen by slot":  regexp.MustCompile(`prog\.slots\[o\.(Out|In)\[[0-9]\]\] *(==|!=)|switch prog\.slots`),
}

// familySlot names a slot after a family's mechanism.
var familySlot = regexp.MustCompile(`^\s+slot(C[A-Z]|N[A-Z]|B[A-Z]|H[CFM]|Idx|IK|IQ|SQ8|SX|SG|SM|SI|SW|SLo|Lat|MGate|MGated|RBias)`)

// genFiles is the program's generators: what an op's kernel is built by.
var genFiles = []string{"genop.go", "conv.go", "convgraph.go", "kernid.go", "devrewrite.go"}

// The generators' own deviation: kernel classes by another name. A named
// kernel id, a kernel taken from the menu rather than formed from the op's
// fragments, a family's mechanism as an activation, and an op handed to a
// former over the old kernels. Each only falls.
var genCeilings = map[string]int{
	"named kernel ids":           21,
	"menu kernels in generators": 35,
	"family activations":         0,
	"formers over old kernels":   5,
}

// fragments are the kernels package's pieces a generator stitches, and its
// pure arithmetic on sizes: not kernels.
var fragments = map[string]bool{
	"ClampWindow": true, "ActMulOK": true, "QuantizeThreads": true, "IsFloat": true,
	"ActFragment": true, "ActMulFragment": true, "QuantFragment": true, "TreeSum": true, "TreeMax": true,
}

var (
	kernCall  = regexp.MustCompile(`kernels\.([A-Z][A-Za-z0-9]*)\(`)
	kidLine   = regexp.MustCompile(`(?m)^\s+kid[A-Z][A-Za-z0-9]*\b`)
	familyAct = regexp.MustCompile(`\b(kernels|graph)\.Act(SwiGLUOAI|SwiGLUClamp|Situ)\b`)
	formerUse = regexp.MustCompile(`former: form[A-Z]`)
	twinKern  = regexp.MustCompile(`\bk: bm\.(act|kern|red)\b`)
)

func TestGeneratorsComposeFromFragments(t *testing.T) {
	got := map[string]int{}
	for _, f := range genFiles {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		src := string(b)
		// A batched twin's own kernels launched by a generator are menu
		// kernels too, though no kernels.X call names them.
		for _, m := range twinKern.FindAllString(src, -1) {
			got["menu kernels in generators"]++
			t.Logf("%s: menu kernel %s", f, m)
		}
		for _, m := range kernCall.FindAllStringSubmatch(src, -1) {
			if !fragments[m[1]] {
				got["menu kernels in generators"]++
				t.Logf("%s: menu kernel kernels.%s", f, m[1])
			}
		}
		got["named kernel ids"] += len(kidLine.FindAllString(src, -1))
		got["family activations"] += len(familyAct.FindAllString(src, -1))
		got["formers over old kernels"] += len(formerUse.FindAllString(src, -1))
	}
	for name, ceil := range genCeilings {
		t.Logf("%-28s %4d (ceiling %d)", name, got[name], ceil)
		if got[name] > ceil {
			t.Errorf("%s rose to %d past its ceiling %d: a generator composes from fragments", name, got[name], ceil)
		}
	}
}

func TestProgramDeviationOnlyFalls(t *testing.T) {
	enc, err := os.ReadFile("encoder.go")
	if err != nil {
		t.Fatal(err)
	}
	const begin, end = "the device program's executor begins here", "the device program's executor ends here"
	s := string(enc)
	i, j := strings.Index(s, begin), strings.Index(s, end)
	if i < 0 || j < i {
		t.Fatal("encoder.go no longer marks the device program's executor")
	}
	prog, err := os.ReadFile("devprog.go")
	if err != nil {
		t.Fatal(err)
	}
	src := s[i:j] + string(prog)
	got := map[string]int{}
	for name, re := range progCounters {
		got[name] = len(re.FindAllString(src, -1))
	}
	for _, line := range strings.Split(string(prog), "\n") {
		if familySlot.MatchString(line) {
			got["family-named slots"]++
		}
	}
	for name, ceil := range progCeilings {
		t.Logf("%-24s %4d (ceiling %d)", name, got[name], ceil)
		if got[name] > ceil {
			t.Errorf("%s rose to %d past its ceiling %d: every op runs a kernel generated from itself", name, got[name], ceil)
		}
	}
}

// oldTier is the block encoder's own files: the scratches, the per-family
// launch tables and the encoder they serve, deleted with it (D5). Every other
// file of the package -- the program, its generators and rewrites, and any
// file added -- holds no family mechanism at all, so moving one out of the
// executor into a file of its own does not lower a count.
var oldTier = map[string]bool{
	"encoder.go": true, "mixers.go": true, "indexer.go": true, "ds4.go": true, "q4e.go": true,
	"mladecode.go": true, "k3.go": true, "altup.go": true, "prep.go": true, "qsa.go": true,
	"gemma4.go": true, "pagedattn.go": true, "tier.go": true, "pagedstream.go": true,
	"mlabatch.go": true, "scratchinit.go": true, "moegroup.go": true, "call.go": true,
	"recipe.go": true, "ffnform.go": true, "rows.go": true, "persist.go": true, "msa.go": true,
	"lifecycle.go": true, "inflight.go": true, "forget.go": true, "addnorm.go": true,
	"devprog.go": true, // counted above, against its ceiling
}

func TestNoFamilyMechanismOutsideTheOldTier(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || oldTier[f] {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		n++
		for name, re := range progCounters {
			if c := len(re.FindAllString(string(src), -1)); c != 0 {
				t.Errorf("%s: %d %s; only the old tier may hold them", f, c, name)
			}
		}
	}
	if n < 50 {
		t.Fatalf("read %d files: the glob did not find the package", n)
	}
}
