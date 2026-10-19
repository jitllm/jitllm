package tier

import (
	"os"
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
	"scratch kernel launches": 222,
	"family references":       292,
	"kernels chosen by slot":  68,
	"family-named slots":      68,
}

var progCounters = map[string]*regexp.Regexp{
	"scratch kernel launches": regexp.MustCompile(`\b(lc\.la|lc\.launch|lg|laRows|ds4Rows)\((bs|pk|ml|mg|x|kd|ix|ms|qs)\.`),
	"family references":       regexp.MustCompile(`\b(bs\.(ds4|k3|q4e)|pk\.(msa|qsa|idx)|l\.(ds4|k3|q4e)|p\.(DS4|Q4E)|(ms|qs|ix)\.|K3Fault|DS4Fault)\b`),
	"kernels chosen by slot":  regexp.MustCompile(`prog\.slots\[o\.(Out|In)\[[0-9]\]\] *(==|!=)|switch prog\.slots`),
}

// familySlot names a slot after a family's mechanism.
var familySlot = regexp.MustCompile(`^\s+slot(C[A-Z]|N[A-Z]|B[A-Z]|H[CFM]|Idx|IK|IQ|SQ8|SX|SG|SM|SI|SW|SLo|Lat|MGate|MGated|RBias)`)

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
