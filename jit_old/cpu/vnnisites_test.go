//go:build amd64

package cpu

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestEveryVPDPBUSDSiteIsCoveredOrRefused asserts that every file emitting
// VPDPBUSD has a declared answer for a host without AVX-VNNI, so a new emitter
// fails until someone writes down which answer it takes. It greps the source
// rather than trusting a remembered site count; a missing check here once let
// the row-major GEMM SIGILL on a pre-VNNI host.
func TestEveryVPDPBUSDSiteIsCoveredOrRefused(t *testing.T) {
	// How each file answers "what happens on a CPU with no AVX-VNNI".
	const (
		covered = "emits the AVX2 sequence instead (dotEmitter, prevnni.go)"
		refused = "the emitter REFUSES under DotVEX, so the caller narrows or declines"
		decline = "cpu.SupportedNative is false for every quantized type, so nn declines " +
			"into the RULE 8 oracle and the kernel is never built"
	)
	answer := map[string]string{
		"prevnni.go": covered,
		// Emits its own int16-accumulating VEX sequence under DotVEX; see
		// EmitPackedMatMulStationary.
		"packedgemm.go": covered,

		// Both take a DotKind and return an error for DotVEX rather than
		// emitting a mixture. tiledFor narrows; the wide kernel has no caller.
		"packedtiled.go": refused,
		"packedwide.go":  refused,

		// The GGUF row-major family, which has no pre-VNNI form: a container
		// never decodes through it (native_amd64.go). engine/nn/jit.go and nn.MatMul
		// both gate on SupportedNative.
		"matvec.go": decline,
		"gemm.go":   decline,
		"gemmk.go":  decline,
	}

	// The emission calls, not the prose. Comments in this package name the
	// instruction constantly and none of them emits anything.
	site := regexp.MustCompile(`\.VPDPBUSD(Mem)?\(`)

	// This gate reads the package source, which does not travel with a test
	// binary shipped to a remote host. Skipping there costs no coverage (the
	// answer is host-independent); what it must not do is pass on finding no
	// sites.
	if _, err := os.Stat("prevnni.go"); err != nil {
		t.Skipf("NO PACKAGE SOURCE HERE (%v) -- this gate reads jit_old/cpu/*.go and a "+
			"shipped test binary carries none. It is not arch-dependent; run it "+
			"in the repo.", err)
	}
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]int{}
	for _, e := range ents {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(".", name))
		if err != nil {
			t.Fatal(err)
		}
		// Count only real emission calls, and only outside comments.
		n := 0
		for _, line := range strings.Split(string(b), "\n") {
			if i := strings.Index(line, "//"); i >= 0 {
				line = line[:i]
			}
			if site.MatchString(line) {
				n++
			}
		}
		if n == 0 {
			continue
		}
		seen[name] = n
		if _, ok := answer[name]; !ok {
			t.Errorf("%s emits VPDPBUSD at %d site(s) and has NO declared answer for a "+
				"host without AVX-VNNI. Every such file must either emit through "+
				"dotEmitter, REFUSE under DotVEX, or be declined by SupportedNative "+
				"before it is built -- pick one and record it in this gate's map. "+
				"A site with no answer is a SIGILL on the first token.", name, n)
		}
	}

	// The map must not outlive its files: a stale entry would pass for a file
	// that does not exist.
	for name := range answer {
		if seen[name] == 0 {
			t.Errorf("the map declares %s but it emits no VPDPBUSD -- delete the entry, "+
				"or this gate is covering a file that is gone", name)
		}
	}
	if len(seen) == 0 {
		t.Fatal("found no VPDPBUSD emission sites at all; the regexp or the cwd is wrong " +
			"and this gate proved nothing")
	}
	if t.Failed() {
		return // the summary below would contradict the errors above
	}
	t.Logf("%d files emit VPDPBUSD, all with a declared pre-VNNI answer: %v", len(seen), seen)
}
