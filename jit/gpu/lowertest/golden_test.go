package lowertest

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/msl"
	"github.com/jitllm/jitllm/jit/ptx"
	"github.com/jitllm/jitllm/jit/spirv"
)

var update = flag.Bool("update", false, "rewrite testdata/kernels.sha")

// TestDenseKernelsUnchanged hashes every shipped kernel through all three
// lowerers and compares against a committed list.
//
// The expert path must be inert when Experts <= 1, and changes elsewhere once
// moved every dense kernel's emitted code without any correctness test
// failing (same arithmetic, different code). The hash answers the one question
// a correctness suite cannot: did a kernel I was not editing change?
//
// A legitimate kernel change should fail this. Rerun with -update and read the
// diff; a moved kernel you did not touch is the finding. Update the golden only
// after re-deriving the behaviour from something that is not the golden.
func TestDenseKernelsUnchanged(t *testing.T) {
	ks := all(t)
	names := make([]string, 0, len(ks))
	for n := range ks {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, n := range names {
		k := ks[n]
		p, err := ptx.Lower(k, "sm_86")
		if err != nil {
			t.Fatalf("%s ptx: %v", n, err)
		}
		m, err := msl.Emit(k)
		if err != nil {
			t.Fatalf("%s msl: %v", n, err)
		}
		v, err := spirv.Emit(k)
		if err != nil {
			t.Fatalf("%s spirv: %v", n, err)
		}
		h := sha256.New()
		h.Write([]byte(p))
		h.Write([]byte(m))
		h.Write(v)
		fmt.Fprintf(&b, "%s %x\n", n, h.Sum(nil)[:8])
	}

	path := filepath.Join("testdata", "kernels.sha")
	if *update {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Log("wrote", path)
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (rerun with -update to create it)", err)
	}
	if string(want) == b.String() {
		return
	}
	// Report per kernel, so the message names what moved rather than saying
	// "the hash differs".
	old := map[string]string{}
	for _, l := range strings.Split(strings.TrimSpace(string(want)), "\n") {
		if f := strings.Fields(l); len(f) == 2 {
			old[f[0]] = f[1]
		}
	}
	for _, l := range strings.Split(strings.TrimSpace(b.String()), "\n") {
		f := strings.Fields(l)
		if len(f) != 2 {
			continue
		}
		switch o, ok := old[f[0]]; {
		case !ok:
			t.Errorf("new kernel %s (rerun -update)", f[0])
		case o != f[1]:
			t.Errorf("kernel %s CHANGED %s -> %s", f[0], o, f[1])
		}
		delete(old, f[0])
	}
	for n := range old {
		t.Errorf("kernel %s disappeared", n)
	}
}
