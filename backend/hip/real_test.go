package hip

import (
	"os"
	"testing"
)

// rocmDir is the ROCm a test was pointed at (JITLLM_ROCM), or "" for the
// default search. Tests read it; the package does not.
func rocmDir() string { return os.Getenv("JITLLM_ROCM") }

// TestRealROCmLoadsOrSaysWhy loads whatever ROCm this host has and reports what
// it found. With no AMD device it must still be (nil, nil), never an error:
// that is the state of a ROCm install on a box whose GPU is not AMD's.
func TestRealROCmLoadsOrSaysWhy(t *testing.T) {
	c := Config{Path: rocmDir()}
	rt, err := Open(c)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if rt == nil {
		t.Logf("no HIP runtime: %s", Why(c))
	} else {
		t.Logf("HIP runtime with %d device(s)", rt.Count())
	}
	cg, err := LoadComgr(c)
	if err != nil {
		t.Skipf("NO COMGR: %v -- the code object gates did not run", err)
	}
	maj, min := cg.Version()
	t.Logf("comgr %d.%d, %d ISAs", maj, min, len(cg.ISAs()))
}
