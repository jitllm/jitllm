package tier

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestCompileFailureIsCounted holds Stats.Failed to what it says: a matvec
// whose kernel the device will not compile is refused AND counted. It was
// summed across devices and never incremented, so `jitllm verify` could not
// tell a compile failure from a matvec that was never offered.
//
// The control arm is the same matvec on a device that compiles: it must be
// served with Failed at zero, or the failing arm proves nothing.
func TestCompileFailureIsCounted(t *testing.T) {
	const rows, k = 128, 256
	run := func(failCompile bool) (bool, Stats) {
		d := &fakeDev{}
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff), WithMinMatVecBytes(0))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		defer g.Close()
		// Armed after New, so only the matvec's own compile can fail.
		d.failCompile = failCompile
		w := make([]byte, rows*(k/32)*18) // Q4_0: 18 bytes per 32 weights
		x := make([]float32, k)
		out := make([]float32, rows)
		ok := g.MatVec(out, quant.Q4_0, w, x, rows, k)
		return ok, g.Stats()
	}

	ok, st := run(false)
	if !ok || st.Served != 1 || st.Failed != 0 {
		t.Fatalf("control: served=%v Served=%d Failed=%d (LastErr %q) -- the fake must serve this matvec "+
			"or the failing arm below proves nothing", ok, st.Served, st.Failed, st.LastErr)
	}

	ok, st = run(true)
	if ok {
		t.Fatal("a matvec whose kernel did not compile was reported as served")
	}
	if st.Failed == 0 {
		t.Fatalf("the kernel did not compile and Stats.Failed is 0 (NoKernel %d, NoRoom %d, TooSmall %d, "+
			"LastErr %q)", st.NoKernel, st.NoRoom, st.TooSmall, st.LastErr)
	}
	if st.LastErr == "" {
		t.Error("the compile failure left no LastErr to say why")
	}
	t.Logf("Failed=%d LastErr=%q", st.Failed, st.LastErr)
}
