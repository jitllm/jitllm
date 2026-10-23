package backend_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestKernelTimingTagsTrials: with CUDA's kernel timing armed, a launch in a
// session marked as a tuner's trial (backend.MarkTrial) is timed under
// TrialPrefix and an unmarked one under its own name, so a profile can read
// the work apart from the trials beside it. The unmarked launch is the
// violation: were the mark ignored, both would land under one name.
func TestKernelTimingTagsTrials(t *testing.T) {
	gpuLock(t)
	var d backend.Device
	for _, x := range backend.Open() {
		defer x.Close()
		if x.API() == "ptx" && d == nil {
			d = x
		}
	}
	if d == nil {
		t.Skip("no CUDA device: the instrument is CUDA's")
	}
	backend.SetCUDAKernelTiming(true)
	defer backend.SetCUDAKernelTiming(false)
	backend.CUDAKernelTimes()
	const n = 256
	kk, err := kernels.Add(n)
	if err != nil {
		t.Fatal(err)
	}
	k, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	g := newGPU(t, d)
	defer g.free()
	a, b, o := g.up(make([]byte, 4*n)), g.up(make([]byte, 4*n)), g.up(make([]byte, 4*n))
	for _, trial := range []bool{true, false} {
		d.Session(func(s backend.Session) {
			if trial {
				backend.MarkTrial(s)
			}
			if err := s.Launch(k, (n+127)/128, 128, a, b, o); err != nil {
				t.Fatal(err)
			}
			if err := s.Sync(); err != nil {
				t.Fatal(err)
			}
		})
	}
	us, _ := backend.CUDAKernelTimes()
	var tagged, plain int
	for name := range us {
		if strings.HasPrefix(name, backend.TrialPrefix) {
			tagged++
		} else {
			plain++
		}
	}
	if tagged != 1 || plain != 1 {
		t.Fatalf("timed names %v: want one under %q and one without", us, backend.TrialPrefix)
	}
}
