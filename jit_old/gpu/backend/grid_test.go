package backend_test

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGridBarrierHoldsEveryWorkgroup runs mainkernels.GridProbe on every device
// that states a co-residency limit, at two workgroups, at half the limit and
// at the limit itself (every workgroup resident, the case a barrier deadlocks
// or leaks in), five launches in a row on one sync buffer (each launch leaves
// it valid for the next), and inside a recorded graph where the device records. Every
// output word is held to the host's answer; the outputs start poisoned, so a
// workgroup that crossed a barrier early reads a word not yet written and the
// poison shows.
//
// Its violation is TestGridBarrierProbeSeesAnEarlyCrossing, in a jitllmtest
// build. A launch one workgroup past the limit must be refused before it
// reaches the device.
func TestGridBarrierHoldsEveryWorkgroup(t *testing.T) {
	ran := 0
	eachGridDevice(t, func(d backend.Device, width, limit int) {
		for _, groups := range dedupe(2, limit/2, limit) {
			t.Run(fmt.Sprintf("%s/w%d/g%d", d.API(), width, groups), func(t *testing.T) {
				gridProbeCase(t, d, groups, width, limit)
			})
			ran++
		}
	})
	if ran == 0 {
		t.Skip("no device states a co-residency limit")
	}
}

// eachGridDevice calls f for every device and workgroup width at which a grid
// barrier runs, with that width's co-residency limit.
func eachGridDevice(t *testing.T, f func(d backend.Device, width, limit int)) {
	t.Helper()
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, width := range []int{128, 256} {
			probe, err := d.Compile(mainkernels.GridProbe(2, width))
			if err != nil {
				t.Logf("%s: compile: %v", d.API(), err)
				continue
			}
			limit, err := backend.GridLimit(d, probe)
			probe.Close()
			if err != nil {
				t.Logf("%s: no grid barrier here: %v", d.API(), err)
				continue
			}
			t.Logf("%s %s: %d workgroups of %d resident", d.API(), d.Name(), limit, width)
			if limit < 2 {
				t.Fatalf("%s: a co-residency limit of %d", d.API(), limit)
			}
			f(d, width, limit)
		}
	}
}

func dedupe(xs ...int) []int {
	var out []int
	for _, x := range xs {
		if x >= 2 && (len(out) == 0 || out[len(out)-1] != x) {
			out = append(out, x)
		}
	}
	return out
}

// gridRig is one compiled probe and its buffers.
type gridRig struct {
	t             *testing.T
	k             backend.Kernel
	outs          []backend.Buf
	sync          backend.Buf
	bufs          []backend.Buf
	want          [][]uint32
	groups, width int
}

func newGridRig(t *testing.T, d backend.Device, groups, width int) *gridRig {
	return newGridRigOn(t, d, groups, width, nil)
}

// newGridRigOn is newGridRig on a sync buffer another rig owns, or its own when
// sync is nil.
func newGridRigOn(t *testing.T, d backend.Device, groups, width int, sync backend.Buf) *gridRig {
	k, err := d.Compile(mainkernels.GridProbe(groups, width))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(k.Close)
	n := groups * width
	alloc := func(words int) backend.Buf {
		b, err := d.Alloc(4 * words)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Free)
		return b
	}
	r := &gridRig{t: t, k: k, groups: groups, width: width, want: mainkernels.GridProbeWant(groups, width)}
	r.outs = make([]backend.Buf, mainkernels.GridProbePhases)
	for p := range r.outs {
		r.outs[p] = alloc(n)
	}
	sink := alloc(mainkernels.GridProbePhases * n)
	r.sync = sync
	if sync == nil {
		r.sync = alloc(ir.GridSyncWords)
		if err := r.sync.Write(wordBytes([]uint32{0, 0})); err != nil {
			t.Fatal(err)
		}
	}
	r.bufs = append(append([]backend.Buf{}, r.outs...), sink, r.sync)
	return r
}

// poison fills every output with a word no phase computes.
func (r *gridRig) poison() {
	w := wordBytes(poisonWords(r.groups*r.width, 0xDEADBEEF))
	for _, o := range r.outs {
		if err := o.Write(w); err != nil {
			r.t.Fatal(err)
		}
	}
}

// wrong counts the output words that differ from the host's answer.
func (r *gridRig) wrong() (int, string) {
	n := r.groups * r.width
	bad, first := 0, ""
	raw := make([]byte, 4*n)
	for p, o := range r.outs {
		if err := o.Read(raw); err != nil {
			r.t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			got := binary.LittleEndian.Uint32(raw[4*i:])
			if got != r.want[p][i] {
				if bad == 0 {
					first = fmt.Sprintf("phase %d word %d (workgroup %d): %#x, want %#x", p, i, i/r.width, got, r.want[p][i])
				}
				bad++
			}
		}
	}
	return bad, first
}

func gridProbeCase(t *testing.T, d backend.Device, groups, width, limit int) {
	r := newGridRig(t, d, groups, width)
	const reps = 5
	for rep := 0; rep < reps; rep++ {
		r.poison()
		if err := r.k.Launch(groups, width, r.bufs...); err != nil {
			t.Fatalf("launch %d: %v", rep, err)
		}
		if bad, first := r.wrong(); bad > 0 {
			t.Fatalf("launch %d: %d of %d words wrong; first %s", rep, bad, mainkernels.GridProbePhases*groups*width, first)
		}
	}
	// Two grids taking turns on one sync buffer, as a decode's persistent
	// kernels of different grids do: each barrier must leave the buffer right
	// for a launch of any grid.
	other := newGridRigOn(t, d, max(2, groups-1), width, r.sync)
	for rep := 0; rep < 3; rep++ {
		for _, x := range []*gridRig{other, r} {
			x.poison()
			if err := x.k.Launch(x.groups, width, x.bufs...); err != nil {
				t.Fatalf("turn %d, %d groups: %v", rep, x.groups, err)
			}
			if bad, first := x.wrong(); bad > 0 {
				t.Fatalf("turn %d, %d groups sharing a sync buffer with %d and %d: %d words wrong; first %s",
					rep, x.groups, groups, other.groups, bad, first)
			}
		}
	}

	// The same inside a session, recorded and replayed where the device
	// records (a CUDA graph holding the cooperative launch).
	r.poison()
	var serr error
	d.Session(func(s backend.Session) {
		if rec, ok := s.(backend.Recorder); ok {
			g, err := backend.Record(rec, func() { serr = s.Launch(r.k, groups, width, r.bufs...) })
			if err != nil {
				serr = fmt.Errorf("record: %w", err)
				return
			}
			defer g.Free()
			if serr == nil {
				serr = rec.Replay(g)
			}
		} else {
			serr = s.Launch(r.k, groups, width, r.bufs...)
		}
		if serr == nil {
			serr = s.Sync()
		}
	})
	if serr != nil {
		t.Fatalf("session: %v", serr)
	}
	if bad, first := r.wrong(); bad > 0 {
		t.Fatalf("in a session: %d words wrong; first %s", bad, first)
	}

	// Past the limit, refused before the device.
	if groups == limit {
		big, err := d.Compile(mainkernels.GridProbe(limit+1, width))
		if err != nil {
			t.Fatal(err)
		}
		defer big.Close()
		if err := big.Launch(limit+1, width, r.bufs...); err == nil {
			t.Fatalf("a launch of %d workgroups past a limit of %d was accepted", limit+1, limit)
		} else {
			t.Logf("past the limit: %v", err)
		}
	}
}
