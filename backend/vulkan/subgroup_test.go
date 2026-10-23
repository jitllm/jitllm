package vulkan

import (
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit/spirv"
)

// shuffleModule is the smallest kernel that needs a 32-lane subgroup: one
// butterfly step, which is what makes it a subject of the guarantee rather than
// of the arithmetic.
func shuffleModule() ([]byte, error) {
	b := ir.New("k", [3]int{32, 1, 1})
	p := b.Param("p", ir.F32)
	q := b.Param("q", ir.F32)
	v := b.Load(ir.F32, p, b.TID(), 0)
	b.Store(q, b.TID(), b.Add(ir.F32, v, b.ShuffleXor(ir.F32, v, 16)), 0)
	k, err := b.Done()
	if err != nil {
		return nil, err
	}
	return spirv.Emit(k)
}

// TestSubgroupGuaranteeIsTheRule states the selection rule as a table, at the
// three kinds of device, in the three modes the engine can run in.
//
// It is a unit test so the rule is checkable where the hardware is not (the
// min 8, max 32 device is rare). The live half is TestSubgroupsMatchTheDevice.
func TestSubgroupGuaranteeIsTheRule(t *testing.T) {
	const shuf = subgroupBasic | subgroupShuffle
	nvidia := Subgroups{Size: 32, Stages: stageCompute, Ops: shuf, Min: 32, Max: 32,
		ReqStages: stageCompute, Control: true, Enabled: true}
	iris := Subgroups{Size: 32, Stages: stageCompute, Ops: shuf, Min: 8, Max: 32,
		ReqStages: stageCompute, Control: true, Enabled: true}
	llvmpipe := Subgroups{Size: 8, Stages: stageCompute, Ops: shuf, Min: 8, Max: 8,
		ReqStages: stageCompute, Control: true, Enabled: true}

	off := func(s Subgroups) Subgroups { s.Off, s.Enabled = true, false; return s }
	forced := func(s Subgroups) Subgroups { s.Forced, s.Enabled = true, false; return s }
	noExt := func(s Subgroups) Subgroups {
		s.Control, s.Enabled, s.Min, s.Max, s.ReqStages = false, false, 0, 0, 0
		return s
	}
	noShuffle := func(s Subgroups) Subgroups { s.Ops = subgroupBasic; return s }
	noComputeStage := func(s Subgroups) Subgroups { s.ReqStages = 0; return s }

	for _, c := range []struct {
		name string
		s    Subgroups
		want bool
	}{
		// The shipping configuration.
		{"nvidia 32/32, pinned", nvidia, true},
		{"iris 8..32, pinned", iris, true},
		{"llvmpipe 8/8, cannot reach 32", llvmpipe, false},

		// With the required-size path off, the promise must come from
		// min == max alone, which separates the two devices.
		{"nvidia 32/32, no pinning", off(nvidia), true},
		{"iris 8..32, no pinning", off(iris), false},
		{"llvmpipe 8/8, no pinning", off(llvmpipe), false},

		// The measurement override that reproduces the bug on purpose.
		{"iris 8..32, forced", forced(iris), true},
		{"llvmpipe 8/8, forced", forced(llvmpipe), true},

		// subgroupSize is not a promise: both of these report 32 and neither
		// may be believed.
		{"nvidia without the extension", noExt(nvidia), false},
		{"iris without the extension", noExt(iris), false},

		// A device with the right width and no shuffle at all, and one that will
		// not pin the compute stage in particular.
		{"iris, no shuffle support", noShuffle(iris), false},
		{"iris, compute stage not pinnable", noComputeStage(iris), false},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, why := c.s.Guarantees(32)
			if got != c.want {
				t.Fatalf("Guarantees(32) = %v (%s), want %v -- %s", got, why, c.want, c.s)
			}
			if !got && why == "" {
				t.Fatal("a refusal with no reason: the kernel it refuses has no way to report why")
			}
			t.Logf("%v: %s", got, why)
		})
	}

	// A kernel that needs no width runs anywhere, including on the device that
	// promises nothing -- otherwise every kernel would need the extension.
	if ok, _ := noExt(iris).Guarantees(0); !ok {
		t.Error("a kernel with no width requirement was refused")
	}
	// And a width nobody can offer is refused even where pinning works.
	if ok, _ := nvidia.Guarantees(64); ok {
		t.Error("32/32 accepted a request for 64 lanes")
	}
}

// TestSubgroupsMatchTheDevice is the live half: every Vulkan device on this host,
// opened, with what it promises checked against what it reports.
//
// It checks that the numbers come out of the driver as expected, that enabling
// the extension produced a device that pins, and that a shuffling pipeline is
// created exactly when the promise says it can be.
func TestSubgroupsMatchTheDevice(t *testing.T) {
	infos, err := List()
	if err != nil {
		t.Skipf("no Vulkan on this host: %v", err)
	}
	for _, in := range infos {
		if !in.Compute {
			continue
		}
		t.Run(strconv.Itoa(in.Index), func(t *testing.T) {
			c, err := OpenDevice(strconv.Itoa(in.Index))
			if err != nil {
				t.Skipf("device %d would not open: %v", in.Index, err)
			}
			defer c.Close()
			s := c.Subgroups()
			ok, why := s.Guarantees(32)
			t.Logf("%-45s %s", c.Name(), s)
			t.Logf("  32 lanes: %v %s", ok, why)

			if s.Size == 0 {
				t.Fatal("the device reported no subgroup size at all; the query is not landing")
			}
			if s.Control && s.Max < s.Min {
				t.Fatalf("min %d > max %d: the two fields are swapped", s.Min, s.Max)
			}
			if s.Control && !s.Enabled && !s.Off && !s.Forced {
				t.Error("the device advertises the extension and it was not enabled, so a " +
					"width that could have been pinned will be refused")
			}
			// The promise and the pipeline have to agree, in both directions:
			// a promise that will not compile is a lie, and a refusal that
			// compiles anyway means selection is back to being a hope.
			k, err := shuffleModule()
			if err != nil {
				t.Fatal(err)
			}
			kern, cerr := c.Compile(k, "k", 2, 32)
			switch {
			case ok && cerr != nil:
				t.Fatalf("promised 32 lanes and would not create the pipeline: %v", cerr)
			case !ok && cerr == nil:
				kern.Close()
				t.Fatalf("refused to promise 32 lanes (%s) and created the pipeline anyway", why)
			case cerr == nil:
				kern.Close()
			}
		})
	}
}
