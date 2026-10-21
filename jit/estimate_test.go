package jit

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

func TestEveryOpHasAName(t *testing.T) {
	for op := Op(0); op < numOps; op++ {
		if opNames[op] == "" {
			t.Errorf("op %d has no name", op)
		}
	}
}

// estimateRun is each kernel's estimate when the whole model runs on d at
// step s, in order.
func estimateRun(t *testing.T, d Device, s Step) ([]*Kernel, []Cost) {
	t.Helper()
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	parts, err := front(f)
	if err != nil {
		t.Fatal(err)
	}
	var run []PartPlan
	for _, p := range parts {
		run = append(run, PartPlan{Part: p, Device: d})
	}
	ks, err := Default().passRun(f, run, 0, s)
	if err != nil {
		t.Fatal(err)
	}
	m := costModel{d: d, s: s}
	var cs []Cost
	for _, k := range ks {
		cs = append(cs, m.cost(k))
	}
	return ks, cs
}

var (
	testCPU = Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, Bandwidth: 40e9, Thread: 15e9, Launch: 50e-9,
		Caches: [3]Cache{{Bytes: 32 << 10, Rate: Rate{Thread: 100e9}}, {Bytes: 1 << 20, Rate: Rate{Thread: 50e9}}, {Bytes: 12 << 20, Rate: Rate{Thread: 30e9}}},
		Rates:  [NumClasses]Rate{ClassFloat: {Thread: 8e9}}}
	testGPU = Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32, Bandwidth: 800e9, Thread: 2e7, Launch: 4e-6,
		Caches: [3]Cache{{Bytes: 6 << 20, Rate: Rate{Total: 2.5e12}}}, Rates: [NumClasses]Rate{ClassFloat: {Total: 14e12}}}
)

// TestEstimateDecidesEveryTrip holds the estimate to deciding every loop's
// trips from the step on both devices.
func TestEstimateDecidesEveryTrip(t *testing.T) {
	for _, d := range []Device{testCPU, testGPU} {
		_, cs := estimateRun(t, d, Step{Rows: 1, Page: 64, Keys: 5})
		for i, c := range cs {
			if c.Unknown > 0 {
				t.Errorf("%v kernel %d: %d loops whose trips the step does not decide", d.Kind, i, c.Unknown)
			}
		}
	}
}

// TestEstimateGrowsWithTheHistory holds the bytes a step moves to growing
// by the same amount for each equal step of the history's length, and
// growing at all.
func TestEstimateGrowsWithTheHistory(t *testing.T) {
	for _, d := range []Device{testCPU, testGPU} {
		var bytes []int64
		for _, keys := range []int{8, 16, 24} {
			_, cs := estimateRun(t, d, Step{Rows: 1, Page: 64, Keys: keys})
			var n int64
			for _, c := range cs {
				n += c.Bytes
			}
			bytes = append(bytes, n)
		}
		if a, b := bytes[1]-bytes[0], bytes[2]-bytes[1]; a <= 0 || a != b {
			t.Errorf("%v: bytes at 8, 16, 24 positions %v: steps %d and %d", d.Kind, bytes, a, b)
		}
		t.Logf("%v: bytes at 8, 16, 24 positions %v", d.Kind, bytes)
	}
}

// TestEstimateServesReuseFromCache holds a step's later reads of what it
// wrote to its caches where the device has them, and to its memory where
// it has none.
func TestEstimateServesReuseFromCache(t *testing.T) {
	s := Step{Rows: 1, Page: 64, Keys: 5}
	_, cached := estimateRun(t, testCPU, s)
	bare := testCPU
	bare.Caches = [3]Cache{}
	_, none := estimateRun(t, bare, s)
	var hit, miss, total int64
	for i := range cached {
		hit += cached[i].Served[0] + cached[i].Served[1] + cached[i].Served[2]
		miss += none[i].Served[0] + none[i].Served[1] + none[i].Served[2]
		total += cached[i].Bytes
	}
	if hit == 0 || miss != 0 {
		t.Errorf("served from caches: %d with caches, %d without", hit, miss)
	}
	t.Logf("%d bytes from caches of %d unique", hit, total)
}

// TestEstimateReadsTheHeadOnce holds the kernel that reads the model's
// largest weight, the head's projection, to moving that weight once: its
// bytes at least the weight's and no more than the weight's, the logits it
// writes and the row it reads.
func TestEstimateReadsTheHeadOnce(t *testing.T) {
	for _, d := range []Device{testCPU, testGPU} {
		ks, cs := estimateRun(t, d, Step{Rows: 1, Page: 64, Keys: 5})
		var weight, out, got int64
		for i, k := range ks {
			for _, b := range k.Bufs {
				if b.Operand.Kind != OperandWeight || b.Shape == nil {
					continue
				}
				n := int64(elemBytes(b))
				for _, x := range b.Shape {
					n *= int64(x)
				}
				if n > weight {
					weight, got = n, cs[i].Bytes
					out = int64(4 * b.Shape[0])
				}
			}
		}
		if got < weight || got > weight+out+4*1024 {
			t.Errorf("%v: the head's kernel moves %d bytes for a weight of %d", d.Kind, got, weight)
		}
		t.Logf("%v: the head's kernel moves %d bytes, its weight %d", d.Kind, got, weight)
	}
}
