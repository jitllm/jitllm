package board

import (
	"fmt"
	"math"
	"slices"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest"
)

// TestEstimateMatchesTheGPU holds the composer's estimate of a decode step
// on the first CUDA device, weighed with the device as the probe measures
// it, to the step's composed kernels launched in order in one session on
// one buffer per storage the estimate assumed, each kernel's time read from
// the device's clock: the median of rounds' means over many steps. It
// reports each kernel's ratio and the step's, and the spread of the
// kernels' ratios.
func TestEstimateMatchesTheGPU(t *testing.T) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	var dev backend.Device
	for _, d := range devs {
		if d.API() == "ptx" && d.Slots() > 0 {
			dev = d
			break
		}
	}
	if dev == nil {
		t.Skip("no CUDA device")
	}
	d, err := probe.GPU(dev)
	if err != nil {
		t.Fatal(err)
	}
	c := d.Caches[1]
	t.Logf("L1 %d KiB %.0f ns, %.1f loads in flight", d.Caches[0].Bytes>>10, d.Caches[0].Latency*1e9, d.Overlap)
	t.Logf("%s: %d slots, launch %.2f us, floor %.2f us; L2 %d KiB %.0f GB/s %.0f ns; memory %.0f GB/s %.0f ns; float %.0f G/s",
		dev.Name(), d.Slots, d.Launch*1e6, d.Floor*1e6, c.Bytes>>10, c.Rate.Total/1e9, c.Latency*1e9, d.Bandwidth/1e9, d.Latency*1e9, d.Rates[jit.ClassFloat].Total/1e9)
	f, err := jlm.Open(testmodels.Path(*boardModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s := jit.Step{Rows: 1, Page: 64, Keys: keys(t, *boardModel)}
	ccs, err := jit.Default().EstimateCodes(f, s, func(jit.Part) jit.Device { return d }, true)
	if err != nil {
		t.Fatal(err)
	}
	size := map[jit.Operand]int64{}
	for _, cc := range ccs {
		for _, b := range cc.Binds {
			size[b.Storage] = max(size[b.Storage], b.Reach)
		}
	}
	store := map[jit.Operand]backend.Buf{}
	type launch struct {
		k      backend.Kernel
		name   string
		groups int
		width  int
		bufs   []backend.Buf
		cost   jit.Cost
		at     int
	}
	var runs []launch
	for i, cc := range ccs {
		if cc.Code.GPU == nil {
			continue
		}
		k := paritytest.Old(cc.Code.GPU)
		k.Name = fmt.Sprintf("est%02d_%s", i, k.Name)
		kern, err := dev.Compile(k)
		if err != nil {
			t.Fatalf("%s: %v", k.Name, err)
		}
		defer kern.Close()
		var bufs []backend.Buf
		for j, b := range cc.Code.Binds {
			at := cc.Binds[j].Storage
			buf, ok := store[at]
			if words := cc.Binds[j].Words; words != nil || !ok {
				n := max(size[at], int64(4*len(words))) + 4096
				if buf, err = dev.Alloc(int(n)); err != nil {
					t.Fatal(err)
				}
				defer buf.Free()
				if err := buf.Write(deviceFill(b.Operand, words, n, s)); err != nil {
					t.Fatal(err)
				}
				if words == nil {
					store[at] = buf
				}
			}
			bufs = append(bufs, buf)
		}
		runs = append(runs, launch{k: kern, name: k.Name, groups: cc.Code.Groups, width: k.Group[0], bufs: bufs, cost: cc.Cost, at: i})
	}
	if len(runs) == 0 {
		t.Fatal("no device kernel in the step")
	}
	step := func() error {
		var err error
		dev.Session(func(se backend.Session) {
			for _, r := range runs {
				if err = se.Launch(r.k, r.groups, r.width, r.bufs...); err != nil {
					return
				}
			}
			err = se.Sync()
		})
		return err
	}
	for i, r := range runs {
		var err error
		dev.Session(func(se backend.Session) {
			if err = se.Launch(r.k, r.groups, r.width, r.bufs...); err == nil {
				err = se.Sync()
			}
		})
		if err != nil {
			cc := ccs[r.at]
			for j, b := range cc.Code.Binds {
				t.Logf("bind %d %s writes %v reach %d words %d size %d", j, b.Operand.Name, b.Writes, cc.Binds[j].Reach, len(cc.Binds[j].Words), size[b.Operand])
			}
			t.Fatalf("run %d %s groups %d width %d: %v", i, r.name, r.groups, r.width, err)
		}
	}
	for range 20 {
		if err := step(); err != nil {
			t.Fatal(err)
		}
	}
	backend.SetCUDAKernelTiming(true)
	backend.CUDAKernelTimes()
	const rounds, steps = 11, 20
	means := map[string][]float64{}
	for range rounds {
		for range steps {
			if err := step(); err != nil {
				backend.SetCUDAKernelTiming(false)
				t.Fatal(err)
			}
		}
		us, n := backend.CUDAKernelTimes()
		for name, x := range us {
			means[name] = append(means[name], x/float64(n[name]))
		}
	}
	backend.SetCUDAKernelTiming(false)
	var est, got float64
	var logs []float64
	for i, r := range runs {
		xs := means[r.name]
		if len(xs) < rounds {
			t.Fatalf("%s was timed in %d of %d rounds", r.name, len(xs), rounds)
		}
		slices.Sort(xs)
		m := xs[len(xs)/2] / 1e6
		c := r.cost
		e := max(c.MemSeconds, c.ComputeSeconds, c.WaitSeconds, c.ChainSeconds) + c.FetchSeconds + c.FloorSeconds
		est += e
		got += m
		ratio := e / m
		logs = append(logs, math.Abs(math.Log2(ratio)))
		t.Logf("%2d %-22s estimate %8.3f us (mem %7.3f compute %7.3f wait %6.3f chain %6.3f) measured %8.3f us ratio %.2f served %v",
			i, r.name, e*1e6, r.cost.MemSeconds*1e6, r.cost.ComputeSeconds*1e6, r.cost.WaitSeconds*1e6, r.cost.ChainSeconds*1e6, m*1e6, ratio, r.cost.Served)
	}
	slices.Sort(logs)
	t.Logf("step on the device's clock: estimate %.1f us measured %.1f us ratio %.2f; kernels within 2x: %d of %d, median error %.2fx",
		est*1e6, got*1e6, est/got, countBelow(logs, 1), len(logs), math.Exp2(logs[len(logs)/2]))
}

// deviceFill is n bytes of storage for operand o at step s: its constants,
// the step's descriptors and page table, zeros for another table of words
// (a decode row's offsets), or small floats.
func deviceFill(o jit.Operand, words []uint32, n int64, s jit.Step) []byte {
	buf := make([]byte, n)
	w := unsafe.Slice((*uint32)(unsafe.Pointer(&buf[0])), n/4)
	switch {
	case words != nil:
		copy(w, words)
	case o.Name == "rows":
		for r := 0; r+3 < len(w); r += 4 {
			w[r], w[r+1], w[r+2], w[r+3] = 0, 0, uint32(s.Keys), uint32(max(0, s.Keys-1))
		}
	case o.Name == "pages":
		for i := range w {
			w[i] = uint32(i)
		}
	case o.Elem == ir.U32:
	default:
		for i := range w {
			w[i] = math.Float32bits(0.5)
		}
	}
	return buf
}
