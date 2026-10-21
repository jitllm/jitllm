package board

import (
	"math"
	"runtime"
	"slices"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/x86"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestEstimateMatchesTheHost holds the composer's estimate of a decode
// step on the host, weighed with the host as the probe measures it, to
// the step's composed kernels run in order on one buffer per operand: each
// call timed over many steps, its median beside its estimate. It runs the
// steps back to back, the steady state of decoding, and with reads of
// twice the caches between them, as a model larger than the caches leaves
// them, each against the estimate of that state. It reports each call's
// ratio and the step's, and the spread of the calls' ratios.
func TestEstimateMatchesTheHost(t *testing.T) {
	d, err := probe.Host()
	if err != nil {
		t.Skip(err)
	}
	f, err := jlm.Open(testmodels.Path(*boardModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for l, c := range d.Caches {
		t.Logf("L%d %d KiB %.1f GB/s %.1f ns, start %.1f ns, keeps %.2f, %.2f cycling", l+1, c.Bytes>>10, c.Rate.Thread/1e9, c.Latency*1e9, c.Start*1e9, c.Retain, c.Cycle)
	}
	t.Logf("memory %.1f GB/s %.1f ns, start %.1f ns", d.Thread/1e9, d.Latency*1e9, d.Start*1e9)
	t.Run("steady", func(t *testing.T) { estimateStep(t, d, f, true) })
	t.Run("cold", func(t *testing.T) { estimateStep(t, d, f, false) })
}

func estimateStep(t *testing.T, d jit.Device, f *jlm.File, steady bool) {
	s := jit.Step{Rows: 1, Page: 64, Keys: keys(t, *boardModel)}
	ccs, err := jit.Default().EstimateCodes(f, s, func(jit.Part) jit.Device { return d }, steady)
	if err != nil {
		t.Fatal(err)
	}
	var evict []byte
	if !steady {
		evict = make([]byte, 2*d.Caches[len(d.Caches)-1].Bytes)
		for i := range evict {
			evict[i] = byte(i)
		}
	}
	sink := byte(0)
	type call struct {
		code  *cpu.Code
		block []uint64
		name  string
		est   float64
		wait  float64
		cost  jit.Cost
		ops   []string
		times []float64
	}
	size := map[jit.Operand]int64{}
	for _, cc := range ccs {
		for _, a := range cc.Code.Args {
			if a.Words == nil {
				size[a.Operand] = max(size[a.Operand], int64(a.Offset)*4+cc.Reach[a.Field])
			}
		}
	}
	store := map[jit.Operand][]byte{}
	var calls []*call
	var keep [][]byte
	for _, cc := range ccs {
		if cc.Code.CPU == nil {
			continue
		}
		c, err := cpu.Map(cc.Code.CPU)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		block := make([]uint64, 64)
		for _, a := range cc.Code.Args {
			if v, ok := cc.Values[a.Field]; ok {
				block[a.Field/8] = v
				continue
			}
			if a.Value || a.Word || a.Length {
				t.Fatalf("%s: argument word %d has no value at the step", cc.Code.Name, a.Field)
			}
			buf, ok := store[a.Operand]
			if a.Words != nil || !ok {
				buf = fill(a, max(size[a.Operand], int64(len(a.Words))*4), s)
				keep = append(keep, buf)
			}
			if a.Words == nil {
				store[a.Operand] = buf
			}
			block[a.Field/8] = uint64(uintptr(unsafe.Pointer(&buf[0])))
		}
		calls = append(calls, &call{code: c, block: block, name: cc.Code.Name, est: cc.Cost.Seconds, wait: cc.Cost.WaitSeconds, cost: cc.Cost, ops: names(cc.Code)})
	}
	if len(calls) == 0 {
		t.Fatal("no host call in the step")
	}
	var empty x86.Buf
	empty.RET()
	code, err := empty.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	ret, err := cpu.Map(code)
	if err != nil {
		t.Fatal(err)
	}
	defer ret.Close()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var floor []float64
	var block [8]uint64
	for range 10000 {
		t0 := time.Now()
		ret.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
		floor = append(floor, time.Since(t0).Seconds())
	}
	slices.Sort(floor)
	clock := floor[len(floor)/2] - d.Launch
	t.Logf("the clock reads add %.1f ns to a call; taken off each", clock*1e9)
	for step := range 400 {
		for i := 0; i < len(evict); i += 64 {
			sink += evict[i]
		}
		for _, c := range calls {
			t0 := time.Now()
			c.code.Call((*cpu.Args)(unsafe.Pointer(&c.block[0])))
			if step >= 100 {
				c.times = append(c.times, time.Since(t0).Seconds())
			}
		}
	}
	runtime.KeepAlive(keep)
	runtime.KeepAlive(sink)
	var est, got float64
	var logs []float64
	for i, c := range calls {
		slices.Sort(c.times)
		m := max(c.times[len(c.times)/2]-clock, 1e-9)
		est += c.est
		got += m
		r := c.est / m
		logs = append(logs, math.Abs(math.Log2(r)))
		t.Logf("%2d %-26s estimate %8.3f us (wait %6.3f fetch %6.3f) measured %8.3f us ratio %.2f served %v %v", i, c.name, c.est*1e6, c.wait*1e6, c.cost.FetchSeconds*1e6, m*1e6, r, c.cost.Served, c.ops)
	}
	slices.Sort(logs)
	t.Logf("step: estimate %.1f us measured %.1f us ratio %.2f; calls within 2x: %d of %d, median error %.2fx",
		est*1e6, got*1e6, est/got, countBelow(logs, 1), len(logs), math.Exp2(logs[len(logs)/2]))
}

func names(c jit.Code) []string {
	var out []string
	for _, b := range c.Binds {
		w := ""
		if b.Writes {
			w = "*"
		}
		out = append(out, b.Operand.Name+w)
	}
	return out
}

func countBelow(xs []float64, v float64) int {
	n := 0
	for _, x := range xs {
		if x <= v {
			n++
		}
	}
	return n
}

// fill is the storage argument word a binds at step s: n bytes and a page
// more, holding the step's descriptors, page table
// and constants where it binds those, small floats else.
func fill(a jit.CallArg, n int64, s jit.Step) []byte {
	n += 4096
	buf := make([]byte, n)
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&buf[0])), n/4)
	switch {
	case a.Words != nil:
		copy(words, a.Words)
	case a.Operand.Name == "rows":
		for r := 0; r+3 < len(words); r += 4 {
			words[r], words[r+1], words[r+2], words[r+3] = 0, 0, uint32(s.Keys), uint32(max(0, s.Keys-1))
		}
	case a.Operand.Name == "pages":
		for i := range words {
			words[i] = uint32(i)
		}
	default:
		for i := range words {
			words[i] = math.Float32bits(0.5)
		}
	}
	return buf
}
