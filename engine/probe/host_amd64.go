// Package probe initialises the machine's devices and measures what the
// composer's estimator weighs: each memory level's bandwidth for one thread
// and for all, the rate of each class of instruction, and what a call or a
// launch costs. It runs measuring kernels only; the estimator never runs a
// model's code.
package probe

import (
	"fmt"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/x86"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// rounds is how many timed runs a measurement takes the median of, after
// one run that warms it.
const rounds = 7

// Host measures the host CPU: its caches' sizes from the operating system,
// then each level's read bandwidth on one P-core and on every P-core, each
// class's instruction rate, and an empty call's cost.
func Host() (jit.Device, error) {
	workers := max(1, len(sched.PCores()))
	d := jit.Device{Kind: jit.DeviceCPU, Slots: workers, Lanes: 8}
	sizes, err := cacheSizes()
	if err != nil {
		return d, err
	}
	code, err := readKernel()
	if err != nil {
		return d, err
	}
	read, err := cpu.Map(code)
	if err != nil {
		return d, err
	}
	defer read.Close()
	for l, c := range sizes {
		if l >= len(d.Caches) {
			break
		}
		all := c.bytes / 2
		if c.shared {
			all /= int64(workers)
		}
		d.Caches[l] = jit.Cache{Bytes: c.bytes, Rate: jit.Rate{Thread: readRate(read, c.bytes/2, 1), Total: readRate(read, all, workers)}}
	}
	const dram = 512 << 20
	d.Thread = readRate(read, dram/int64(workers), 1)
	d.Bandwidth = readRate(read, dram/int64(workers), workers)
	for cl := jit.Class(0); cl < jit.NumClasses; cl++ {
		code, err := rateKernel(cl)
		if err != nil {
			return d, err
		}
		if code == nil {
			continue
		}
		k, err := cpu.Map(code)
		if err != nil {
			return d, err
		}
		d.Rates[cl] = jit.Rate{Thread: issueRate(k, 1), Total: issueRate(k, workers)}
		k.Close()
	}
	if cl := jit.ClassMem; d.Rates[cl].Thread == 0 && d.Caches[0].Bytes > 0 {
		d.Rates[cl] = jit.Rate{Thread: d.Caches[0].Rate.Thread / 32, Total: d.Caches[0].Rate.Total / 32}
	}
	var empty x86.Buf
	empty.RET()
	code, err = empty.Bytes()
	if err != nil {
		return d, err
	}
	call, err := cpu.Map(code)
	if err != nil {
		return d, err
	}
	defer call.Close()
	d.Launch = callCost(call)
	return d, nil
}

// cache is a data cache level: its size and whether CPUs share it.
type cache struct {
	bytes  int64
	shared bool
}

// cacheSizes is the data caches, nearest first, as Linux states them for
// the first CPU: a level is shared where more than one CPU's threads that
// are not the first's siblings share it -- a list longer than two CPUs.
func cacheSizes() ([]cache, error) {
	var out []cache
	for i := 0; ; i++ {
		dir := fmt.Sprintf("/sys/devices/system/cpu/cpu0/cache/index%d/", i)
		typ, err := os.ReadFile(dir + "type")
		if err != nil {
			break
		}
		if strings.TrimSpace(string(typ)) == "Instruction" {
			continue
		}
		b, err := os.ReadFile(dir + "size")
		if err != nil {
			return nil, err
		}
		n, err := parseSize(strings.TrimSpace(string(b)))
		if err != nil {
			return nil, err
		}
		list, err := os.ReadFile(dir + "shared_cpu_list")
		if err != nil {
			return nil, err
		}
		out = append(out, cache{bytes: n, shared: cpuCount(strings.TrimSpace(string(list))) > 2})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("probe: no cache sizes under /sys/devices/system/cpu/cpu0/cache")
	}
	return out, nil
}

// cpuCount is how many CPUs a list like "0-1,4" names.
func cpuCount(list string) int {
	n := 0
	for _, part := range strings.Split(list, ",") {
		lo, hi, ok := strings.Cut(part, "-")
		if !ok {
			n++
			continue
		}
		a, errA := strconv.Atoi(lo)
		b, errB := strconv.Atoi(hi)
		if errA == nil && errB == nil {
			n += b - a + 1
		}
	}
	return n
}

func parseSize(s string) (int64, error) {
	mul := int64(1)
	switch {
	case strings.HasSuffix(s, "K"):
		mul, s = 1<<10, strings.TrimSuffix(s, "K")
	case strings.HasSuffix(s, "M"):
		mul, s = 1<<20, strings.TrimSuffix(s, "M")
	}
	n, err := strconv.ParseInt(s, 10, 64)
	return n * mul, err
}

// readKernel sums words [0] for [1] vectors of 256 bytes: eight loads a
// trip into eight accumulators, so the loads, not the adds, bind it.
func readKernel() ([]byte, error) {
	var a x86.Buf
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 0))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 8))
	for r := range 8 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for r := range 8 {
		a.VADDPSMem(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.At(x86.RSI, int32(32*r)))
	}
	a.ADDimm(x86.RSI, 256)
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// rateKernel runs [0] trips of eight independent instructions of class cl
// on registers, nil for a class with no instruction measured alone. The
// index class adds to four of the call-clobbered general registers.
func rateKernel(cl jit.Class) ([]byte, error) {
	var a x86.Buf
	var op func(r x86.Reg)
	switch cl {
	case jit.ClassFloat:
		op = func(r x86.Reg) { a.VFMADD231PS(r, x86.Y14, x86.Y15) }
	case jit.ClassSpecial:
		op = func(r x86.Reg) { a.VDIVPS(r, x86.Y14, x86.Y15) }
	case jit.ClassInt:
		op = func(r x86.Reg) { a.VPADDD(r, r, x86.Y15) }
	case jit.ClassLanes:
		op = func(r x86.Reg) { a.VPERMILPS(r, r, 0xB1) }
	case jit.ClassIndex:
		regs := [4]x86.Reg{x86.R8, x86.R9, x86.R10, x86.R11}
		op = func(r x86.Reg) { a.ADDimm(regs[(r-x86.Y0)%4], 1) }
	default:
		return nil, nil
	}
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for r := range 8 {
		op(x86.Y0 + x86.Reg(r))
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// readRate is the bytes a second threads reading set bytes each, over and
// over, get together: the median of the timed rounds.
func readRate(k *cpu.Code, set int64, threads int) float64 {
	set = max(256, set&^255)
	reps := max(1, int((256<<20)/set))
	bufs := make([][]byte, threads)
	for t := range bufs {
		bufs[t] = make([]byte, set)
		for i := range bufs[t] {
			bufs[t][i] = byte(i)
		}
	}
	run := func(t int) {
		block := [2]uint64{uint64(uintptr(unsafe.Pointer(&bufs[t][0]))), uint64(set / 256)}
		for range reps {
			k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
		}
		runtime.KeepAlive(bufs[t])
	}
	secs := parallel(threads, run)
	return float64(set) * float64(reps) * float64(threads) / secs
}

// issueRate is the instructions a second threads issue together.
func issueRate(k *cpu.Code, threads int) float64 {
	const trips = 1 << 22
	run := func(int) {
		block := [1]uint64{trips}
		k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
	}
	return 8 * trips * float64(threads) / parallel(threads, run)
}

// callCost is the seconds an empty call costs.
func callCost(k *cpu.Code) float64 {
	const calls = 1 << 20
	var block [8]uint64
	return parallel(1, func(int) {
		for range calls {
			k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
		}
	}) / calls
}

// parallel is the median seconds of rounds of threads running run at once,
// each on an OS thread of its own, after one round that warms them.
func parallel(threads int, run func(t int)) float64 {
	var times []float64
	for r := range rounds + 1 {
		var wg sync.WaitGroup
		start := make(chan struct{})
		for t := range threads {
			wg.Add(1)
			go func() {
				defer wg.Done()
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				<-start
				run(t)
			}()
		}
		t0 := time.Now()
		close(start)
		wg.Wait()
		if r > 0 {
			times = append(times, time.Since(t0).Seconds())
		}
	}
	slices.Sort(times)
	return times[len(times)/2]
}
