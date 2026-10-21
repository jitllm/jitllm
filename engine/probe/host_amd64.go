package probe

import (
	"fmt"
	"math/rand/v2"
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
	chase, err := chaseKernel()
	if err != nil {
		return d, err
	}
	walk, err := cpu.Map(chase)
	if err != nil {
		return d, err
	}
	defer walk.Close()
	largest := sizes[len(sizes)-1].bytes
	evict := make([]byte, 2*largest)
	nearer := int64(0)
	for l, c := range sizes {
		if l >= len(d.Caches) {
			break
		}
		set := c.bytes / 2
		if nearer > 0 {
			set = min(set, 2*nearer)
		}
		all := set
		if c.shared {
			all = max(min(set, c.bytes/2/int64(workers)), min(set, 2*nearer))
		}
		lat, err := chaseLatency(walk, set)
		if err != nil {
			return d, err
		}
		rate := readRate(read, set, 1)
		d.Caches[l] = jit.Cache{Bytes: c.bytes, Rate: jit.Rate{Thread: rate, Total: readRate(read, all, workers)}, Latency: lat}
		if nearer > 0 {
			d.Caches[l].Start = streamStart(read, rate, evict[:2*nearer])
		}
		nearer = c.bytes
	}
	const dram = 512 << 20
	d.Thread = readRate(read, dram/int64(workers), 1)
	d.Bandwidth = readRate(read, dram/int64(workers), workers)
	d.Start = streamStart(read, d.Thread, evict)
	for l := 1; l < len(sizes) && l < len(d.Caches); l++ {
		c := d.Caches[l]
		rate, start := d.Thread, d.Start
		if l+1 < len(sizes) && l+1 < len(d.Caches) {
			rate, start = d.Caches[l+1].Rate.Thread, d.Caches[l+1].Start
		}
		d.Caches[l].Retain = retention(read, c, rate, start, evict)
		d.Caches[l].Cycle = cycling(read, c, rate, evict)
	}
	if d.Latency, err = chaseLatency(walk, dram); err != nil {
		return d, err
	}
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
		d.Rates[cl] = jit.Rate{Thread: issueRate(k, 8, 1), Total: issueRate(k, 8, workers)}
		k.Close()
		code, err = chainKernel(cl)
		if err != nil {
			return d, err
		}
		k, err = cpu.Map(code)
		if err != nil {
			return d, err
		}
		d.Delay[cl] = 1 / issueRate(k, 8, 1)
		k.Close()
	}
	code, err = mixKernel()
	if err != nil {
		return d, err
	}
	mix, err := cpu.Map(code)
	if err != nil {
		return d, err
	}
	d.Issue = jit.Rate{Thread: issueRate(mix, 24, 1), Total: issueRate(mix, 24, workers)}
	mix.Close()
	d.Delay[jit.ClassMem] = d.Caches[0].Latency
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

// chaseKernel follows [1] pointers from the address in [0], each read
// waiting for the one before.
func chaseKernel() ([]byte, error) {
	var a x86.Buf
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 0))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 8))
	top := a.Label()
	a.Bind(top)
	a.MOVLoad(x86.RSI, x86.At(x86.RSI, 0))
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.RET()
	return a.Bytes()
}

// chaseLatency is the seconds one read waits walking a random cycle
// through the cache lines of set bytes, so the prefetchers cannot run
// ahead. A cache's set is twice the level before it, past that level and
// within the TLB's reach, and the set is on huge pages where the kernel
// grants them, as the engine's frames ask, so the walk waits for the cache
// and not for page walks.
func chaseLatency(k *cpu.Code, set int64) (float64, error) {
	const line = 64
	lines := int(max(2, set/line))
	raw, free, err := hugeBuffer(lines * line)
	if err != nil {
		return 0, err
	}
	defer free()
	buf := unsafe.Slice((*uint64)(unsafe.Pointer(&raw[0])), len(raw)/8)
	order := rand.Perm(lines)
	for i, at := range order {
		next := order[(i+1)%lines]
		buf[at*line/8] = uint64(uintptr(unsafe.Pointer(&buf[next*line/8])))
	}
	steps := max(1<<20, 4*lines)
	block := [2]uint64{uint64(uintptr(unsafe.Pointer(&buf[order[0]*line/8]))), uint64(steps)}
	secs := parallel(1, func(int) {
		k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
	})
	return secs / float64(steps), nil
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

// chainKernel runs [0] trips of eight instructions of class cl, each
// using the one before's result, nil for a class rateKernel has none of.
// The index chain subtracts a register: a core may fold a chain of
// immediate adds in its renamer, which an index computed from registers
// does not get.
func chainKernel(cl jit.Class) ([]byte, error) {
	var a x86.Buf
	var op func()
	switch cl {
	case jit.ClassFloat:
		op = func() { a.VFMADD231PS(x86.Y0, x86.Y14, x86.Y15) }
	case jit.ClassSpecial:
		op = func() { a.VDIVPS(x86.Y0, x86.Y0, x86.Y15) }
	case jit.ClassInt:
		op = func() { a.VPADDD(x86.Y0, x86.Y0, x86.Y15) }
	case jit.ClassLanes:
		op = func() { a.VPERMILPS(x86.Y0, x86.Y0, 0xB1) }
	case jit.ClassIndex:
		op = func() { a.SUBQ(x86.R8, x86.R9) }
	default:
		return nil, nil
	}
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	top := a.Label()
	a.Bind(top)
	for range 8 {
		op()
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// mixKernel runs [0] trips of twenty-four instructions: eight float
// chains, as many as a float rate needs to hide its latency, and eight
// each of the index and memory classes, which take ports of their own, so
// that the rate is what the core renames a second rather than any class's
// units.
func mixKernel() ([]byte, error) {
	var a x86.Buf
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 0))
	for r := range 16 {
		a.VPXOR(x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r), x86.Y0+x86.Reg(r))
	}
	regs := [4]x86.Reg{x86.R8, x86.R9, x86.R10, x86.R11}
	top := a.Label()
	a.Bind(top)
	for j := range 8 {
		a.VFMADD231PS(x86.Y0+x86.Reg(j), x86.Y14, x86.Y15)
		a.ADDimm(regs[j%4], 1)
		a.VMOVDQULoad(x86.Y8+x86.Reg(j%4), x86.At(x86.RDI, int32(8*(j%4))))
	}
	a.DEC(x86.RCX)
	a.JNZ(top)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// retention is the fraction of a buffer an eighth of cache c's size that
// c still serves after reads of twice its size: the buffer read, the reads
// made, the buffer read again and timed, its time placed between all of it
// served by c and all of it by the level beyond, which serves at rate
// after start.
func retention(k *cpu.Code, c jit.Cache, rate, start float64, evict []byte) float64 {
	n := max(256, c.Bytes/8&^255)
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = byte(i)
	}
	at := [2]uint64{uint64(uintptr(unsafe.Pointer(&buf[0]))), uint64(n / 256)}
	out := [2]uint64{uint64(uintptr(unsafe.Pointer(&evict[0]))), uint64(min(int64(len(evict)), 2*c.Bytes) / 256)}
	clock := clockCost()
	var times []float64
	for range 4 * rounds {
		k.Call((*cpu.Args)(unsafe.Pointer(&at[0])))
		k.Call((*cpu.Args)(unsafe.Pointer(&out[0])))
		t0 := time.Now()
		k.Call((*cpu.Args)(unsafe.Pointer(&at[0])))
		times = append(times, time.Since(t0).Seconds()-clock)
	}
	runtime.KeepAlive(buf)
	slices.Sort(times)
	hit := float64(n)/c.Rate.Thread + c.Start
	miss := float64(n)/rate + start
	if miss <= hit {
		return 0
	}
	return min(1, max(0, (miss-times[len(times)/2])/(miss-hit)))
}

// cycling is the share of cache c's capacity a buffer twice its size,
// read over and over, finds in it: each pass's time placed between all of
// it served by c and all of it by the level beyond, at rate.
func cycling(k *cpu.Code, c jit.Cache, rate float64, evict []byte) float64 {
	n := min(int64(len(evict)), 2*c.Bytes) &^ 255
	at := [2]uint64{uint64(uintptr(unsafe.Pointer(&evict[0]))), uint64(n / 256)}
	var times []float64
	for range 2 * rounds {
		t0 := time.Now()
		k.Call((*cpu.Args)(unsafe.Pointer(&at[0])))
		times = append(times, time.Since(t0).Seconds())
	}
	runtime.KeepAlive(evict)
	slices.Sort(times)
	hit, miss := float64(n)/c.Rate.Thread, float64(n)/rate
	if miss <= hit {
		return 0
	}
	h := min(1, max(0, (miss-times[len(times)/2])/(miss-hit)))
	return h * float64(n) / float64(c.Bytes)
}

// clockCost is the seconds reading the clock twice adds to what it times.
func clockCost() float64 {
	var times []float64
	for range 1000 {
		t0 := time.Now()
		times = append(times, time.Since(t0).Seconds())
	}
	slices.Sort(times)
	return times[len(times)/2]
}

// streamStart is the seconds a short read from a level waits beyond its
// bytes at the level's rate: a 32 KB buffer read, then evict read so the
// buffer leaves every nearer level, then the buffer read again and timed,
// the prefetchers starting cold on it.
func streamStart(k *cpu.Code, rate float64, evict []byte) float64 {
	const n = 32 << 10
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = byte(i)
	}
	for i := range evict {
		evict[i] = byte(i)
	}
	at := [2]uint64{uint64(uintptr(unsafe.Pointer(&buf[0]))), n / 256}
	out := [2]uint64{uint64(uintptr(unsafe.Pointer(&evict[0]))), uint64(len(evict) / 256)}
	clock := clockCost()
	var times []float64
	for range 4 * rounds {
		k.Call((*cpu.Args)(unsafe.Pointer(&at[0])))
		k.Call((*cpu.Args)(unsafe.Pointer(&out[0])))
		t0 := time.Now()
		k.Call((*cpu.Args)(unsafe.Pointer(&at[0])))
		times = append(times, time.Since(t0).Seconds()-clock)
	}
	runtime.KeepAlive(buf)
	runtime.KeepAlive(evict)
	slices.Sort(times)
	return max(0, times[len(times)/2]-n/rate)
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

// issueRate is the instructions a second threads issue together, each trip
// perTrip of them; the block is wide enough for a kernel's loads to read.
func issueRate(k *cpu.Code, perTrip float64, threads int) float64 {
	const trips = 1 << 22
	run := func(int) {
		block := [8]uint64{trips}
		k.Call((*cpu.Args)(unsafe.Pointer(&block[0])))
	}
	return perTrip * trips * float64(threads) / parallel(threads, run)
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
