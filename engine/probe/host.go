//go:build amd64 || arm64

package probe

import (
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/cpu"
)

// The host probe's measuring kernels are generated per architecture: each
// provides hostCheck (whether this host runs them), hostLanes (a vector's
// floats), and readKernel, chaseKernel, rateKernel, chainKernel, mixKernel
// and emptyKernel, with one contract for the argument block they read.

// cache is a data cache level: its size and whether CPUs share it.
type cache struct {
	bytes  int64
	shared bool
}

// Host measures the host CPU: its caches' sizes from the operating system,
// then each level's read bandwidth on one P-core and on every P-core, each
// class's instruction rate, and an empty call's cost.
func Host() (cost.Device, error) {
	if err := hostCheck(); err != nil {
		return cost.Device{}, err
	}
	workers := max(1, len(sched.PCores()))
	d := cost.Device{Device: jit.Device{Kind: jit.DeviceCPU, Slots: workers, Lanes: hostLanes()}}
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
		d.Caches[l] = cost.Cache{Bytes: c.bytes, Rate: cost.Rate{Thread: rate, Total: readRate(read, all, workers)}, Latency: lat}
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
	for cl := cost.Class(0); cl < cost.NumClasses; cl++ {
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
		d.Rates[cl] = cost.Rate{Thread: issueRate(k, 8, 1), Total: issueRate(k, 8, workers)}
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
	d.Issue = cost.Rate{Thread: issueRate(mix, 24, 1), Total: issueRate(mix, 24, workers)}
	mix.Close()
	d.Delay[cost.ClassMem] = d.Caches[0].Latency
	if cl := cost.ClassMem; d.Rates[cl].Thread == 0 && d.Caches[0].Bytes > 0 {
		d.Rates[cl] = cost.Rate{Thread: d.Caches[0].Rate.Thread / 32, Total: d.Caches[0].Rate.Total / 32}
	}
	code, err = emptyKernel()
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

// retention is the fraction of a buffer an eighth of cache c's size that
// c still serves after reads of twice its size: the buffer read, the reads
// made, the buffer read again and timed, its time placed between all of it
// served by c and all of it by the level beyond, which serves at rate
// after start.
func retention(k *cpu.Code, c cost.Cache, rate, start float64, evict []byte) float64 {
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
func cycling(k *cpu.Code, c cost.Cache, rate float64, evict []byte) float64 {
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
// each on an OS thread of its own, after one round that warms them; for one
// thread the fastest round, since an operating system may place a lone
// thread on a slower core for some rounds (macOS does, at default
// priority), and a thread's capability is its fastest placement.
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
	if threads == 1 {
		return times[0]
	}
	return times[len(times)/2]
}
