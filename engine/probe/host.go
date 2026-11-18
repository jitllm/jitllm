//go:build amd64 || arm64

package probe

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/sched"
)

// The host probe's measuring kernels are the composer's forms of graphs of
// ops, each gated against the hand-built kernel it replaced (hostforms.go):
// the reads, the chase, each class's rate and delay, the lanes class's an
// iteration's sums over pairs of a vector's lanes, the mix and a call, each
// rate the instructions of its class the estimator counts in a form over
// the seconds it took. Each architecture provides hostCheck (whether this
// host runs them) and hostLanes (a vector's floats) and formLanes (a
// counted vector's).

// cache is a data cache level: its size and whether CPUs share it.
type cache struct {
	bytes  int64
	shared bool
}

// Host measures the host CPU: its caches' sizes from the operating system,
// then each level's read bandwidth on one P-core and on every P-core, each
// class's instruction rate, and an empty call's cost.
func Host() (cost.Device, error) { return HostOn(cpu.New()) }

// HostOn is Host for the CPU h states: the SSE tier's kernels where h is at
// that tier.
func HostOn(h *cpu.Target) (cost.Device, error) {
	if err := hostCheck(h); err != nil {
		return cost.Device{}, err
	}
	e := emitter{sse: h.Tier() == cpu.TierSSE}
	workers := max(1, len(sched.PCores()))
	d := cost.Device{Device: composer.Device{Kind: composer.DeviceCPU, Slots: workers, Lanes: e.hostLanes(), ExactDiv: true}}
	sizes, err := cacheSizes()
	if err != nil {
		return d, err
	}
	fm := newFormer(h)
	read := &reader{fm: fm, forms: map[int]*formed{}}
	defer read.close()
	const dram = 512 << 20
	walk, err := newWalk(fm, dram)
	if err != nil {
		return d, err
	}
	defer walk.close()
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
		rate, err := readRate(read, set, 1)
		if err != nil {
			return d, err
		}
		total, err := readRate(read, all, workers)
		if err != nil {
			return d, err
		}
		d.Caches[l] = cost.Cache{Bytes: c.bytes, Rate: cost.Rate{Thread: rate, Total: total}, Latency: lat}
		if nearer > 0 {
			if d.Caches[l].Start, err = streamStart(read, rate, evict[:2*nearer]); err != nil {
				return d, err
			}
		}
		nearer = c.bytes
	}
	if d.Thread, err = readRate(read, dram/int64(workers), 1); err != nil {
		return d, err
	}
	if d.Bandwidth, err = readRate(read, dram/int64(workers), workers); err != nil {
		return d, err
	}
	if d.Start, err = streamStart(read, d.Thread, evict); err != nil {
		return d, err
	}
	for l := 1; l < len(sizes) && l < len(d.Caches); l++ {
		c := d.Caches[l]
		rate, start := d.Thread, d.Start
		if l+1 < len(sizes) && l+1 < len(d.Caches) {
			rate, start = d.Caches[l+1].Rate.Thread, d.Caches[l+1].Start
		}
		if d.Caches[l].Retain, err = retention(read, c, rate, start, evict); err != nil {
			return d, err
		}
		if d.Caches[l].Cycle, err = cycling(read, c, rate, evict); err != nil {
			return d, err
		}
	}
	if d.Latency, err = chaseLatency(walk, dram); err != nil {
		return d, err
	}
	for cl := cost.Class(0); cl < cost.NumClasses; cl++ {
		if cl == cost.ClassLanes {
			if d.Rates[cl], d.Delay[cl], err = lanesRate(fm, e.formLanes(), workers); err != nil {
				return d, err
			}
			continue
		}
		if d.Rates[cl], d.Delay[cl], err = classRate(fm, cl, e.formLanes(), workers); err != nil {
			return d, err
		}
	}
	if d.Issue, err = mixRate(fm, workers); err != nil {
		return d, err
	}
	d.Delay[cost.ClassMem] = d.Caches[0].Latency
	if cl := cost.ClassMem; d.Rates[cl].Thread == 0 && d.Caches[0].Bytes > 0 {
		d.Rates[cl] = cost.Rate{Thread: d.Caches[0].Rate.Thread / 32, Total: d.Caches[0].Rate.Total / 32}
	}
	if d.Launch, err = callCost(fm); err != nil {
		return d, err
	}
	return d, nil
}

// emitter emits the probe's kernels for a host: on amd64, the SSE tier's
// legacy-encoded 128-bit forms where sse is set.
type emitter struct{ sse bool }

// walk is the chase's two forms, chaseShort and chaseLong steps of
// chaseGraph over a table of up to words words, and its values.
type walk struct {
	forms         [2]*formed
	words         int
	at, tab, next graph.ValueID
}

// newWalk forms the chase over a table of up to bytes bytes.
func newWalk(fm former, bytes int64) (*walk, error) {
	w := &walk{words: int(bytes / 4)}
	for i, trips := range [2]int{chaseShort, chaseLong} {
		g, tab, at := chaseGraph(w.words, trips)
		f, err := fm.form(g, 1)
		if err != nil {
			w.close()
			return nil, err
		}
		w.forms[i], w.tab, w.at, w.next = f, tab, at, g.Out[0]
	}
	return w, nil
}

func (w *walk) close() {
	for _, f := range w.forms {
		if f != nil {
			f.close()
		}
	}
}

// chaseCalls is the calls each walk length is timed over.
const chaseCalls = 16

// chaseLatency is the seconds one read waits walking a random cycle
// through the cache lines of set bytes, so the prefetchers cannot run
// ahead, between the walk's two lengths so a call's own cost cancels. Each
// call starts where the one before stopped, so neither length re-reads a
// prefix. A cache's set is twice the level before it, past that level and
// within the TLB's reach, and the set is on huge pages where the kernel
// grants them, as the engine's frames ask, so the walk waits for the cache
// and not for page walks.
func chaseLatency(w *walk, set int64) (float64, error) {
	const line = 64
	lines := int(max(2, set/line))
	if lines*line/4 > w.words {
		return 0, fmt.Errorf("probe: a chase of %d bytes past the walk's %d words", set, w.words)
	}
	raw, free, err := hugeBuffer(lines * line)
	if err != nil {
		return 0, err
	}
	defer free()
	words := unsafe.Slice((*uint32)(unsafe.Pointer(&raw[0])), len(raw)/4)
	order := rand.Perm(lines)
	for i, at := range order {
		words[at*line/4] = uint32(order[(i+1)%lines] * line / 4)
	}
	// A line of its own: the word a walk starts at and stops on.
	at := make([]uint32, 16)
	at[0] = uint32(order[0] * line / 4)
	var secs [2]float64
	for i, f := range w.forms {
		b, err := f.bind(map[graph.ValueID]unsafe.Pointer{w.at: unsafe.Pointer(&at[0]), w.tab: unsafe.Pointer(&raw[0]), w.next: unsafe.Pointer(&at[0])})
		if err != nil {
			return 0, err
		}
		secs[i] = parallel(1, func(int) {
			for range chaseCalls {
				b.run()
			}
		})
		runtime.KeepAlive(b)
	}
	runtime.KeepAlive(at)
	if !(secs[1] > secs[0]) {
		return 0, fmt.Errorf("probe: walks of %d and %d steps took %v seconds", chaseShort, chaseLong, secs)
	}
	return (secs[1] - secs[0]) / (chaseCalls * (chaseLong - chaseShort)), nil
}

// retention is the fraction of a buffer an eighth of cache c's size that
// c still serves after reads of twice its size: the buffer read, the reads
// made, the buffer read again and timed, its time placed between all of it
// served by c and all of it by the level beyond, which serves at rate
// after start.
func retention(r *reader, c cost.Cache, rate, start float64, evict []byte) (float64, error) {
	n := max(256, c.Bytes/8&^255)
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = byte(i)
	}
	at, err := r.on(buf)
	if err != nil {
		return 0, err
	}
	out, err := r.on(evict[:min(int64(len(evict)), 2*c.Bytes)&^255])
	if err != nil {
		return 0, err
	}
	clock := clockCost()
	var times []float64
	for range 4 * rounds {
		at.run()
		out.run()
		t0 := time.Now()
		at.run()
		times = append(times, time.Since(t0).Seconds()-clock)
	}
	runtime.KeepAlive(buf)
	slices.Sort(times)
	hit := float64(n)/c.Rate.Thread + c.Start
	miss := float64(n)/rate + start
	if miss <= hit {
		return 0, nil
	}
	return min(1, max(0, (miss-times[len(times)/2])/(miss-hit))), nil
}

// cycling is the share of cache c's capacity a buffer twice its size,
// read over and over, finds in it: each pass's time placed between all of
// it served by c and all of it by the level beyond, at rate.
func cycling(r *reader, c cost.Cache, rate float64, evict []byte) (float64, error) {
	n := min(int64(len(evict)), 2*c.Bytes) &^ 255
	at, err := r.on(evict[:n])
	if err != nil {
		return 0, err
	}
	var times []float64
	for range 2 * rounds {
		t0 := time.Now()
		at.run()
		times = append(times, time.Since(t0).Seconds())
	}
	runtime.KeepAlive(evict)
	slices.Sort(times)
	hit, miss := float64(n)/c.Rate.Thread, float64(n)/rate
	if miss <= hit {
		return 0, nil
	}
	h := min(1, max(0, (miss-times[len(times)/2])/(miss-hit)))
	return h * float64(n) / float64(c.Bytes), nil
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
func streamStart(r *reader, rate float64, evict []byte) (float64, error) {
	const n = 32 << 10
	buf := make([]byte, n)
	for i := range buf {
		buf[i] = byte(i)
	}
	for i := range evict {
		evict[i] = byte(i)
	}
	at, err := r.on(buf)
	if err != nil {
		return 0, err
	}
	out, err := r.on(evict[:len(evict)&^255])
	if err != nil {
		return 0, err
	}
	clock := clockCost()
	var times []float64
	for range 4 * rounds {
		at.run()
		out.run()
		t0 := time.Now()
		at.run()
		times = append(times, time.Since(t0).Seconds()-clock)
	}
	runtime.KeepAlive(buf)
	runtime.KeepAlive(evict)
	slices.Sort(times)
	return max(0, times[len(times)/2]-n/rate), nil
}

// readRate is the bytes a second threads reading set bytes each, over and
// over, get together through the composed sum: the median of the timed
// rounds.
func readRate(r *reader, set int64, threads int) (float64, error) {
	set = max(256, set&^255)
	reps := max(1, int((256<<20)/set))
	bufs := make([][]byte, threads)
	runs := make([]*bound, threads)
	for t := range bufs {
		bufs[t] = make([]byte, set)
		for i := range bufs[t] {
			bufs[t][i] = byte(i)
		}
		var err error
		if runs[t], err = r.on(bufs[t]); err != nil {
			return 0, err
		}
	}
	run := func(t int) {
		for range reps {
			runs[t].run()
		}
		runtime.KeepAlive(bufs[t])
	}
	secs := parallel(threads, run)
	return float64(set) * float64(reps) * float64(threads) / secs, nil
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
