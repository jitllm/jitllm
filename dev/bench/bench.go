// Package bench is the measurement harness. Every performance claim jitllm makes
// goes through it; see AGENTS.md RULE 1 and RULE 2.
package bench

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Case is one side of an A/B comparison.
type Case struct {
	Name string
	// BytesPerIter is how many bytes of DRAM one iteration must move. It powers
	// the physics guard: a result implying more bandwidth than the host has is a
	// harness bug, not a discovery. Zero disables the check.
	BytesPerIter uint64
	// Threads is how many cores Fn uses, which selects the ceiling to check
	// against. Zero means one.
	Threads int
	Fn      func(iters int)
}

// Result is a paired A/B measurement.
type Result struct {
	A, B    string
	Rounds  int
	Ratios  []float64 // per-round B/A, sorted
	Median  float64
	IQR     float64
	AMedian time.Duration
	BMedian time.Duration
}

// Stable reports whether the spread is tight enough to believe: IQR/median at
// most 10%.
func (r Result) Stable() bool { return r.Median != 0 && r.IQR/r.Median <= 0.10 }

func (r Result) String() string {
	verdict := "  UNSTABLE — do not quote this"
	if r.Stable() {
		verdict = ""
	}
	return fmt.Sprintf("%s vs %s: %.4fx  (IQR/median %.1f%%, n=%d, A %v, B %v)%s",
		r.B, r.A, r.Median, 100*r.IQR/r.Median, r.Rounds,
		r.AMedian.Round(time.Microsecond), r.BMedian.Round(time.Microsecond), verdict)
}

// AB runs a paired, interleaved A/B comparison and returns the median of the
// per-round ratios.
//
//   - Interleaved, never A-then-B: thermal drift would be attributed to the
//     change.
//   - Order alternates every round, so any residual ordering bias cancels.
//   - Median of per-round ratios, not the ratio of medians, which rewards a
//     change that happened to run while the machine was cold.
func AB(a, b Case, rounds, iters int) Result {
	if rounds < 2 {
		rounds = 2
	}
	as := make([]time.Duration, 0, rounds)
	bs := make([]time.Duration, 0, rounds)
	ratios := make([]float64, 0, rounds)

	// One untimed round of each, so neither side pays for cold caches or the
	// first-touch page faults on a freshly mmap'd model.
	a.Fn(iters)
	b.Fn(iters)

	for i := 0; i < rounds; i++ {
		var ta, tb time.Duration
		if i%2 == 0 {
			ta, tb = time_(a.Fn, iters), time_(b.Fn, iters)
		} else {
			tb, ta = time_(b.Fn, iters), time_(a.Fn, iters)
		}
		as = append(as, ta)
		bs = append(bs, tb)
		ratios = append(ratios, float64(ta)/float64(tb)) // >1 means B is faster
	}

	check(a, ta_(as), iters)
	check(b, ta_(bs), iters)

	sort.Float64s(ratios)
	return Result{
		A: a.Name, B: b.Name, Rounds: rounds, Ratios: ratios,
		Median: medianF(ratios), IQR: iqr(ratios),
		AMedian: medianD(as), BMedian: medianD(bs),
	}
}

func time_(fn func(int), iters int) time.Duration {
	start := time.Now()
	fn(iters)
	return time.Since(start)
}

// check is the physics guard. A benchmark that implies more DRAM bandwidth than
// the host can deliver has measured something else (a working set that fits in
// cache, or a loop the compiler deleted). Panicking is correct: an impossible
// number that is silently returned gets quoted.
func check(c Case, fastest time.Duration, iters int) {
	if c.BytesPerIter == 0 || fastest <= 0 {
		return
	}
	threads := max(c.Threads, 1)
	implied := float64(c.BytesPerIter) * float64(iters) / fastest.Seconds()
	if raceEnabled {
		// See guard_race.go: the ceiling is instrumented and the kernel is not.
		return
	}
	wall := MemWall(threads)
	if implied <= wall*guardFactor {
		return
	}
	// The wall may have been measured while the machine was busy, so
	// re-measure before accusing and keep the higher reading.
	wall = remeasure(threads, wall)
	if implied <= wall*guardFactor {
		return
	}
	panic(fmt.Sprintf("bench: %q implies %.1f GB/s, which exceeds the %d-thread ceiling of %.1f GB/s — the benchmark is wrong, not fast",
		c.Name, implied/1e9, threads, wall*guardFactor/1e9))
}

// MemWall measures sustained DRAM read bandwidth with the given number of
// goroutines, in bytes/sec.
//
// It is measured rather than hardcoded, and it is a lower bound, not the
// hardware's peak: scalar Go loads on unpinned goroutines read well below a
// vectorised C probe. Treat it as a sanity bound; see guardFactor.
func MemWall(threads int) float64 {
	if threads < 1 {
		threads = 1
	}
	// The working set must not fit in L3, or this measures cache as DRAM.
	const perThread = (64 << 20) / 8 // 64 MiB of float64 per thread
	bufs := make([][]float64, threads)
	for i := range bufs {
		bufs[i] = make([]float64, perThread)
		for j := range bufs[i] {
			bufs[i][j] = float64(j)
		}
	}
	best := 0.0
	for rep := 0; rep < 3; rep++ { // take the best of three; noise only slows it
		var wg sync.WaitGroup
		sums := make([]float64, threads)
		start := time.Now()
		for t := 0; t < threads; t++ {
			wg.Add(1)
			go func(t int) {
				defer wg.Done()
				// Eight independent accumulators: a single chain is bound by
				// the FP add's latency, not by memory, and understated the
				// wall by about half.
				b := bufs[t]
				var a0, a1, a2, a3, a4, a5, a6, a7 float64
				for i := 0; i+7 < len(b); i += 8 {
					a0 += b[i]
					a1 += b[i+1]
					a2 += b[i+2]
					a3 += b[i+3]
					a4 += b[i+4]
					a5 += b[i+5]
					a6 += b[i+6]
					a7 += b[i+7]
				}
				sums[t] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
			}(t)
		}
		wg.Wait()
		el := time.Since(start).Seconds()
		runtime.KeepAlive(sums)
		if bw := float64(threads) * perThread * 8 / el; bw > best {
			best = bw
		}
	}
	return best
}

// guardFactor is how far above the measured Go-side wall a claim may sit before
// the physics guard calls it impossible.
//
// It is deliberately loose because MemWall is a lower bound. The guard's job
// is catching order-of-magnitude lies (cache reported as DRAM, a deleted loop),
// which are off by 10x or by infinity, never by 50%.
const guardFactor = 2.0

// remeasure is the higher of a reading already taken and a fresh one.
func remeasure(threads int, prev float64) float64 {
	return max(prev, MemWall(threads))
}

// Guard refuses to measure on a machine that is busy or hot, and returns why.
//
// The thermal limit is 90 C: a mobile CPU's package zones read 99-101 C in
// ordinary use, and a gate that never opens gets bypassed.
//
// The busy check is PSI (/proc/pressure/cpu some avg10), not the load average,
// which counts uninterruptible tasks and read 4.42 on an idle desktop. The load
// average is only the fallback where PSI is absent. PSI cannot see a tenant on
// the specific cores a run pins to; see AGENTS.md RULE 2.
//
// skip is an argument, not an environment variable, so bypassing the guard is
// visible at the call (cmd/jitllm maps JITLLM_NO_GUARD onto it).
func Guard(skip bool) error {
	if skip {
		return nil
	}
	if p, err := cpuPressure(); err == nil {
		if p > 25 {
			return fmt.Errorf("bench: CPU pressure is %.1f%% over the last 10s, want < 25 — something else is running", p)
		}
	} else if la, err := loadavg(); err == nil && la > 8.0 {
		return fmt.Errorf("bench: load average is %.2f and this kernel has no PSI, want < 8.0", la)
	}
	if c, zone, err := hottest(); err == nil && c > 90 {
		return fmt.Errorf("bench: %s is at %.1f C, want < 90 — let it cool", zone, c)
	}
	return nil
}

// cpuPressure is the percentage of the last 10 seconds in which at least one
// runnable task was waiting for a CPU. Zero means nothing ever queued.
func cpuPressure() (float64, error) {
	b, err := os.ReadFile("/proc/pressure/cpu")
	if err != nil {
		return 0, err
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(ln, "some ") {
			continue
		}
		for _, f := range strings.Fields(ln) {
			if v, ok := strings.CutPrefix(f, "avg10="); ok {
				return strconv.ParseFloat(v, 64)
			}
		}
	}
	return 0, fmt.Errorf("bench: no `some avg10` in /proc/pressure/cpu")
}

func loadavg() (float64, error) {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, err
	}
	f := strings.Fields(string(b))
	if len(f) == 0 {
		return 0, fmt.Errorf("bench: unparseable /proc/loadavg")
	}
	return strconv.ParseFloat(f[0], 64)
}

func hottest() (float64, string, error) {
	dirs, err := os.ReadDir("/sys/class/thermal")
	if err != nil {
		return 0, "", err
	}
	best, name := 0.0, ""
	for _, d := range dirs {
		if !strings.HasPrefix(d.Name(), "thermal_zone") {
			continue
		}
		b, err := os.ReadFile("/sys/class/thermal/" + d.Name() + "/temp")
		if err != nil {
			continue
		}
		mC, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
		if err != nil {
			continue
		}
		if c := mC / 1000; c > best {
			best, name = c, d.Name()
		}
	}
	if name == "" {
		return 0, "", fmt.Errorf("bench: no thermal zones")
	}
	return best, name, nil
}

// Report formats a throughput claim the way AGENTS.md RULE 1 requires: never a bare rate, but
// always as a fraction of a wall measured in this same process, so a reader with
// a calculator can falsify it.
func Report(label string, bytesMoved uint64, elapsed time.Duration, threads int) string {
	t := max(threads, 1)
	gbs := float64(bytesMoved) / elapsed.Seconds()
	wall := MemWall(t)
	// A fraction over 100% means the wall was mismeasured (MemWall reads low
	// on a busy box), so re-measure once.
	if gbs > wall {
		wall = remeasure(t, wall)
	}
	over := ""
	if gbs > wall {
		over = "  (ABOVE the measured wall -- the wall probe is a lower bound, not the hardware peak)"
	}
	return fmt.Sprintf("%s: %v, %.2f GB/s = %.0f%% of the %.1f GB/s %d-thread Go-side wall%s",
		label, elapsed.Round(time.Millisecond), gbs/1e9, 100*gbs/wall, wall/1e9, t, over)
}

// Cores reports the physical P-cores worth using. Deliberately not NumCPU,
// which counts hyperthreads and E-cores that add no read bandwidth.
func Cores() int { return min(6, max(1, runtime.NumCPU()/2)) }

func medianF(sorted []float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

func iqr(sorted []float64) float64 {
	n := len(sorted)
	if n < 4 {
		return 0
	}
	return math.Abs(sorted[3*n/4] - sorted[n/4])
}

func medianD(ds []time.Duration) time.Duration {
	c := append([]time.Duration(nil), ds...)
	sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	return c[len(c)/2]
}

// ta_ returns the fastest duration, which is the one the physics guard must
// check.
func ta_(ds []time.Duration) time.Duration {
	best := ds[0]
	for _, d := range ds {
		if d < best {
			best = d
		}
	}
	return best
}
