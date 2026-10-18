package tier

import (
	"bufio"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// choose picks which GPU API to run on when a host offers more than one.
//
// The answer is measured, not assumed: which API is faster depends on the
// card, the driver and the kernel, and an integrated GPU sharing host memory
// can be slower than the CPU. WithAPI pins it; otherwise it is probed once and
// cached against the devices present, as engine/nn/tune.go caches pack width. TuneOff
// skips the probe and takes the first device; TuneForce re-probes.
//
// One physical card enumerated by two backends is one candidate, not two:
// backend.APIRank decides between them (see AGENTS.md, Scope), and the probe
// only compares genuinely different devices.
func choose(devs []backend.Device, kb knobs) backend.Device {
	if len(devs) == 1 {
		return devs[0]
	}
	keep := func(i int) backend.Device {
		for j, d := range devs {
			if j != i {
				d.Close()
			}
		}
		return devs[i]
	}
	// The pin is checked against the whole list, before deduplication: an
	// explicit API request must win even when the same card also answers to a
	// preferred API. The preference governs only the default.
	if want := kb.api; want != "" {
		for i, d := range devs {
			if d.API() == want {
				return keep(i)
			}
		}
		fmt.Fprintf(os.Stderr, "jitllm: no %q device; using %s\n", want, devs[0].API())
		return keep(0)
	}
	// Candidates are indices because keep() closes by position in the original
	// list: a dropped duplicate is still owned here and must not leak.
	cand, merges := preferOnePerCard(devs)
	if backend.Verbose() {
		for _, m := range merges {
			fmt.Fprintf(os.Stderr, "jitllm: %s\n", m)
		}
	}
	if len(cand) == 1 {
		return keep(cand[0])
	}
	sub := make([]backend.Device, len(cand))
	for i, j := range cand {
		sub[i] = devs[j]
	}
	key := deviceKey(sub)
	if kb.tune == TuneOff {
		return keep(cand[0])
	}
	if api, ok := lookupCache(kb.cacheDir, "gpuapi", key); ok && kb.tune != TuneForce {
		for _, j := range cand {
			if devs[j].API() == api {
				return keep(j)
			}
		}
	}
	best, rates := probe(sub)
	if backend.Verbose() {
		for i, d := range sub {
			fmt.Fprintf(os.Stderr, "jitllm: probe %-6s %-40s %8.1f Gmac/s\n", d.API(), d.Name(), rates[i])
		}
	}
	saveCache(kb.cacheDir, "gpuapi", key, sub[best].API())
	return keep(cand[best])
}

// probe times one representative decode matvec on each device.
//
// The shape is gemma's q/o projection -- 2048x2048 Q4_0, about 2.6 MB of
// weights -- which is big enough to leave any cache and small enough that the
// probe costs a few milliseconds. It measures the kernel, not the bandwidth
// ceiling, because the kernel is what will run.
func probe(devs []backend.Device) (best int, rates []float64) {
	const rows, k = 2048, 2048
	rates = make([]float64, len(devs))
	nb := k / 32
	raw := make([]byte, rows*nb*18)
	rng := rand.New(rand.NewSource(1))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	qs, dw, scw, err := kernels.PackWeights(kernels.Q4_0, raw, rows, k)
	if err != nil {
		return 0, rates
	}
	_ = scw
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	a, as, sum, err := kernels.PackActivations(x)
	if err != nil {
		return 0, rates
	}
	ax := append(append([]float32{}, as...), sum...)
	macs := float64(rows) * float64(k)

	for i, d := range devs {
		rates[i] = timeOne(d, qs, dw, a, ax, rows, k, macs)
		if rates[i] > rates[best] {
			best = i
		}
	}
	return best, rates
}

func timeOne(d backend.Device, qs, dw, a []uint32, ax []float32,
	rows, k int, macs float64) float64 {

	ker, err := kernels.MatVec(kernels.MatVecShape{T: kernels.Q4_0, K: k, Rows: rows, Split: 1})
	if err != nil {
		return 0
	}
	kern, err := d.Compile(ker)
	if err != nil {
		return 0
	}
	defer kern.Close()

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(p []byte) backend.Buf {
		b, err := d.Alloc(len(p))
		if err != nil {
			return nil
		}
		bufs = append(bufs, b)
		if b.Write(p) != nil {
			return nil
		}
		return b
	}
	bQS, bD, bSC := up(u32b(qs)), up(u32b(dw)), up(u32b([]uint32{0}))
	bA, bAX := up(u32b(a)), up(f32b(ax))
	bOut := up(make([]byte, rows*4))
	for _, b := range []backend.Buf{bQS, bD, bSC, bA, bAX, bOut} {
		if b == nil {
			return 0
		}
	}
	run := func() bool {
		ok := true
		d.Session(func(s backend.Session) {
			if s.Launch(kern, (rows+127)/128, 128, bQS, bD, bSC, bA, bAX, bOut) != nil {
				ok = false
				return
			}
			ok = s.Sync() == nil
		})
		return ok
	}
	for i := 0; i < 5; i++ {
		if !run() {
			return 0
		}
	}
	var rs []float64
	for r := 0; r < 9; r++ {
		t0 := time.Now()
		if !run() {
			return 0
		}
		rs = append(rs, macs/time.Since(t0).Seconds()/1e9)
	}
	sort.Float64s(rs)
	return rs[len(rs)/2]
}

// deviceKey identifies the set of devices present by UUID and driver version,
// so a cached answer does not survive a hardware or driver change, and two
// identical cards are never one entry by their shared name. "" when any device
// cannot say, and then nothing is cached.
func deviceKey(devs []backend.Device) string {
	var names []string
	for _, d := range devs {
		id, drv := backend.IdentityOf(d), backend.DriverOf(d)
		if !id.Known() || drv == "" {
			return ""
		}
		names = append(names, d.API()+"/"+id.Key+"/"+drv)
	}
	sort.Strings(names)
	h := fnv.New64a()
	for _, n := range names {
		h.Write([]byte(n))
		h.Write([]byte{0})
	}
	return fmt.Sprintf("%016x", h.Sum64())
}

// cachePath is a tuner cache file in the directory WithTuneCache named:
// "gpuapi" for choose, "gpusplit1" for tuneSplit, "gpusplit" and "gpusplitid"
// for the sequence retunes. "" when there is no directory.
func cachePath(dir, name string) string {
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, name)
}

// lookupCache reads one entry. A file that cannot be read or a line that does
// not parse is no entry: the caller measures again and rewrites it.
func lookupCache(dir, name, key string) (string, bool) {
	if key == "" {
		return "", false
	}
	p := cachePath(dir, name)
	if p == "" {
		return "", false
	}
	f, err := os.Open(p)
	if err != nil {
		return "", false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), " ")
		if ok && k == key {
			return v, true
		}
	}
	return "", false
}

func saveCache(dir, name, key, val string) {
	p := cachePath(dir, name)
	if p == "" || key == "" {
		return
	}
	if os.MkdirAll(filepath.Dir(p), 0o755) != nil {
		return
	}
	lines := map[string]string{}
	if f, err := os.Open(p); err == nil {
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if k, v, ok := strings.Cut(sc.Text(), " "); ok {
				lines[k] = v
			}
		}
		f.Close()
	}
	lines[key] = val
	var out strings.Builder
	keys := make([]string, 0, len(lines))
	for k := range lines {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(&out, "%s %s\n", k, lines[k])
	}
	// Written whole and renamed into place, so a concurrent reader sees the
	// old file or the new one, never half of either.
	tmp := fmt.Sprintf("%s.%d.tmp", p, os.Getpid())
	if os.WriteFile(tmp, []byte(out.String()), 0o644) == nil {
		os.Rename(tmp, p)
	}
}
