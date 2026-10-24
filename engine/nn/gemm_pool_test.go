//go:build amd64 || arm64

package nn

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"sort"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestGEMMPoolRate measures the int8 GEMM across the whole P-core pool, which
// is the number that compares against a GPU.
//
// jit's TestGEMMTileSweep runs one core; this reports the single-core and pool
// rates and the scaling between them, so a CPU-against-GPU comparison is
// checkable.
func TestGEMMPoolRate(t *testing.T) {
	// The emitters are host-pure: EmitGEMM succeeds for a target this CPU
	// cannot run, so "it emitted" is not "it executes" and picking a tile by
	// emitter success walks straight into a SIGILL on a pre-VNNI host.
	requireRowMajorHost(t, quant.Q8_0)
	if testing.Short() {
		t.Skip("allocates and measures")
	}
	const k = 2048
	nb := k / 32
	konst := cpu.KernelConst(quant.Q8_0)

	// Pick the best tile this architecture actually emits.
	var mr, nr int
	for _, tl := range [][2]int{{2, 3}, {3, 2}, {2, 2}, {4, 2}, {6, 2}, {4, 1}, {1, 1}} {
		if _, err := cpu.EmitGEMMWindow(quant.Q8_0, tl[0], tl[1], nb, cpu.Q8Block); err == nil {
			mr, nr = tl[0], tl[1]
			break
		}
	}
	if mr == 0 {
		t.Skip("no GEMM tile available on this architecture")
	}
	code, err := cpu.EmitGEMMWindow(quant.Q8_0, mr, nr, nb, cpu.Q8Block)
	if err != nil {
		t.Fatal(err)
	}
	kern, err := cpu.Map(code)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()

	cores := sched.PCores()
	tok := cpu.GEMMTokens(nr)
	// Enough rows that every worker gets real work and the set leaves L3.
	groups := 96 * len(cores)
	rows := groups * mr
	rowBytes := cpu.RowBytes(quant.Q8_0, k)

	w := make([]byte, rows*rowBytes)
	for i := range w {
		w[i] = byte(i * 7)
	}
	for off := 0; off+2 <= len(w); off += 34 {
		w[off], w[off+1] = 0x00, 0x34
	}
	x := make([]float32, tok*k)
	for i := range x {
		x[i] = float32((i%251)-125) / 64
	}
	q := make([]int8, tok*k)
	sc := make([]float32, nb*tok)
	sm := make([]int32, nb*tok)
	hm := make([]int32, 2*nb*tok)
	if err := oracle.PackGEMMActivations(q, sc, sm, hm, x, tok, k, 0, tok, -1, -1, cpu.Q8Block, int(cpu.BiasC(quant.Q8_0)), cpu.NeedsHalfSums(quant.Q8_0)); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, rows*tok)

	// Per-worker scratch: the kernel writes into it, so sharing one would be a
	// data race that shows up as a wrong number rather than a crash.
	call := func(lo, hi int, scratch []float32) {
		if lo >= hi {
			return
		}
		args := cpu.Args{
			Out: &out[lo*mr*tok], W: &w[lo*mr*rowBytes], A: &q[0], AScale: &sc[0],
			Rows: int64(hi - lo), K: int64(nb), RowStr: int64(rowBytes),
			Scr: &konst[0], Cols: int64(tok), OutStr: int64(tok * 4),
			ASum: &sm[0], AHalfSum: &hm[0],
			Scratch: (*byte)(unsafe.Pointer(&scratch[0])),
		}
		kern.Call(&args)
	}

	macs := float64(rows) * float64(tok) * float64(k)
	measure := func(name string, n int) float64 {
		scratch := make([][]float32, n)
		for i := range scratch {
			scratch[i] = make([]float32, 12*8)
		}
		var run func()
		if n == 1 {
			run = func() { call(0, groups, scratch[0]) }
		} else {
			p := sched.New(cores[:n])
			defer p.Close()
			run = func() {
				p.Do(groups, 1, func(worker, lo, hi int) { call(lo, hi, scratch[worker]) })
			}
		}
		run()
		reps := 1
		start := time.Now()
		run()
		if d := time.Since(start); d > 0 {
			reps = int(5*time.Millisecond/d) + 1
		}
		var ts []time.Duration
		for r := 0; r < 21; r++ {
			s := time.Now()
			for i := 0; i < reps; i++ {
				run()
			}
			ts = append(ts, time.Since(s)/time.Duration(reps))
		}
		sort.Slice(ts, func(i, j int) bool { return ts[i] < ts[j] })
		med := ts[len(ts)/2]
		iqr := float64(ts[len(ts)*3/4]-ts[len(ts)/4]) / float64(med)
		g := macs / med.Seconds() / 1e9
		flag := ""
		if iqr > 0.10 {
			flag = "  UNSTABLE -- reporting, not concluding"
		}
		t.Logf("%-12s %8.1f Gmac/s   IQR/med %4.1f%%%s", name, g, 100*iqr, flag)
		return g
	}

	t.Logf("tile %dx%d, %d rows x %d tok x k=%d, %d P-cores", mr, nr, rows, tok, k, len(cores))
	one := measure("1 core", 1)
	all := measure("all cores", len(cores))
	t.Logf("scaling %.2fx across %d cores (%.0f%% of linear)",
		all/one, len(cores), 100*all/one/float64(len(cores)))
}
