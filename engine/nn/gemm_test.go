//go:build amd64 && linux

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

// TestGEMMScaling asks whether the prefill kernel's single-core rate survives
// the pool.
//
// Prefill should scale differently from decode: decode is memory-bound, while
// prefill re-reads a weight tile for every token tile and should be
// compute-bound.
func TestGEMMScaling(t *testing.T) {
	// The emitters are host-pure: EmitGEMM succeeds for a target this CPU
	// cannot run, so "it emitted" is not "it executes" and picking a tile by
	// emitter success walks straight into a SIGILL on a pre-VNNI host.
	requireRowMajorHost(t, quant.Q8_0)
	if testing.Short() {
		t.Skip("allocates and measures")
	}
	const k, mr, nr = 2048, 4, 3 // the single-core sweep's winner
	nb := k / 32
	tok := cpu.GEMMTokens(nr)
	rowBytes := cpu.RowBytes(quant.Q8_0, k)
	const rows = 4096 // 8.9 MB of weights: bigger than L2, resident in L3

	code, err := cpu.EmitGEMMWindow(quant.Q8_0, mr, nr, nb, cpu.Q8Block)
	if err != nil {
		t.Fatal(err)
	}
	kern, err := cpu.Map(code)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()

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
	konst := cpu.KernelConst(quant.Q8_0)
	out := make([]float32, rows*tok)

	macs := float64(rows) * float64(tok) * float64(k)
	cpus := sched.PCores()
	type row struct {
		n    int
		gmac float64
		iqr  float64
	}
	var res []row
	for _, n := range []int{1, 2, 3, 4, 5, 6} {
		if len(cpus) < n {
			continue
		}
		p := sched.New(cpus[:n])
		// One scratch tile per participant; the kernel's float accumulators
		// spill there when the tile is too wide to keep them in registers.
		scratch := make([]float32, p.N()*cpu.GEMMScratchMax)
		groups := rows / mr
		chunk := max(1, groups/(4*p.N()))
		run := func() {
			p.Do(groups, chunk, func(worker, lo, hi int) {
				args := cpu.Args{
					Out: &out[lo*mr*tok], W: &w[lo*mr*rowBytes], A: &q[0], AScale: &sc[0],
					Rows: int64(hi - lo), K: int64(nb), RowStr: int64(rowBytes),
					Scr: &konst[0], Cols: int64(tok), OutStr: int64(tok * 4), ASum: &sm[0], AHalfSum: &hm[0],
					Scratch: (*byte)(unsafe.Pointer(&scratch[worker*128])),
				}
				kern.Call(&args)
			})
		}
		run()
		var ts []time.Duration
		for i := 0; i < 15; i++ {
			s := time.Now()
			run()
			ts = append(ts, time.Since(s))
		}
		p.Close()
		sort.Slice(ts, func(i, j int) bool { return ts[i] < ts[j] })
		med := ts[len(ts)/2]
		iqr := ts[len(ts)*3/4] - ts[len(ts)/4]
		res = append(res, row{n, macs / med.Seconds() / 1e9, float64(iqr) / float64(med)})
		var raw string
		for _, d := range ts {
			raw += " " + d.Round(10*time.Microsecond).String()
		}
		t.Logf("  n=%d raw:%s", n, raw)
	}
	base := res[0].gmac
	for _, r := range res {
		t.Logf("cores=%d  %7.1f Gmac/s  = %6.1f GFLOP/s   %.2fx   IQR/med %4.1f%%",
			r.n, r.gmac, 2*r.gmac, r.gmac/base, 100*r.iqr)
	}
	t.Logf("llama.cpp pp512 on this box is 141.8 tok/s = 281 GFLOP/s, whole model, six cores")
}
