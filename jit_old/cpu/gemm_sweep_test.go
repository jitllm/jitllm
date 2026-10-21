//go:build amd64 || arm64

package cpu

import (
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// TestGEMMTileSweep measures the register-tiling span against the kernel that
// ships. Tiles are measured round-robin rather than sequentially, so thermal
// drift does not favour the early or late shapes.
func TestGEMMTileSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates and measures")
	}
	// The format is a knob: Q8_0 has no unpack, so its tile span is flat,
	// while the k-quants differ a lot by tile (hence nn.tileFor).
	qt := quant.Q8_0
	switch os.Getenv("SWEEP_T") {
	case "Q3_K":
		qt = quant.Q3_K
	case "Q4_K":
		qt = quant.Q4_K
	case "Q5_K":
		qt = quant.Q5_K
	case "Q6_K":
		qt = quant.Q6_K
	case "Q4_0":
		qt = quant.Q4_0
	case "Q8_0":
		qt = quant.Q8_0
	}
	win := Q8Block
	if os.Getenv("SWEEP_WIN") == "256" {
		win = 256
	}
	t.Logf("format %v, activation window %d", qt, win)
	if ggufDeclined(t, qt) {
		return
	}
	// The shape is a knob too: a model's projections differ in k and rows, and
	// the tuner keys on rows because the tile that fits one does not fit the
	// other.
	k, targetRows := 2048, 1152 // divisible by 1,2,3,4,6
	if v := os.Getenv("SWEEP_K"); v != "" {
		k, _ = strconv.Atoi(v)
	}
	if v := os.Getenv("SWEEP_ROWS"); v != "" {
		targetRows, _ = strconv.Atoi(v)
	}
	// The 1xN column: on arm64 it is the only way to get wide. The register
	// budget is 2*nacc + ... with nacc = mr*vpk, so at mr=1 a k-quant reaches
	// 24 token columns where mr=2 caps at 16.
	tiles := [][2]int{{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 6},
		{2, 1}, {4, 1}, {6, 1}, {8, 1}, {2, 2}, {3, 2}, {4, 2}, {6, 2},
		{2, 3}, {4, 3}, {3, 4}, {2, 6}}

	type arm struct {
		mr, nr, rows, tok int
		kern              *Code
		run               func()
		macs              float64
		times             []time.Duration
	}
	var arms []*arm
	nb := BlocksPerRow(qt, k)
	nbAct := k / 32
	konst := KernelConst(qt)
	for _, tl := range tiles {
		mr, nr := tl[0], tl[1]
		code, err := EmitGEMMWindow(qt, mr, nr, nb, win)
		if err != nil {
			continue
		}
		kern := mustMap(t, code)
		defer kern.Close()
		groups := targetRows / mr
		rows, tok := groups*mr, GEMMTokens(nr)

		w := make([]byte, rows*RowBytes(qt, k))
		for i := range w {
			w[i] = byte(i * 7)
		}
		// Give each block a sane fp16 scale rather than random bits. Where it
		// sits differs by format: leading for Q4_0/Q8_0/Q4_K, trailing for
		// Q3_K/Q6_K.
		bb := int(qt.BlockBytes())
		off0 := bb - 2
		switch qt {
		case quant.Q4_0, quant.Q8_0, quant.Q4_K, quant.Q5_K:
			off0 = 0
		}
		for off := off0; off+2 <= len(w); off += bb {
			w[off], w[off+1] = 0x00, 0x34
		}
		x := make([]float32, tok*k)
		for i := range x {
			x[i] = float32((i%251)-125) / 64
		}
		q := make([]int8, tok*k)
		sc := make([]float32, nbAct*tok)
		sm := make([]int32, nbAct*tok)
		hm := make([]int32, 2*nbAct*tok)
		if err := PackGEMMActivationsRange(qt, q, sc, sm, hm, x, tok, k, 0, tok, win); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, rows*tok)
		scratch := make([]float32, GEMMScratchMax)
		args := Args{
			Out: &out[0], W: &w[0], A: &q[0], AScale: &sc[0],
			Rows: int64(groups), K: int64(nb), RowStr: int64(RowBytes(qt, k)),
			Scr: &konst[0], Cols: int64(tok), OutStr: int64(tok * 4), ASum: &sm[0], AHalfSum: &hm[0],
			Scratch: (*byte)(unsafe.Pointer(&scratch[0])),
		}
		a := &arm{mr: mr, nr: nr, rows: rows, tok: tok, kern: kern,
			macs: float64(rows) * float64(tok) * float64(k)}
		a.run = func() { kern.Call(&args) }
		arms = append(arms, a)
	}

	for _, a := range arms {
		a.run() // warm
	}
	// Each sample must be long enough to measure: one call is ~0.2 ms, where
	// scheduler noise and timer granularity dominate, so a sample repeats the
	// call to span ~5 ms.
	reps := 1
	{
		start := time.Now()
		arms[0].run()
		if d := time.Since(start); d > 0 {
			reps = int(5*time.Millisecond/d) + 1
		}
	}
	const rounds = 21
	for r := 0; r < rounds; r++ {
		for _, a := range arms {
			start := time.Now()
			for i := 0; i < reps; i++ {
				a.run()
			}
			a.times = append(a.times, time.Since(start)/time.Duration(reps))
		}
	}
	t.Logf("%d rounds of %d repetitions per tile", rounds, reps)
	type res struct {
		name string
		gmac float64
		iqr  float64
	}
	var out []res
	for _, a := range arms {
		sort.Slice(a.times, func(i, j int) bool { return a.times[i] < a.times[j] })
		med := a.times[len(a.times)/2]
		iqr := a.times[len(a.times)*3/4] - a.times[len(a.times)/4]
		out = append(out, res{
			name: itoa(a.mr) + "x" + itoa(a.nr) + " (" + itoa(a.mr) + " rows x " + itoa(a.tok) + " tok)",
			gmac: a.macs / med.Seconds() / 1e9,
			iqr:  float64(iqr) / float64(med),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].gmac > out[j].gmac })
	for _, r := range out {
		flag := ""
		if r.iqr > 0.10 {
			flag = "  UNSTABLE"
		}
		t.Logf("%-24s %7.1f Gmac/s   IQR/med %4.1f%%%s", r.name, r.gmac, 100*r.iqr, flag)
	}
	t.Logf("span best/worst = %.2fx", out[0].gmac/out[len(out)-1].gmac)
}
