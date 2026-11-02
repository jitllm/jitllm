//go:build amd64 && linux

package composer

import (
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/offheap"
)

// TestComposedCPUMatVecAgainstMain times the composed decode matvec against
// main's MatVecPacked on main's pool, interleaved, A/A first, on ~192 MiB of
// weight copies. Knobs: JITLLM_CPU_{ROWS,K,TYPE,THREADS,ROUNDS,HUGE,VNNI,
// ONLY,SECONDS,AHEAD,JAM,APART,APART_LINES}. A measurement: JITLLM_CPU_SPEED=1.
func TestComposedCPUMatVecAgainstMain(t *testing.T) {
	if os.Getenv("JITLLM_CPU_SPEED") == "" {
		t.Skip("a measurement: set JITLLM_CPU_SPEED=1")
	}
	rows, k, threads, ahead, jam, secs, rounds := 8192, 2048, 0, -1, -1, 10, 20
	for v, p := range map[string]*int{"JITLLM_CPU_ROWS": &rows, "JITLLM_CPU_K": &k, "JITLLM_CPU_THREADS": &threads,
		"JITLLM_CPU_AHEAD": &ahead, "JITLLM_CPU_JAM": &jam, "JITLLM_CPU_SECONDS": &secs, "JITLLM_CPU_ROUNDS": &rounds} {
		if s := os.Getenv(v); s != "" {
			n, err := strconv.Atoi(s)
			if err != nil {
				t.Fatalf("%s: %v", v, err)
			}
			*p = n
		}
	}
	typ := quant.Q4_K
	if s := os.Getenv("JITLLM_CPU_TYPE"); s != "" {
		found := false
		for _, qt := range quant.PackedTypes {
			if qt.String() == s {
				typ, found = qt, true
			}
		}
		if !found {
			t.Fatalf("JITLLM_CPU_TYPE %q is not a packed type", s)
		}
	}
	var jo []nn.Option
	if threads > 0 {
		jo = append(jo, nn.WithSched(sched.WithCores(threads)))
	}
	j := nn.NewJIT(k, rows, []quant.Type{typ}, jo...)
	if j == nil {
		t.Fatal("no host JIT")
	}
	defer j.Close()
	workers := j.Workers()

	rng := rand.New(rand.NewSource(5))
	q, _ := layout.QuantOf(typ)
	jt, _ := jlm.TypeOf(typ)
	w := cpuTestWeights(typ, q, rows, k, rng)
	qs, d, sc, err := layout.PackWeights(q, w, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	planes := [3][]uint32{qs, d, sc}
	wbytes := 4 * (len(qs) + len(d) + len(sc))
	var span [3]int
	per := 0
	for i, p := range planes {
		span[i] = (4*len(p) + 4095) &^ 4095
		per += span[i]
	}
	n := max(2, (192<<20)/per)
	mem, mapped, err := offheap.MapOffHeap(n * per)
	if err != nil {
		t.Fatal(err)
	}
	if mapped {
		defer offheap.UnmapOffHeap(mem)
	}
	huge := os.Getenv("JITLLM_CPU_HUGE") != "0"
	if huge {
		if err := syscall.Madvise(mem, 14); err != nil { // MADV_HUGEPAGE
			t.Fatalf("madvise: %v", err)
		}
	}
	copies := make([][3][]uint32, n)
	packed := make([]nn.Packed, n)
	for i := range copies {
		off := i * per
		var bs [3][]byte
		for r, p := range planes {
			if len(p) == 0 {
				continue
			}
			b := mem[off : off+4*len(p)]
			copy(b, unsafe.Slice((*byte)(unsafe.Pointer(&p[0])), 4*len(p)))
			copies[i][r] = unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(p))
			bs[r] = b
			off += span[r]
		}
		packed[i] = nn.Packed{QS: bs[0], D: bs[1], SC: bs[2], Stride: rows}
	}

	reg := speedRegistry(ahead, jam)
	isa := uint64(cpu.CPU().ISA())
	if os.Getenv("JITLLM_CPU_VNNI") == "0" {
		isa &^= uint64(x86.ISAVNNI)
	}
	_, codes, err := reg.Form(Request{
		Graph:   graph.MatVec(jlm.RoleOutput, rows, k),
		Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
		Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceCPU, Slots: workers, Lanes: 8, ISA: isa, ExactDiv: true},
	})
	if err != nil {
		t.Fatalf("%v: %v", typ, err)
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, rows+16)
	dst := make([]float32, rows)
	scratch := map[Operand][]uint32{}
	type bound struct {
		code   *cpu.Code
		sp     callSpace
		ncalls int
		args   []cpu.Args // a template a copy
		coords []CallArg
		moving [][]movingArg // a copy's
	}
	kerns := make([]bound, len(codes))
	for ci, c := range codes {
		code, err := cpu.Map(c.CPU)
		if err != nil {
			t.Fatal(err)
		}
		defer code.Close()
		sp := spaceOf(&c, workers)
		b := bound{code: code, sp: sp, ncalls: sp.n, args: make([]cpu.Args, n), moving: make([][]movingArg, n)}
		for i := range n {
			base := unsafe.Pointer(&b.args[i])
			for _, a := range c.Args {
				slot := (*uint64)(unsafe.Add(base, a.Field))
				var words []uint32
				o := a.Operand
				switch {
				case a.Coord > 0:
					if i == 0 {
						b.coords = append(b.coords, a)
					}
					continue
				case a.Length || a.Value:
					*slot = argValue(a, sp)
					continue
				case a.Words != nil:
					words = append([]uint32(nil), a.Words...)
				case o.Kind == OperandValue && o.Value == 0:
					words = unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x))
				case o.Kind == OperandValue && o.Value == 1:
					words = unsafe.Slice((*uint32)(unsafe.Pointer(&out[0])), len(out))
				case o.Kind == OperandWeight:
					words = copies[i][o.Region]
				case o.Kind == OperandScratch:
					if scratch[o] == nil {
						size := 1 << 16
						for _, bd := range c.Binds {
							if bd.Operand == o && bd.Bytes > 0 {
								size = int(bd.Bytes)/4 + 1
							}
						}
						scratch[o] = make([]uint32, size)
					}
					words = scratch[o]
				default:
					t.Fatalf("%s binds %+v", c.Name, o)
				}
				if a.Word {
					*slot = uint64(words[a.Offset])
					continue
				}
				elem := 4
				if a.Elem > 0 {
					elem = a.Elem
				}
				*slot = uint64(uintptr(unsafe.Pointer(&words[0]))) + uint64(elem*a.Offset)
				if len(a.Per) > 0 {
					b.moving[i] = append(b.moving[i], movingArg{a, *slot, elem})
				}
			}
		}
		kerns[ci] = b
		t.Logf("%v %dx%d kernel %d %s: %d B of code, calls %v passes %v", typ, rows, k, ci, c.Name, len(c.CPU), c.Calls, c.Passes)
	}
	wargs := make([]cpu.Args, workers)
	var cur *bound
	var curCopy int
	body := func(wk, lo, hi int) {
		a := &wargs[wk]
		*a = cur.args[curCopy]
		callRange(cur.code, a, cur.sp, cur.coords, cur.moving[curCopy], lo, hi)
	}
	composed := func(i int) {
		curCopy = i
		for ci := range kerns {
			cur = &kerns[ci]
			if cur.ncalls <= 1 {
				a := &wargs[0]
				*a = cur.args[i]
				callAll(cur.code, a, cur.sp, cur.coords, cur.moving[i])
				continue
			}
			j.ParallelWorkers(cur.ncalls, 1, body)
		}
	}
	mainMV := func(i int) {
		clear(dst)
		j.NewInput()
		if !j.MatVecPacked(dst, typ, &packed[i], x, rows, k) {
			t.Fatal("main's MatVecPacked declined")
		}
	}
	clear(out)
	composed(0)
	mainMV(0)
	var num, den float64
	for r := range rows {
		e := float64(out[r] - dst[r])
		num += e * e
		den += float64(dst[r]) * float64(dst[r])
	}
	if den == 0 || math.IsNaN(num) || num/den > 1e-3 {
		t.Fatalf("the composed matvec and main's disagree: NMSE %.3e (den %.3e)", num/den, den)
	}
	t.Logf("%v %dx%d, %d copies of %d B (%.0f MiB), %d workers, huge pages %v: composed against main NMSE %.2e",
		typ, rows, k, n, wbytes, float64(n*per)/(1<<20), workers, huge, num/den)
	pass := func(f func(int)) time.Duration {
		t0 := time.Now()
		for i := range n {
			f(i)
		}
		return time.Since(t0)
	}
	gbs := func(d time.Duration) float64 { return float64(n) * float64(wbytes) / d.Seconds() / 1e9 }
	if only := os.Getenv("JITLLM_CPU_ONLY"); only != "" {
		f := map[string]func(int){"main": mainMV, "composed": composed}[only]
		if f == nil {
			t.Fatalf("JITLLM_CPU_ONLY %q: want main or composed", only)
		}
		pass(f)
		var best time.Duration
		var reps int
		t0 := time.Now()
		for time.Since(t0) < time.Duration(secs)*time.Second {
			d := pass(f)
			if best == 0 || d < best {
				best = d
			}
			reps++
		}
		el := time.Since(t0)
		t.Logf("%s alone: %d passes, %.2f GB/s mean, %.2f GB/s best, %.1f us a matvec", only, reps,
			float64(reps)*float64(n)*float64(wbytes)/el.Seconds()/1e9, gbs(best), el.Seconds()*1e6/float64(reps*n))
		return
	}
	ab := func(name string, a, b func(int)) {
		pass(a)
		pass(b)
		var ra, rb, ratio []float64
		for r := range rounds {
			var da, db time.Duration
			if r%2 == 0 {
				da = pass(a)
				db = pass(b)
			} else {
				db = pass(b)
				da = pass(a)
			}
			ra, rb = append(ra, gbs(da)), append(rb, gbs(db))
			ratio = append(ratio, gbs(db)/gbs(da))
		}
		med := func(v []float64) (float64, float64) {
			s := slices.Clone(v)
			slices.Sort(s)
			m := s[len(s)/2]
			return m, (s[3*len(s)/4] - s[len(s)/4]) / m
		}
		am, aq := med(ra)
		bm, bq := med(rb)
		rm, rq := med(ratio)
		t.Logf("%s: %v %dx%d %d workers, %d rounds: A %.2f GB/s (IQR/med %.3f), B %.2f GB/s (%.3f), B/A %.4f (IQR/med %.3f)",
			name, typ, rows, k, workers, rounds, am, aq, bm, bq, rm, rq)
	}
	ab("A/A main against main", mainMV, mainMV)
	ab("main (A) against composed (B)", mainMV, composed)
}
