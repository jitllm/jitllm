package jit

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/x86"
)

// TestComposedCPUMatVecSpeed times each kernel of a decode matvec the
// composer forms for the CPU, a weight of JITLLM_CPU_ROWS rows (default
// 8192) by JITLLM_CPU_K (2048), as bytes of the weight a second: formed for
// JITLLM_CPU_SLOTS workers, its calls run on JITLLM_CPU_THREADS threads, a step of
// JITLLM_CPU_STEP rows (1) jammed JITLLM_CPU_JAM rows a call (RowJam).
// JITLLM_CPU_AHEAD sets Ahead's trips, JITLLM_CPU_NOVNNI forms for the host
// without VNNI, and JITLLM_CPU_DUMP names a directory each kernel's code is
// written to. The weight sits on the Go heap, not on the frames' huge pages
// a model's does. A measurement: JITLLM_CPU_SPEED=1.
func TestComposedCPUMatVecSpeed(t *testing.T) {
	if os.Getenv("JITLLM_CPU_SPEED") == "" {
		t.Skip("a measurement: set JITLLM_CPU_SPEED=1")
	}
	rows, k, slots, threads, ahead, step, jam := 8192, 2048, 1, 1, -1, 1, -1
	for v, p := range map[string]*int{"JITLLM_CPU_ROWS": &rows, "JITLLM_CPU_K": &k, "JITLLM_CPU_SLOTS": &slots, "JITLLM_CPU_THREADS": &threads, "JITLLM_CPU_AHEAD": &ahead, "JITLLM_CPU_STEP": &step, "JITLLM_CPU_JAM": &jam} {
		if s := os.Getenv(v); s != "" {
			n, err := strconv.Atoi(s)
			if err != nil {
				t.Fatalf("%s: %v", v, err)
			}
			*p = n
		}
	}
	reg := speedRegistry(ahead, jam)
	isa := uint64(cpu.CPU().ISA())
	if os.Getenv("JITLLM_CPU_NOVNNI") != "" {
		isa &^= uint64(x86.ISAVNNI)
	}
	rng := rand.New(rand.NewSource(5))
	for _, typ := range []quant.Type{quant.Q4_K, quant.Q8_0, quant.Q6_K, quant.Q4_0} {
		q, _ := kernels.QuantOf(typ)
		jt, _ := jlm.TypeOf(typ)
		w := cpuTestWeights(typ, q, rows, k, rng)
		qs, d, sc, err := kernels.PackWeights(q, w, rows, k)
		if err != nil {
			t.Fatal(err)
		}
		wbytes := 4 * (len(qs) + len(d) + len(sc))
		_, codes, err := reg.Form(Request{
			Graph:   graph.MatVec(jlm.RoleOutput, rows, k),
			Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jt},
			Step:    Step{Rows: step, Page: 64, Keys: 1, ActWindow: 32},
			Device:  Device{Kind: DeviceCPU, Slots: slots, Lanes: 8, ISA: isa, ExactDiv: true},
		})
		if err != nil {
			t.Fatalf("%v: %v", typ, err)
		}
		x := make([]float32, step*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		out := make([]float32, step*rows+16)
		scratch := map[Operand][]uint32{}
		for ci, c := range codes {
			if dir := os.Getenv("JITLLM_CPU_DUMP"); dir != "" {
				if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("%v-%d.bin", typ, ci)), c.CPU, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			code, err := cpu.Map(c.CPU)
			if err != nil {
				t.Fatal(err)
			}
			var args cpu.Args
			base := unsafe.Pointer(&args)
			var coords []CallArg
			var moving []movingArg
			for _, a := range c.Args {
				slot := (*uint64)(unsafe.Add(base, a.Field))
				var words []uint32
				o := a.Operand
				switch {
				case a.Coord > 0:
					coords = append(coords, a)
					continue
				case a.Length || a.Value:
					*slot = argValue(a, spaceOf(&c, threads))
					continue
				case a.Words != nil:
					words = append([]uint32(nil), a.Words...)
				case o.Kind == OperandValue && o.Value == 0:
					words = unsafe.Slice((*uint32)(unsafe.Pointer(&x[0])), len(x))
				case o.Kind == OperandValue && o.Value == 1:
					words = unsafe.Slice((*uint32)(unsafe.Pointer(&out[0])), len(out))
				case o.Kind == OperandWeight:
					words = [3][]uint32{qs, d, sc}[o.Region]
				case o.Kind == OperandScratch:
					if scratch[o] == nil {
						size := 1 << 16
						for _, b := range c.Binds {
							if b.Operand == o && b.Bytes > 0 {
								size = int(b.Bytes)/4 + 1
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
					moving = append(moving, movingArg{a, *slot, elem})
				}
			}
			sp := spaceOf(&c, threads)
			n := sp.n
			run := func() {
				if threads <= 1 || n <= 1 {
					callAll(code, &args, sp, coords, moving)
					return
				}
				var wg sync.WaitGroup
				for w := range threads {
					wg.Add(1)
					go func() {
						defer wg.Done()
						a := args
						callRange(code, &a, sp, coords, moving, w*n/threads, (w+1)*n/threads)
					}()
				}
				wg.Wait()
			}
			run()
			reps := 0
			t0 := time.Now()
			for time.Since(t0) < 500*time.Millisecond {
				run()
				reps++
			}
			per := time.Since(t0) / time.Duration(reps)
			t.Logf("%v %dx%d at %d rows %s: %d B of code, calls %v, %v a run, %.2f GMAC/s, %.2f GB/s of the weight's %d B", typ, rows, k, step, c.Name, len(c.CPU), c.Calls, per, float64(rows*k*step)/per.Seconds()/1e9,
				float64(wbytes)/per.Seconds()/1e9, wbytes)
			code.Close()
		}
		runtime.KeepAlive(scratch)
	}
}

// speedRegistry is the default registry with the CPU's Ahead trips and
// RowJam rows set where ahead and jam are not negative, and Jam's strip
// to JITLLM_CPU_STRIP where that is set.
func speedRegistry(ahead, jam int) *Registry {
	reg := Default()
	strip, most, least := -1, -1, -1
	if s := os.Getenv("JITLLM_CPU_STRIP"); s != "" {
		strip, _ = strconv.Atoi(s)
	}
	if s := os.Getenv("JITLLM_CPU_STREAMS_MOST"); s != "" {
		most, _ = strconv.Atoi(s)
	}
	if s := os.Getenv("JITLLM_CPU_WORDS_LEAST"); s != "" {
		least, _ = strconv.Atoi(s)
	}
	p := reg.Profiles[DeviceCPU]
	p.Passes = slices.Clone(p.Passes)
	for i, ps := range p.Passes {
		if j, ok := ps.(RowJam); ok && jam >= 0 {
			j.N = jam
			p.Passes[i] = j
		}
		if a, ok := ps.(Ahead); ok && ahead >= 0 {
			a.Trips = ahead
			p.Passes[i] = a
		}
		if q, ok := ps.(Streams); ok {
			if most >= 0 {
				q.Most = most
			}
			if s := os.Getenv("JITLLM_CPU_APART"); s != "" {
				q.Apart, _ = strconv.Atoi(s)
			}
			if s := os.Getenv("JITLLM_CPU_APART_LINES"); s != "" {
				q.Fill, _ = strconv.Atoi(s)
			}
			p.Passes[i] = q
		}
		if q, ok := ps.(StreamWords); ok && least >= 0 {
			q.Least = least
			p.Passes[i] = q
		}
		if j, ok := ps.(Jam); ok && strip >= 0 {
			j.Strip = strip
			p.Passes[i] = j
		}
	}
	reg.Profiles[DeviceCPU] = p
	return reg
}

// callRange is callAll over the calls [lo, hi) alone.
func callRange(code *cpu.Code, args *cpu.Args, sp callSpace, coords []CallArg, moving []movingArg, lo, hi int) {
	base := unsafe.Pointer(args)
	for at := lo; at < hi; at++ {
		for _, a := range coords {
			*(*uint64)(unsafe.Add(base, a.Field)) = uint64(sp.coord(at, a.Coord-1))
		}
		if sp.rows > 0 {
			*(*uint64)(unsafe.Add(base, sp.field)) = uint64(min(sp.rows, sp.share-at*sp.rows))
		}
		for _, m := range moving {
			addr := m.at
			for ax, per := range m.a.Per {
				addr += uint64(sp.coord(at, ax) * per * m.elem)
			}
			*(*uint64)(unsafe.Add(base, m.a.Field)) = addr
		}
		code.Call(args)
	}
}
