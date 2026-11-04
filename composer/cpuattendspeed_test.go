package composer

import (
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/quant"
)

// TestComposedCPUAttendSpeed times each kernel of a prompt's attention the
// composer forms for the CPU: a chunk of JITLLM_ATT_ROWS rows (64) after
// JITLLM_ATT_PAST positions (448) of one sequence's binary16 history, at
// JITLLM_ATT_HEADS query heads (16), JITLLM_ATT_KV kv heads (8) and
// JITLLM_ATT_DIM (128), its calls on one thread; JITLLM_CPU_NOVNNI forms
// for a host without VNNI. A measurement: JITLLM_ATT_SPEED=1.
func TestComposedCPUAttendSpeed(t *testing.T) {
	if os.Getenv("JITLLM_ATT_SPEED") == "" {
		t.Skip("a measurement: set JITLLM_ATT_SPEED=1")
	}
	rows, past, heads, kvh, dim := 64, 448, 16, 8, 128
	for v, p := range map[string]*int{"JITLLM_ATT_ROWS": &rows, "JITLLM_ATT_PAST": &past, "JITLLM_ATT_HEADS": &heads, "JITLLM_ATT_KV": &kvh, "JITLLM_ATT_DIM": &dim} {
		if s := os.Getenv(v); s != "" {
			n, err := strconv.Atoi(s)
			if err != nil {
				t.Fatalf("%s: %v", v, err)
			}
			*p = n
		}
	}
	const page = 64
	isa := uint64(cpu.CPU().ISA())
	if os.Getenv("JITLLM_CPU_NOVNNI") != "" {
		isa &^= uint64(x86.ISAVNNI)
	}
	g := attendGraph(heads, kvh, dim)
	keys := (past + rows + page - 1) / page * page
	_, codes, err := speedRegistry(-1, -1).Form(Request{Graph: g,
		Step:   Step{Rows: rows, Page: page, Keys: keys, ActWindow: 32, Seqs: 1},
		Device: Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: isa, ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	w, qw := kvh*dim, heads*dim
	words := func(n int) []uint32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return f32Words(x)
	}
	pool := make([]uint32, keys*w/2)
	for i := range pool {
		pool[i] = uint32(quant.EncodeHalf(float32(rng.NormFloat64()))) | uint32(quant.EncodeHalf(float32(rng.NormFloat64())))<<16
	}
	vpool := append([]uint32(nil), pool...)
	tab := make([]uint32, rows*4)
	for r := range rows {
		p := uint32(past + r)
		copy(tab[r*4:], []uint32{0, 0, p + 1, p})
	}
	pages := make([]uint32, keys/page)
	for i := range pages {
		pages[i] = uint32(i)
	}
	bufs := map[string][]uint32{"q": words(rows * qw), "k": words(rows * w), "v": words(rows * w), "rows": tab, "pages": pages,
		"q.off": make([]uint32, rows*4), "rows.n": {uint32(rows)}, "kv.k": pool, "kv.v": vpool}
	bufs[g.Values[g.Out[0]].Name] = make([]uint32, rows*qw)
	own := map[Operand][]uint32{}
	var keep [][]uint32
	var total time.Duration
	for _, c := range codes {
		code, err := cpu.NewHost().Map(c.CPU)
		if err != nil {
			t.Fatal(err)
		}
		calls := c.Calls
		if c.Live > 0 && len(calls) > 0 {
			calls = append([]int{c.LiveUnitsOf(rows)}, calls[1:]...)
		}
		n := 1
		for _, e := range calls {
			n *= e
		}
		args := make([]cpu.Args, n)
		for at := range n {
			base := unsafe.Pointer(&args[at])
			for _, a := range c.Args {
				slot := (*uint64)(unsafe.Add(base, a.Field))
				o := a.Operand
				coord := func(ax int) int {
					v := at
					for x := len(calls) - 1; x > ax; x-- {
						v /= calls[x]
					}
					return v % calls[ax]
				}
				switch {
				case a.Coord > 0:
					*slot = uint64(coord(a.Coord - 1))
					continue
				case a.Length:
					*slot = uint64(a.Extent)
					continue
				case a.Value || a.Moving:
					t.Fatalf("%s: a word the caller computes", c.Name)
				}
				o.Copy = 0
				var ws []uint32
				switch {
				case a.Words != nil:
					ws = a.Words
				case o.Kind == OperandValue && bufs[g.Values[o.Value].Name] != nil:
					ws = bufs[g.Values[o.Value].Name]
				case o.Kind == OperandTable && bufs[o.Name] != nil:
					ws = bufs[o.Name]
				case o.Kind == OperandValue || o.Kind == OperandScratch:
					if own[o] == nil {
						own[o] = make([]uint32, ownWords(codes, o))
					}
					ws = own[o]
				default:
					t.Fatalf("%s binds %+v", c.Name, o)
				}
				keep = append(keep, ws)
				if a.Word {
					*slot = uint64(ws[a.Offset])
					continue
				}
				elem := 4
				if a.Elem > 0 {
					elem = a.Elem
				}
				off := a.Offset
				for ax, per := range a.Per {
					off += coord(ax) * per
				}
				*slot = uint64(uintptr(unsafe.Pointer(&ws[0]))) + uint64(elem*off)
			}
		}
		run := func() {
			for i := range args {
				a := args[i]
				code.Call(&a)
			}
		}
		run()
		reps := 0
		t0 := time.Now()
		for time.Since(t0) < 300*time.Millisecond {
			run()
			reps++
		}
		per := time.Since(t0) / time.Duration(reps)
		total += per
		t.Logf("att %dq%dkv%dd %d rows after %d: %s: %d B, calls %v, %v a run", heads, kvh, dim, rows, past, c.Name, len(c.CPU), calls, per)
		code.Close()
	}
	runtime.KeepAlive(keep)
	t.Logf("att %dq%dkv%dd %d rows after %d: %v the chunk", heads, kvh, dim, rows, past, total)
}
