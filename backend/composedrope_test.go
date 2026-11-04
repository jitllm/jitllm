package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedRopeTableOverAxes runs the composer's rotary table over one
// position axis, contiguous sections and interleaved ones, at ragged pair
// counts, on every backend present at one row and at five. Each row is held
// to the float64 table of each pair at its axis's position, and each pair of
// a table over several axes must be, to the bit, the composer's one-axis
// table's pair at the position of the pair's axis as the runs assign it. A
// guard after the output must survive, and the table at two axes' positions
// swapped must be far from it. The positions reach the table as the model's
// graph takes them, through the rows' marks (OpRopePos, gated on its own
// beside the composer): an XD-RoPE picture's mark states a row's four axes
// apart. On this host's CPU no argument may be one whose offset a caller
// would have to compute.
func TestComposedRopeTableOverAxes(t *testing.T) {
	t.Run("cpu/1 rows", func(t *testing.T) { composedRopeAxes(t, cpuRope, 1) })
	gpuLock(t)
	devs := backend.OpenForTest()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, rows := range []int{1, 5} {
			t.Run(fmt.Sprintf("%s/%d rows", d.API(), rows), func(t *testing.T) { composedRopeAxes(t, gpuRope(d), rows) })
		}
	}
}

// composedRopeGuard is the word after a table's output and in every element
// before the table writes it: a quiet NaN, so an element left unwritten
// shows, and a word written past the table changes it.
const composedRopeGuard = 0x7fc0dead

// composedRopeAxes holds run's tables at rows rows.
func composedRopeAxes(t *testing.T, run ropeRunner, rows int) {
	// The table's bound against the float64 table's
	// (TestRopeTableOnDeviceMatchesHost's).
	const bound = 3e-7
	cases := []struct {
		pairs       int
		sections    [4]int
		interleaved bool
	}{
		{pairs: 1}, {pairs: 7}, {pairs: 9}, {pairs: 17}, {pairs: 64}, {pairs: 129},
		{pairs: 64, sections: [4]int{16, 24, 24, 0}},
		{pairs: 13, sections: [4]int{3, 5, 4, 1}},
		{pairs: 9, sections: [4]int{0, 2, 7, 0}},
		{pairs: 1, sections: [4]int{1, 0, 0, 0}},
		{pairs: 64, sections: [4]int{24, 20, 20, 0}, interleaved: true},
		{pairs: 11, sections: [4]int{2, 3, 2, 0}, interleaved: true},
		{pairs: 2, sections: [4]int{1, 1, 0, 0}, interleaved: true},
	}
	// Each row's position on each axis: the axes apart, and every digit of
	// the decomposition reached. Axis 0 is the row's cache position, which a
	// plain table turns by.
	at := make([][composer.RopeAxes]uint32, rows)
	for r := range at {
		u := uint32(r)
		at[r] = [composer.RopeAxes]uint32{1<<27 + 5*u, 1000 + 3*u, 37 + u, 70003 + 11*u}
	}
	ran := 0
	for _, tc := range cases {
		multi := tc.sections != [4]int{}
		name := fmt.Sprintf("%d pairs", tc.pairs)
		if multi {
			name = fmt.Sprintf("%d pairs over sections %v interleaved %v", tc.pairs, tc.sections, tc.interleaved)
		}
		t.Run(name, func(t *testing.T) {
			rope := nn.Rope{NRot: 2 * tc.pairs, Base: 1e6, Neox: true}
			axis := make([]int, tc.pairs)
			if multi {
				rope.Runs = nn.MRopeRuns(tc.sections[:])
				if tc.interleaved {
					rope.Runs = nn.IMRopeRuns(tc.sections[:], tc.pairs)
				}
				p := 0
				for _, u := range rope.Runs {
					for range u.Pairs {
						axis[p] = u.Axis
						p++
					}
				}
				if p != tc.pairs {
					t.Fatalf("main's runs cover %d of %d pairs", p, tc.pairs)
				}
			}
			got := run(t, rope, tc.sections, tc.interleaved, at)
			worst := 0.0
			for r := range rows {
				for p := range tc.pairs {
					ref := ropeTabOracle(rope, int(at[r][axis[p]]), tc.pairs)
					for j := range 2 {
						i := r*2*tc.pairs + 2*p + j
						d := math.Abs(float64(got[i]) - ref[2*p+j])
						if math.IsNaN(d) {
							t.Fatalf("row %d pair %d is %v where the float64 table's is %v", r, p, got[i], ref[2*p+j])
						}
						worst = max(worst, d)
					}
				}
			}
			if worst > bound {
				t.Fatalf("the composer's table is %.3e from the float64 table, past %.0e", worst, bound)
			}
			if !multi {
				t.Logf("one axis: %.3e from the float64 table", worst)
				ran++
				return
			}
			// Each row's position on each axis, the one-axis table of a row
			// there: pair p of row r is pair p of that table at p's axis.
			plain := rope
			plain.Runs = nil
			for r := range rows {
				var flat [composer.RopeAxes][]float32
				for a, p := range at[r] {
					flat[a] = run(t, plain, [4]int{}, false, [][composer.RopeAxes]uint32{{p, p, p, p}})
				}
				for p, a := range axis {
					dst := r*2*tc.pairs + 2*p
					if math.Float32bits(got[dst]) != math.Float32bits(flat[a][2*p]) || math.Float32bits(got[dst+1]) != math.Float32bits(flat[a][2*p+1]) {
						t.Fatalf("row %d pair %d on axis %d is (%v, %v), the one-axis table's at its position (%v, %v)",
							r, p, a, got[dst], got[dst+1], flat[a][2*p], flat[a][2*p+1])
					}
				}
			}
			// The violation: two axes that turn pairs, their positions swapped.
			// A kernel reading one for the other writes that table, which must
			// be far from the one it wrote.
			var used []int
			for a := range composer.RopeAxes {
				for _, b := range axis {
					if a == b {
						used = append(used, a)
						break
					}
				}
			}
			if len(used) < 2 {
				t.Logf("one axis turns every pair: %.3e from the float64 table", worst)
				ran++
				return
			}
			x, y := used[len(used)-2], used[len(used)-1]
			apart := 0.0
			for r := range rows {
				swap := at[r]
				swap[x], swap[y] = swap[y], swap[x]
				for p := range tc.pairs {
					ref := ropeTabOracle(rope, int(swap[axis[p]]), tc.pairs)
					for j := range 2 {
						apart = max(apart, math.Abs(float64(got[r*2*tc.pairs+2*p+j])-ref[2*p+j]))
					}
				}
			}
			if !(apart > 1e3*bound) {
				t.Fatalf("the table at axes %d and %d swapped is %.3e from the one written: the gate cannot see a wrong axis", x, y, apart)
			}
			t.Logf("%.3e from the float64 table; axes %d and %d swapped, %.3e", worst, x, y, apart)
			ran++
		})
	}
	if ran != len(cases) {
		t.Fatalf("%d of %d tables ran", ran, len(cases))
	}
}

// ropeRunner runs the composer's table for rope over sections at a step of
// a row each of at, each row at its positions on the rotary axes, with the
// planes the model folds (nn.Rope.TabPlanes) and the blocks the plan
// carries, and returns the table, its guard checked.
type ropeRunner func(t *testing.T, rope nn.Rope, sections [4]int, interleaved bool, at [][composer.RopeAxes]uint32) []float32

// ropeRowTables is the rows' table and the marks that put each row of at at
// its positions (OpRopePos): a row whose axes agree is at its cache
// position under no mark; otherwise each row is its own sequence under one
// XD-RoPE picture's mark, which states a row's cache position, its column,
// its row and the picture's ordinal apart, its grid of 2048 by 2048 a
// picture's patches (fewer than 2^31, as the composer takes a product of a
// table's words to be). marks is the step's marks a sequence.
func ropeRowTables(t *testing.T, at [][composer.RopeAxes]uint32) (rows, marksBuf []uint32, marks int) {
	t.Helper()
	const w, h = 1 << 11, 1 << 11
	rows = []uint32{uint32(len(at))}
	for _, a := range at {
		if a[1] != a[0] || a[2] != a[0] || a[3] != a[0] {
			marks = 1
		}
	}
	for r, a := range at {
		off := uint32(r * (1 + marks*graph.MarkWords))
		rows = append(rows, a[0], off)
		if marks == 0 {
			continue
		}
		jj := a[2]*w + a[1]
		if a[1] >= w || a[2] >= h || jj > a[0] {
			t.Fatalf("row %d's positions %v are not one XD-RoPE mark's", r, a)
		}
		var m [graph.MarkWords]uint32
		m[graph.MarkAt], m[graph.MarkKind] = a[0]-jj, graph.MarkGridXD
		m[graph.MarkH], m[graph.MarkW], m[graph.MarkIndex] = h, w, a[3]
		marksBuf = append(marksBuf, 1)
		marksBuf = append(marksBuf, m[:]...)
	}
	if marksBuf == nil {
		marksBuf = []uint32{0}
	}
	return rows, marksBuf, marks
}

// passedWords is the words of each buffer the codes pass between them (the
// positions OpRopePos writes for the table, a scan's scratch), a buffer a
// name: the most any code reaches of it past the offset its copy is bound
// at, and at least a value's width a row of rows; a scratch stating no
// reach is given a page of words.
func passedWords(codes []composer.Code, g *graph.Graph, rows int) map[string]int {
	n := map[string]int{}
	for _, c := range codes {
		at := map[string]map[int]int{}
		for _, a := range c.Args {
			if o := a.Operand; o.Copy > 0 {
				if at[o.Name] == nil {
					at[o.Name] = map[int]int{}
				}
				at[o.Name][o.Copy] = a.Offset
			}
		}
		for _, b := range c.Binds {
			o := b.Operand
			if o.Kind != composer.OperandValue && o.Kind != composer.OperandScratch {
				continue
			}
			w := int(b.Bytes+3)/4 + at[o.Name][o.Copy] + 1
			switch {
			case o.Kind == composer.OperandValue:
				w = max(w, g.Values[o.Value].Width*rows)
			case b.Bytes == 0:
				w = max(w, 1<<14)
			}
			n[o.Name] = max(n[o.Name], w)
		}
	}
	return n
}

func composedRopeCodes(t *testing.T, d composer.Device, rope nn.Rope, sections [4]int, interleaved bool, rows, marks int) ([]composer.Code, *graph.Graph) {
	t.Helper()
	rt := graph.RopeTable{NRot: rope.NRot, Sections: sections, Interleaved: interleaved}
	g := rt.Build()
	_, codes, err := composer.Default().Form(composer.Request{Graph: g, Part: composer.PartTable,
		Step: composer.Step{Rows: rows, Page: 64, Keys: 1, Seqs: rows, Marks: marks}, Device: d})
	if err != nil {
		t.Fatal(err)
	}
	return codes, g
}

// guardedTable is a table of n words and the guard after it, every word the
// guard.
func guardedTable(n int) []uint32 {
	w := make([]uint32, n+16)
	for i := range w {
		w[i] = composedRopeGuard
	}
	return w
}

// unguard is the table in w, its guard checked.
func unguard(t *testing.T, w []uint32, n int) []float32 {
	t.Helper()
	for i := n; i < len(w); i++ {
		if w[i] != composedRopeGuard {
			t.Fatalf("the table wrote word %d, past its %d", i, n)
		}
	}
	got := make([]float32, n)
	for i := range got {
		got[i] = math.Float32frombits(w[i])
	}
	return got
}

func planeWords(rope nn.Rope) []uint32 {
	var w []uint32
	for _, v := range rope.TabPlanes(rope.NRot / 2) {
		w = append(w, math.Float32bits(v))
	}
	return w
}

// gpuRope launches the table on d.
func gpuRope(d backend.Device) ropeRunner {
	return func(t *testing.T, rope nn.Rope, sections [4]int, interleaved bool, at [][composer.RopeAxes]uint32) []float32 {
		t.Helper()
		rows := len(at)
		rowTab, marksTab, marks := ropeRowTables(t, at)
		lanes := 1
		if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); ok {
			lanes = ir.SubgroupLanes
		}
		units := 0
		if m, ok := d.(backend.MultiprocessorDevice); ok {
			units = m.Multiprocessors()
		}
		codes, g := composedRopeCodes(t, composer.Device{Kind: composer.DeviceGPU, Slots: d.Slots(), Lanes: lanes, Units: units, ExactDiv: composer.ExactDivOn(d.API())},
			rope, sections, interleaved, rows, marks)
		gg := newGPU(t, d)
		defer gg.free()
		n := rows * rope.NRot
		guarded := guardedTable(n)
		out := gg.up(u32bytes(guarded))
		// rows.n is every row of the form, as the tier binds it for a part
		// whose call states none.
		tables := map[string]backend.Buf{"rope.rows": gg.up(u32bytes(rowTab)), "rope.marks": gg.up(u32bytes(marksTab)), "rows.n": gg.up(u32bytes([]uint32{^uint32(0)})),
			"rope.planes": gg.up(u32bytes(planeWords(rope))), "rope.cs": out}
		for name, n := range passedWords(codes, g, rows) {
			if tables[name] == nil {
				tables[name] = gg.up(make([]byte, 4*n))
			}
		}
		for _, c := range codes {
			k, err := d.Compile(c.GPU)
			if err != nil {
				t.Fatalf("compiling %s: %v", c.Name, err)
			}
			var bufs []backend.Buf
			for _, b := range c.Binds {
				o := b.Operand
				switch {
				case b.Words != nil:
					bufs = append(bufs, gg.up(u32bytes(b.Words)))
				case tables[o.Name] != nil:
					bufs = append(bufs, tables[o.Name])
				default:
					k.Close()
					t.Fatalf("%s binds %+v, which the table's call site does not hold", c.Name, o)
				}
			}
			err = k.Launch(c.Groups, c.GPU.Group[0], bufs...)
			k.Close()
			if err != nil {
				t.Fatalf("launching %s: %v", c.Name, err)
			}
		}
		raw := make([]byte, 4*len(guarded))
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		for i := range guarded {
			guarded[i] = binary.LittleEndian.Uint32(raw[4*i:])
		}
		return unguard(t, guarded, n)
	}
}

// cpuRope calls the table on this host's CPU, each code once a call of its
// calls' space, each operand where its call words say.
func cpuRope(t *testing.T, rope nn.Rope, sections [4]int, interleaved bool, at [][composer.RopeAxes]uint32) []float32 {
	t.Helper()
	rows := len(at)
	rowTab, marksTab, marks := ropeRowTables(t, at)
	codes, g := composedRopeCodes(t, composer.Device{Kind: composer.DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true},
		rope, sections, interleaved, rows, marks)
	n := rows * rope.NRot
	out := guardedTable(n)
	tables := map[string][]uint32{"rope.rows": rowTab, "rope.marks": marksTab, "rope.planes": planeWords(rope), "rope.cs": out}
	for name, n := range passedWords(codes, g, rows) {
		if tables[name] == nil {
			tables[name] = make([]uint32, n)
		}
	}
	var keep [][]uint32
	for _, c := range codes {
		if c.CPU == nil {
			t.Fatalf("%s has no CPU code", c.Name)
		}
		code, err := cpu.Map(c.CPU)
		if err != nil {
			t.Fatalf("mapping %s: %v", c.Name, err)
		}
		calls := 1
		for _, e := range c.Calls {
			calls *= e
		}
		// coord is axis ax's coordinate of call at in the calls' space,
		// its last axis the fastest.
		coord := func(at, ax int) int {
			for x := len(c.Calls) - 1; x > ax; x-- {
				at /= c.Calls[x]
			}
			return at % c.Calls[ax]
		}
		for at := range calls {
			var args cpu.Args
			base := unsafe.Pointer(&args)
			for _, a := range c.Args {
				if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(args) {
					code.Close()
					t.Fatalf("%s: a call word at %d past the argument block", c.Name, a.Field)
				}
				slot := (*uint64)(unsafe.Add(base, a.Field))
				switch {
				case a.Coord > 0:
					*slot = uint64(coord(at, a.Coord-1))
					continue
				case a.Length || a.Value:
					*slot = uint64(a.Extent)
					continue
				case a.Moving:
					code.Close()
					t.Fatalf("%s: %s sits at an offset a caller would compute, which no caller of a table can", c.Name, a.Operand.Name)
				}
				words := a.Words
				switch o := a.Operand; {
				case words != nil:
				case tables[o.Name] != nil:
					words = tables[o.Name]
				default:
					code.Close()
					t.Fatalf("%s binds %+v, which the table's call site does not hold", c.Name, o)
				}
				keep = append(keep, words)
				if a.Word {
					*slot = uint64(words[a.Offset])
					continue
				}
				elem, off := 4, a.Offset
				if a.Elem > 0 {
					elem = a.Elem
				}
				for ax, per := range a.Per {
					off += coord(at, ax) * per
				}
				*slot = uint64(uintptr(unsafe.Pointer(&words[0]))) + uint64(elem*off)
			}
			code.Call(&args)
		}
		code.Close()
	}
	runtime.KeepAlive(keep)
	return unguard(t, out, n)
}

// ropeTabOracle is the table in float64, formed the way the float64 table the
// kernel replaced formed it: the same frequency recurrence rather than a fresh
// Pow per pair, so a difference here is about the kernel and not about two
// different sets of frequencies.
func ropeTabOracle(r nn.Rope, pos, npairs int) []float64 {
	out := make([]float64, 2*npairs)
	step := math.Pow(r.Base, -2/float64(r.NRot))
	ms := r.Scale
	if ms == 0 {
		ms = 1
	}
	f := 1.0
	for p := 0; p < npairs; p++ {
		g := f
		if p < len(r.Freqs) {
			g /= float64(r.Freqs[p])
		}
		th := float64(pos) * g
		out[2*p] = float64(float32(math.Cos(th) * ms))
		out[2*p+1] = float64(float32(math.Sin(th) * ms))
		f *= step
	}
	return out
}
