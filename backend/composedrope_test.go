package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestComposedRopeTableOverAxes runs the composer's rotary table on every
// backend present, five rows of it, over one position axis, contiguous
// sections and interleaved ones, at ragged pair counts, and holds each row
// to main's host table at the row's positions (nn.Rope.Table, and
// nn.Rope.TableAt over the runs main builds from the same sections) and to
// the float64 table of each pair at its axis's position. Each pair of a
// table over several axes must also be, to the bit, the composer's one-axis
// table's pair at the position of the pair's axis as main's runs assign it.
// A guard after the output must survive, and with two axes' positions
// swapped the table must not match.
func TestComposedRopeTableOverAxes(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) { composedRopeAxes(t, gpuRope(d), 5) })
	}
}

// composedRopeGuard is the word after a table's output and in every element
// before the table writes it: a quiet NaN, so an element left unwritten
// shows, and a word written past the table changes it.
const composedRopeGuard = 0x7fc0dead

func composedRopeAxes(t *testing.T, run ropeRunner, rows int) {
	// The device table's bound against the host's and the float64 table's
	// (TestRopeTableOnDeviceMatchesHost's).
	const bound = 3e-7
	cases := []struct {
		pairs       int
		sections    [4]int
		interleaved bool
	}{
		{pairs: 1}, {pairs: 7}, {pairs: 64},
		{pairs: 64, sections: [4]int{16, 24, 24, 0}},
		{pairs: 13, sections: [4]int{3, 5, 4, 1}},
		{pairs: 9, sections: [4]int{0, 2, 7, 0}},
		{pairs: 1, sections: [4]int{1, 0, 0, 0}},
		{pairs: 64, sections: [4]int{24, 20, 20, 0}, interleaved: true},
		{pairs: 11, sections: [4]int{2, 3, 2, 0}, interleaved: true},
		{pairs: 2, sections: [4]int{1, 1, 0, 0}, interleaved: true},
	}
	// Each row's position on each axis: the axes apart, and every digit of
	// the decomposition reached.
	at := make([][jit.RopeAxes]int, rows)
	for r := range at {
		at[r] = [jit.RopeAxes]int{37 + r, 1000 + 3*r, 70003 + 11*r, 1<<27 + 5*r}
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
			host := make([]float32, rows*2*tc.pairs)
			pos := []uint32{uint32(rows)}
			for r := range rows {
				cs := host[r*2*tc.pairs : (r+1)*2*tc.pairs]
				if !multi {
					rope.Table(cs, at[r][0])
					pos = append(pos, uint32(at[r][0]))
					continue
				}
				rope.TableAt(cs, at[r][:])
				for _, p := range at[r] {
					pos = append(pos, uint32(p))
				}
			}
			got := run(t, rope, tc.sections, tc.interleaved, rows, pos)
			worstHost, worstOracle := 0.0, 0.0
			for r := range rows {
				for p := range tc.pairs {
					ref := ropeTabOracle(rope, at[r][axis[p]], tc.pairs)
					for j := range 2 {
						i := r*2*tc.pairs + 2*p + j
						dh := math.Abs(float64(got[i]) - float64(host[i]))
						do := math.Abs(float64(got[i]) - ref[2*p+j])
						if math.IsNaN(dh) || math.IsNaN(do) {
							t.Fatalf("row %d pair %d is %v where the host's is %v", r, p, got[i], host[i])
						}
						worstHost, worstOracle = max(worstHost, dh), max(worstOracle, do)
					}
				}
			}
			if worstHost > bound || worstOracle > bound {
				t.Fatalf("the composer's table is %.3e from the host's and %.3e from the float64 table, past %.0e", worstHost, worstOracle, bound)
			}
			if !multi {
				t.Logf("one axis: %.3e from the host's table, %.3e from the float64 table", worstHost, worstOracle)
				ran++
				return
			}
			// Each row's position on each axis, the one-axis table of a row
			// there: pair p of row r is pair p of that table at p's axis.
			plain := rope
			plain.Runs = nil
			for r := range rows {
				var flat [jit.RopeAxes][]float32
				for a, p := range at[r] {
					flat[a] = run(t, plain, [4]int{}, false, 1, []uint32{1, uint32(p)})
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
			var used []int
			for a := range jit.RopeAxes {
				for _, b := range axis {
					if a == b {
						used = append(used, a)
						break
					}
				}
			}
			if len(used) < 2 {
				t.Logf("one axis turns every pair: %.3e from the host's table, %.3e from the float64 table", worstHost, worstOracle)
				ran++
				return
			}
			x, y := used[len(used)-2], used[len(used)-1]
			bad := append([]uint32(nil), pos...)
			for r := range rows {
				bad[1+r*jit.RopeAxes+x], bad[1+r*jit.RopeAxes+y] = bad[1+r*jit.RopeAxes+y], bad[1+r*jit.RopeAxes+x]
			}
			swapped := run(t, rope, tc.sections, tc.interleaved, rows, bad)
			apart := 0.0
			for i := range host {
				apart = max(apart, math.Abs(float64(swapped[i])-float64(host[i])))
			}
			if !(apart > 1e3*bound) {
				t.Fatalf("with axes %d and %d swapped the table is %.3e from the host's: the gate cannot see a wrong axis", x, y, apart)
			}
			t.Logf("%.3e from the host's table, %.3e from the float64 table; axes %d and %d swapped, %.3e", worstHost, worstOracle, x, y, apart)
			ran++
		})
	}
	if ran != len(cases) {
		t.Fatalf("%d of %d tables ran", ran, len(cases))
	}
}

// ropeRunner runs the composer's table for rope over sections at a step of
// rows on the positions pos (the live row count, then the rows'), the planes
// main folds (nn.Rope.TabPlanes) and the blocks the plan carries, and
// returns the table, its guard checked.
type ropeRunner func(t *testing.T, rope nn.Rope, sections [4]int, interleaved bool, rows int, pos []uint32) []float32

func composedRopeCodes(t *testing.T, d jit.Device, rope nn.Rope, sections [4]int, interleaved bool, rows int) []jit.Code {
	t.Helper()
	rt := graph.RopeTable{NRot: rope.NRot, Sections: sections, Interleaved: interleaved}
	_, codes, err := jit.Default().Form(jit.Request{Graph: rt.Build(), Part: jit.PartTable, Step: jit.Step{Rows: rows, Page: 64, Keys: 1}, Device: d})
	if err != nil {
		t.Fatal(err)
	}
	return codes
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
	return func(t *testing.T, rope nn.Rope, sections [4]int, interleaved bool, rows int, pos []uint32) []float32 {
		t.Helper()
		lanes := 1
		if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); ok {
			lanes = ir.SubgroupLanes
		}
		units := 0
		if m, ok := d.(backend.MultiprocessorDevice); ok {
			units = m.Multiprocessors()
		}
		codes := composedRopeCodes(t, jit.Device{Kind: jit.DeviceGPU, Slots: d.Slots(), Lanes: lanes, Units: units, ExactDiv: kernels.DivExact(d.API())},
			rope, sections, interleaved, rows)
		gg := newGPU(t, d)
		defer gg.free()
		n := rows * rope.NRot
		guarded := guardedTable(n)
		out := gg.up(u32bytes(guarded))
		tables := map[string]backend.Buf{"rope.pos": gg.up(u32bytes(pos)), "rope.pos.axes": gg.up(u32bytes(pos)),
			"rope.planes": gg.up(u32bytes(planeWords(rope))), "rope.cs": out}
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
				case o.Kind == jit.OperandTable && tables[o.Name] != nil:
					bufs = append(bufs, tables[o.Name])
				case o.Kind == jit.OperandScratch:
					bufs = append(bufs, gg.up(make([]byte, max(4, b.Bytes))))
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
