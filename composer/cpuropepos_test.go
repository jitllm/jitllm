package composer

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// ropeMark is a test's rotary mark (graph.MarkWords' layout).
type ropeMark [graph.MarkWords]uint32

// ropePosOracle is a row's four rotary positions at cache position p under
// a sequence's marks, as the engine lays a prompt's spans out: text after a
// mark at its base plus the distance, a grid's cell after its base, an
// XD-RoPE picture's (position, column, row, ordinal) inside its grid and
// its position outside.
func ropePosOracle(p uint32, marks []ropeMark) [4]uint32 {
	k := -1
	for i, m := range marks {
		if m[graph.MarkAt] <= p {
			k = i
		}
	}
	if k < 0 {
		return [4]uint32{p, p, p, p}
	}
	m := marks[k]
	j := p - m[graph.MarkAt]
	base := m[graph.MarkBase]
	switch m[graph.MarkKind] {
	case graph.MarkText:
		return [4]uint32{j + base, j + base, j + base, j + base}
	case graph.MarkGrid:
		h, w := m[graph.MarkH], m[graph.MarkW]
		t := base + j/(h*w)
		return [4]uint32{t, base + (j/w)%h, base + j%w, t}
	}
	ends, h, w := m[graph.MarkEnds], m[graph.MarkH], m[graph.MarkW]
	if j < ends || j-ends >= h*w {
		return [4]uint32{p, p, p, p}
	}
	jj := j - ends
	return [4]uint32{p, jj % w, jj / w, m[graph.MarkIndex]}
}

// TestComposedCPURopePos forms OpRopePos on the CPU and holds every row's
// four rotary positions to the oracle: rows before any mark, text after a
// picture, a picture grid's cells across frames, an XD-RoPE picture's grid
// and its text-like ends, three sequences' marks in one table, a sequence
// of no marks, and rows past the live ones untouched by the oracle. A table
// whose marks are read one too few, or the grid's axes swapped, must part.
func TestComposedCPURopePos(t *testing.T) {
	if !HostCPULowers() {
		t.Skip("no CPU lowering on this host")
	}
	const M = 4
	seqs := [][]ropeMark{
		nil,
		{{5, 9, graph.MarkText}, {10, 12, graph.MarkGrid, 2, 2, 3}, {22, 15, graph.MarkText}},
		{{4, 0, graph.MarkGridXD, 1, 2, 3, 1, 0}, {12, 0, graph.MarkGridXD, 1, 1, 2, 0, 1}},
	}
	rows := [][2]uint32{{3, 0}, {7, 1}, {11, 1}, {19, 1}, {25, 1}, {4, 2}, {7, 2}, {13, 2}}
	marks := make([]uint32, len(seqs)*(1+M*graph.MarkWords))
	off := make([]uint32, len(seqs))
	for s, ms := range seqs {
		o := s * (1 + M*graph.MarkWords)
		off[s] = uint32(o)
		marks[o] = uint32(len(ms))
		for i, m := range ms {
			copy(marks[o+1+i*graph.MarkWords:], m[:])
		}
	}
	// Every row at once, then each alone in a step of one row, the step a
	// decode's table runs.
	cpuRopePosAt(t, seqs, rows, marks, off, M, true)
	for i := range rows {
		t.Run(fmt.Sprintf("row %d alone", i), func(t *testing.T) {
			cpuRopePosAt(t, seqs, rows[i:i+1], marks, off, M, false)
		})
	}
	// A row far into its sequence under one XD-RoPE picture's mark of a
	// 2048 by 2048 grid, the step's one sequence of one mark: its four axes
	// apart, each in its own digit of a position.
	t.Run("a far row of a wide picture", func(t *testing.T) {
		const w, h, p = 1 << 11, 1 << 11, 1<<27 + 5
		jj := uint32(37*w + 1000)
		one := [][]ropeMark{{{p - jj, 0, graph.MarkGridXD, 1, h, w, 0, 70003}}}
		buf := append([]uint32{1}, one[0][0][:]...)
		cpuRopePosAt(t, one, [][2]uint32{{p, 0}}, buf, []uint32{0}, 1, false)
		// And at a rotary table's step on one slot, as an instance asks.
		cpuRopePosIn(t, one, [][2]uint32{{p, 0}}, buf, []uint32{0}, Step{Rows: 1, Page: 64, Keys: 1, Seqs: 1, Marks: 1}, 1, false)
	})
}

// cpuRopePosAt forms OpRopePos on the CPU at a step of len(rows) rows and
// holds each row's positions to the oracle; violations runs the marks read
// short, which must part.
func cpuRopePosAt(t *testing.T, seqs [][]ropeMark, rows [][2]uint32, marks, off []uint32, M int, violations bool) {
	cpuRopePosIn(t, seqs, rows, marks, off, Step{Rows: len(rows), Page: 16, Keys: 64, Seqs: len(seqs), Marks: M}, 4, violations)
}

// cpuRopePosIn is cpuRopePosAt at step st on a CPU of slots slots.
func cpuRopePosIn(t *testing.T, seqs [][]ropeMark, rows [][2]uint32, marks, off []uint32, st Step, slots int, violations bool) {
	t.Helper()
	R := len(rows)
	b := graph.New()
	y := b.Emit(graph.OpRopePos, nil, []graph.Result{{Name: "rope.pos", Width: 4, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(y)
	_, codes, err := Default().Form(Request{Graph: b.G, Step: st,
		Device: Device{Kind: DeviceCPU, Slots: slots, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
	if err != nil {
		t.Fatal(err)
	}
	desc := []uint32{uint32(R)}
	for _, r := range rows {
		desc = append(desc, r[0], off[r[1]])
	}
	for _, c := range []struct {
		name   string
		marks  func([]uint32) []uint32
		oracle func(s int) []ropeMark
		parts  bool
	}{
		{"as laid out", func(m []uint32) []uint32 { return m }, func(s int) []ropeMark { return seqs[s] }, false},
		{"a mark read short", func(m []uint32) []uint32 {
			m = append([]uint32(nil), m...)
			m[off[1]]--
			return m
		}, func(s int) []ropeMark { return seqs[s] }, true},
	} {
		if c.parts && !violations {
			continue
		}
		got := make([]uint32, R*4)
		for i := range got {
			got[i] = 0xdeadbeef
		}
		bufs := map[string][]uint32{"rope.rows": desc, "rope.marks": c.marks(marks), b.G.Values[y].Name: got}
		if err := runCPUCalls(codes, b.G, bufs); err != nil {
			t.Fatal(err)
		}
		bad := 0
		for i, r := range rows {
			want := ropePosOracle(r[0], c.oracle(int(r[1])))
			for a := range 4 {
				if got[4*i+a] != want[a] {
					bad++
					if !c.parts {
						t.Errorf("%s: row %d (position %d of sequence %d) axis %d: %d, want %d", c.name, i, r[0], r[1], a, got[4*i+a], want[a])
					}
				}
			}
		}
		if c.parts && bad == 0 {
			t.Errorf("%s: every row matches the oracle: the gate cannot see it", c.name)
		}
	}
}
