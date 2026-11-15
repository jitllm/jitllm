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
// a sequence's marks, as OpRopePos states them: the last mark at or before
// p, each axis its base, plus its position times p, plus its row's index
// past the mark's skip divided and wrapped times its step, inside the
// mark's span, and p outside it and before every mark.
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
	if j < m[graph.MarkSkip] || m[graph.MarkCount] != 0 && j-m[graph.MarkSkip] >= m[graph.MarkCount] {
		return [4]uint32{p, p, p, p}
	}
	j -= m[graph.MarkSkip]
	var out [4]uint32
	for a := range out {
		w := func(i int) uint32 { return m[graph.MarkAxis(a, i)] }
		q := j
		if w(graph.MarkDiv) > 1 {
			q /= w(graph.MarkDiv)
		}
		if w(graph.MarkMod) > 0 {
			q %= w(graph.MarkMod)
		}
		out[a] = w(graph.MarkBase) + w(graph.MarkPos)*p + w(graph.MarkMul)*q
	}
	return out
}

// textMark, gridMark and xdMark are the marks of the prompts the engine lays
// out: text from at at base onwards; a grid of h rows of w cells from base,
// its frames stride apart; a picture's h rows of w cells after ends rows at
// their positions, its rows (position, column, row, ordinal idx).
func textMark(at, base uint32) (m ropeMark) {
	m[graph.MarkAt] = at
	for a := range 4 {
		m[graph.MarkAxis(a, graph.MarkBase)], m[graph.MarkAxis(a, graph.MarkMul)] = base, 1
	}
	return m
}

func gridMark(at, base, h, w, stride uint32) (m ropeMark) {
	m[graph.MarkAt] = at
	for a, s := range [4][3]uint32{{stride, h * w, 0}, {1, w, h}, {1, 1, w}, {stride, h * w, 0}} {
		m[graph.MarkAxis(a, graph.MarkBase)] = base
		m[graph.MarkAxis(a, graph.MarkMul)], m[graph.MarkAxis(a, graph.MarkDiv)], m[graph.MarkAxis(a, graph.MarkMod)] = s[0], s[1], s[2]
	}
	return m
}

func xdMark(at, ends, h, w, idx uint32) (m ropeMark) {
	m[graph.MarkAt], m[graph.MarkSkip], m[graph.MarkCount] = at, ends, h*w
	m[graph.MarkAxis(0, graph.MarkPos)] = 1
	m[graph.MarkAxis(1, graph.MarkMul)], m[graph.MarkAxis(1, graph.MarkMod)] = 1, w
	m[graph.MarkAxis(2, graph.MarkMul)], m[graph.MarkAxis(2, graph.MarkDiv)] = 1, w
	m[graph.MarkAxis(3, graph.MarkBase)] = idx
	return m
}

// TestComposedCPURopePos forms OpRopePos on the CPU and holds every row's
// four rotary positions to the oracle: rows before any mark, text after a
// picture, a picture grid's cells across frames, an XD-RoPE picture's grid
// and its text-like ends, three sequences' marks in one table, a sequence
// of no marks, and rows past the live ones untouched by the oracle. A table
// whose marks are read one too few, or the grid's axes swapped, must part.
func TestComposedCPURopePos(t *testing.T) {
	if !LowersCPU(uint64(cpu.Probe().ISA())) {
		t.Skip("no CPU lowering on this host")
	}
	const M = 4
	seqs := [][]ropeMark{
		nil,
		{textMark(5, 9), gridMark(10, 12, 2, 3, 3), textMark(22, 15)},
		{xdMark(4, 1, 2, 3, 0), xdMark(12, 0, 1, 2, 1)},
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
	cpuRopePosAt(t, hostArm(), seqs, rows, marks, off, M, true)
	for i := range rows {
		t.Run(fmt.Sprintf("row %d alone", i), func(t *testing.T) {
			cpuRopePosAt(t, hostArm(), seqs, rows[i:i+1], marks, off, M, false)
		})
	}
	// A row far into its sequence under one XD-RoPE picture's mark of a
	// 2048 by 2048 grid, the step's one sequence of one mark: its four axes
	// apart, each in its own digit of a position.
	t.Run("a far row of a wide picture", func(t *testing.T) {
		const w, h, p = 1 << 11, 1 << 11, 1<<27 + 5
		jj := uint32(37*w + 1000)
		one := [][]ropeMark{{xdMark(p-jj, 0, h, w, 70003)}}
		buf := append([]uint32{1}, one[0][0][:]...)
		cpuRopePosAt(t, hostArm(), one, [][2]uint32{{p, 0}}, buf, []uint32{0}, 1, false)
		// And at a rotary table's step on one slot, as an instance asks.
		cpuRopePosIn(t, hostArm(), one, [][2]uint32{{p, 0}}, buf, []uint32{0}, Step{Rows: 1, Page: 64, Keys: 1, Seqs: 1, Marks: 1}, 1, false)
	})
}

// cpuRopePosAt forms OpRopePos on the CPU at a step of len(rows) rows and
// holds each row's positions to the oracle; violations runs the marks read
// short, which must part.
func cpuRopePosAt(t *testing.T, cpuOn cpuArm, seqs [][]ropeMark, rows [][2]uint32, marks, off []uint32, M int, violations bool) {
	cpuRopePosIn(t, cpuOn, seqs, rows, marks, off, Step{Rows: len(rows), Page: 16, Keys: 64, Seqs: len(seqs), Marks: M}, 4, violations)
}

// cpuRopePosIn is cpuRopePosAt at step st on a CPU of slots slots.
func cpuRopePosIn(t *testing.T, cpuOn cpuArm, seqs [][]ropeMark, rows [][2]uint32, marks, off []uint32, st Step, slots int, violations bool) {
	t.Helper()
	R := len(rows)
	b := graph.New()
	y := b.Emit(graph.OpRopePos, nil, []graph.Result{{Name: "rope.pos", Width: 4, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(y)
	_, codes, err := Default().Form(Request{Graph: b.G, Step: st,
		Device: Device{Kind: DeviceCPU, Slots: slots, Lanes: 8, ISA: uint64(cpuOn.tg.Features().ISA()), ExactDiv: true}})
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
		if err := runCPUCalls(cpuOn, codes, b.G, bufs); err != nil {
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
