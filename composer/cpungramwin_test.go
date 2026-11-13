package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// ngramWindowOracle is the windows of a step's rows and each run's
// predecessors after it, as the definition steps a sequence: a row's
// window is its token (img where it names none), then its predecessors
// nearest first, a row's id plus one; a run starts from its slot's.
func ngramWindowOracle(toks []uint32, runs [][]int, held [][]uint32, n int, img uint32) (wins [][]uint32, after [][]uint32) {
	id := func(t uint32) uint32 {
		if t == ^uint32(0) {
			return img
		}
		return t
	}
	wins = make([][]uint32, len(toks))
	for j, run := range runs {
		pred := append([]uint32(nil), held[j]...)
		for _, r := range run {
			wins[r] = append([]uint32{id(toks[r])}, pred...)
			pred = append([]uint32{id(toks[r]) + 1}, pred[:n-2]...)
		}
		after = append(after, pred)
	}
	return wins, after
}

// TestComposedCPUNGramWindow executes OpNGramWindow on the CPU: a decode
// row's window from its slot, and a step of three sequences' interleaved
// rows, a picture's row among them, each run reading its predecessors from
// one pool's slot and writing them to another's. Every window and every
// run's predecessors after it are held to the definition, the slots no run
// names untouched; read as each row its own run, the windows part.
func TestComposedCPUNGramWindow(t *testing.T) {
	const n, off, span, slots = 4, 5, 12, 4
	const img = 999
	form := func(R int, keeps bool) (*graph.Graph, graph.ValueID, graph.ValueID, []Code) {
		b := graph.New()
		tok := b.In("tok", 1)
		b.G.Values[tok].Elem = graph.ElemU32
		at := b.G.StateVersion(graph.StateRecState)
		win := b.Emit(graph.OpNGramWindow, []graph.ValueID{tok}, []graph.Result{{Name: "win", Width: n, Elem: graph.ElemU32}}, nil,
			graph.Attrs{Taps: n, Offset: off, Span: span, Words: []uint64{img}, Keeps: keeps},
			[]graph.StateRef{{State: graph.StateRecState, Version: at}}, []graph.StateRef{{State: graph.StateRecState, Version: at + 1}})[0]
		b.Out(win)
		if err := graph.Verify(b.G, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		_, codes, err := Default().Form(Request{Graph: b.G, Step: Step{Rows: R, Page: 1, Keys: 1},
			Device: Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}})
		if err != nil {
			t.Fatal(err)
		}
		return b.G, tok, win, codes
	}
	pool := func(seed uint32) []float32 {
		p := make([]float32, slots*span)
		for i := range p {
			p[i] = math.Float32frombits(seed + uint32(i)*7)
		}
		return p
	}
	slotPreds := func(p []float32, s int) []uint32 {
		out := make([]uint32, n-1)
		for k := range out {
			out[k] = math.Float32bits(p[s*span+off+k])
		}
		return out
	}
	// run steps the window; with keep set it is formed keeping its rows, the
	// keep table keep.
	run := func(R int, toks []uint32, desc []uint32, in, out, keep []float32) []uint32 {
		g, tok, win, codes := form(R, keep != nil)
		got := make([]uint32, R*n)
		bufs := map[string][]uint32{g.Values[tok].Name: append([]uint32(nil), toks...), g.Values[win].Name: got,
			"rec.state": f32Words(in), "rec.state.out": f32Words(out), "rec.rows": desc}
		if keep != nil {
			bufs["rec.state.keep"] = f32Words(keep)
		}
		if err := runCPUCalls(hostArm(), codes, g, bufs); err != nil {
			t.Fatal(err)
		}
		return got
	}
	check := func(name string, got []uint32, wins [][]uint32, out []float32, before []float32, written map[int][]uint32) {
		for r, w := range wins {
			for k := range n {
				if got[r*n+k] != w[k] {
					t.Fatalf("%s: row %d word %d is %d, the definition's %d (windows %v, the definition's %v)", name, r, k, got[r*n+k], w[k], got, wins)
				}
			}
		}
		for s := range slots {
			for i := range span {
				want := math.Float32bits(before[s*span+i])
				if p, ok := written[s]; ok && i >= off && i < off+n-1 {
					want = p[i-off]
				}
				if have := math.Float32bits(out[s*span+i]); have != want {
					t.Fatalf("%s: slot %d word %d is %#x, want %#x", name, s, i, have, want)
				}
			}
		}
	}

	// A decode row reads slot 2 and writes slot 1.
	t.Run("decode", func(t *testing.T) {
		in, outPool := pool(0x3f800000), pool(0x40000000)
		before := append([]float32(nil), outPool...)
		toks := []uint32{41}
		wins, after := ngramWindowOracle(toks, [][]int{{0}}, [][]uint32{slotPreds(in, 2)}, n, img)
		got := run(1, toks, []uint32{1, 2, 1}, in, outPool, nil)
		check("decode", got, wins, outPool, before, map[int][]uint32{1: after[0]})
	})

	// Three sequences' rows interleaved, a picture's row among them.
	const R = 6
	runs := [][]int{{1, 3, 0}, {2, 5}, {4}}
	rin, rout := []int{0, 3, 2}, []int{1, 3, 0}
	toks := []uint32{11, 12, 13, ^uint32(0), 15, 16}
	in, outPool := pool(0x3f000000), pool(0x41000000)
	before := append([]float32(nil), outPool...)
	held := [][]uint32{slotPreds(in, rin[0]), slotPreds(in, rin[1]), slotPreds(in, rin[2])}
	wins, after := ngramWindowOracle(toks, runs, held, n, img)
	got := run(R, toks, scanDesc(runs, R, rin, rout), in, outPool, nil)
	written := map[int][]uint32{}
	for j := range runs {
		written[rout[j]] = after[j]
	}
	check("runs", got, wins, outPool, before, written)

	// Formed keeping its rows, each entry's slot of the keep table holds its
	// run's predecessors after the entry's row; a decode row's is entry 0's.
	t.Run("keeps", func(t *testing.T) {
		nan := float32(math.NaN())
		keepOf := func(R int) []float32 {
			k := make([]float32, R*span)
			for i := range k {
				k[i] = nan
			}
			return k
		}
		holds := func(name string, keep []float32, entries [][]uint32) {
			for e, pred := range entries {
				for k := range n - 1 {
					if have := math.Float32bits(keep[e*span+off+k]); have != pred[k] {
						t.Fatalf("%s: entry %d predecessor %d is %#x, want %#x", name, e, k, have, pred[k])
					}
				}
			}
			for i := len(entries) * span; i < len(keep); i++ {
				if !math.IsNaN(float64(keep[i])) {
					t.Fatalf("%s: the form kept %#x at word %d, past the step's %d entries", name, math.Float32bits(keep[i]), i, len(entries))
				}
			}
		}
		din, dout := pool(0x3f800000), pool(0x40000000)
		dbefore := append([]float32(nil), dout...)
		dtoks := []uint32{41}
		dwins, dafter := ngramWindowOracle(dtoks, [][]int{{0}}, [][]uint32{slotPreds(din, 2)}, n, img)
		dkeep := keepOf(1)
		// Formed keeping, the one row takes the runs' form, which reads the
		// whole run descriptor.
		check("decode keeps", run(1, dtoks, scanDesc([][]int{{0}}, 1, []int{2}, []int{1}), din, dout, dkeep), dwins, dout, dbefore,
			map[int][]uint32{1: dafter[0]})
		holds("decode keeps", dkeep, dafter)

		kin, kout := pool(0x3f000000), pool(0x41000000)
		kbefore := append([]float32(nil), kout...)
		keep := keepOf(R)
		check("runs keep", run(R, toks, scanDesc(runs, R, rin, rout), kin, kout, keep), wins, kout, kbefore, written)
		var entries [][]uint32
		for j, rr := range runs {
			for r := range rr {
				_, a := ngramWindowOracle(toks, [][]int{rr[:r+1]}, [][]uint32{held[j]}, n, img)
				entries = append(entries, a[0])
			}
		}
		holds("runs keep", keep, entries)
	})

	// The same rows as runs of their own read only their slots.
	alone := [][]int{{1}, {3}, {0}, {2}, {5}, {4}}
	aheld := [][]uint32{held[0], held[0], held[0], held[1], held[1], held[2]}
	lone, _ := ngramWindowOracle(toks, alone, aheld, n, img)
	same := true
	for r := range R {
		for k := range n {
			same = same && lone[r][k] == got[r*n+k]
		}
	}
	if same {
		t.Fatal("the windows read as each row its own run: the gate cannot see a run's earlier rows")
	}
}
