package composer

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// NGramRef is the n-gram hash as transformers' Qwen4ExpTextNGramEmbedding
// computes it, written out plainly in 64-bit words: the window's ids (none,
// or past a cut, read as eos), then per head the mixed product modulo its
// vocabulary plus its offset; next is the next position's predecessors.
// Where drop, head drop reads the head before it's row (the violation of a
// head dropped).
func NGramRef(win []uint32, words []uint64, per int, reset bool, drop int) (rows, next []uint32) {
	n := len(win)
	eos, mul := words[0], words[1:1+n]
	ctx := make([]uint64, n)
	ctx[0] = uint64(win[0])
	cut := false
	for s := 1; s < n; s++ {
		v := eos
		if win[s] != 0 {
			v = uint64(win[s]) - 1
		}
		if cut {
			v = eos
		}
		if reset && v == eos {
			cut = true
		}
		ctx[s] = v
	}
	for g := 2; g <= n; g++ {
		m := ctx[0] * mul[0]
		for j := 1; j < g; j++ {
			m ^= ctx[j] * mul[j]
		}
		for k := range per {
			h := (g-2)*per + k
			if h == drop && h > 0 {
				rows = append(rows, rows[h-1])
				continue
			}
			rows = append(rows, uint32(m%words[2+n+2*h]+words[1+n+2*h]))
		}
	}
	next = append([]uint32{win[0] + 1}, win[1:n-1]...)
	return rows, next
}

// NGramCase is a hash's shape and constants, and windows to run it over:
// multipliers past 2^44, vocabularies near 2^25 at offsets that sum past
// 2^28, windows with an end of sequence or none in each slot.
type NGramCase struct {
	N, Per int
	Reset  bool
	Words  []uint64
	Wins   [][]uint32
}

// NGramCases are the shapes the hash gates sweep: the release's (3-grams,
// eight heads each) and others, each with and without the reset.
func NGramCases() []NGramCase {
	rng := rand.New(rand.NewPCG(7, 13))
	const eos = 248044
	var cs []NGramCase
	for _, sh := range []struct{ n, per int }{{3, 8}, {2, 1}, {4, 3}, {3, 2}} {
		for _, reset := range []bool{true, false} {
			heads := (sh.n - 1) * sh.per
			w := []uint64{eos}
			for range sh.n {
				w = append(w, 2*rng.Uint64N(1<<44)+1)
			}
			var at uint64
			for range heads {
				voc := 20000003 + uint64(rng.IntN(1000))
				w = append(w, at, voc)
				at += voc
			}
			c := NGramCase{N: sh.n, Per: sh.per, Reset: reset, Words: w}
			for trial := range 40 {
				win := make([]uint32, sh.n)
				win[0] = uint32(rng.IntN(248320))
				for s := 1; s < sh.n; s++ {
					switch rng.IntN(4) {
					case 0:
					case 1:
						win[s] = eos + 1
					default:
						win[s] = uint32(rng.IntN(248320)) + 1
					}
				}
				if trial%5 == 0 {
					win[0] = eos
				}
				c.Wins = append(c.Wins, win)
			}
			cs = append(cs, c)
		}
	}
	return cs
}

// NGramGraph is the hash over c's windows written in expressions into
// words (graph.Builder.Words, MulXor, ModAdd): the window's ids, each g-gram's
// product and each head's row.
func NGramGraph(c NGramCase) (g *graph.Graph, win, rows graph.ValueID) {
	b := graph.New()
	win = b.In("win", c.N)
	b.G.Values[win].Elem = graph.ElemU32
	at := func(s int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{win}, []graph.Result{{Name: "w", Width: 1, Elem: graph.ElemU32}}, nil,
			graph.Attrs{Offset: s, Width: 1}, nil, nil)[0]
	}
	eos := graph.Word(uint32(c.Words[0]))
	ids := []graph.ValueID{at(0)}
	for s := 1; s < c.N; s++ {
		var e graph.WordTerms
		none := e.Fn(graph.MapEqual, graph.WordIn(0), graph.Word(0))
		in := []graph.ValueID{at(s)}
		if c.Reset && s > 1 {
			none = e.Fn(graph.MapAdd, none, e.Fn(graph.MapEqual, graph.WordIn(1), eos))
			in = append(in, ids[s-1])
		}
		e.Fn(graph.MapWhere, none, eos, e.Fn(graph.MapSub, graph.WordIn(0), graph.Word(1)))
		ids = append(ids, b.Words("id", 1, &e, in...))
	}
	gram := b.MulXor("g", nil, ids[0], c.Words[1])
	var hs []graph.ValueID
	for n := 2; n <= c.N; n++ {
		gram = b.MulXor("g", &gram, ids[n-1], c.Words[n])
		for k := range c.Per {
			h := (n-2)*c.Per + k
			hs = append(hs, b.ModAdd("h", gram, c.Words[2+c.N+2*h], c.Words[1+c.N+2*h]))
		}
	}
	rows = b.Emit(graph.OpConcat, hs, []graph.Result{{Name: "rows", Width: len(hs), Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
	b.Out(rows)
	return b.G, win, rows
}

// TestComposedCPUNGramHash forms NGramGraph -- expressions into words, wide
// products and remainders over 16-bit limbs -- on the CPU at each of
// NGramCases' shapes, a decode step's one row at a time and a prompt's step
// of every window at once, held word for word to NGramRef: the rows each
// head reads, guard words past them. The reset ignored, and a head reading
// the row of the head before it, must part from the form.
func TestComposedCPUNGramHash(t *testing.T) {
	ran, reset, dropped := 0, 0, 0
	for _, c := range NGramCases() {
		g, win, rows := NGramGraph(c)
		if err := graph.Verify(g, graph.Facts{}); err != nil {
			t.Fatal(err)
		}
		heads := (c.N - 1) * c.Per
		run := func(wins [][]uint32) []uint32 {
			R := len(wins)
			_, codes, err := Default().Form(Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: Step{Rows: R, Page: 64, Keys: 1},
				Device: cpuDevice(hostArm()), Apart: true})
			if err != nil {
				t.Fatalf("%d-grams x%d at %d rows: %v", c.N, c.Per, R, err)
			}
			vals := map[graph.ValueID][]uint32{win: slices.Concat(wins...), rows: make([]uint32, R*heads)}
			if err := runCPUGridValues(hostArm(), codes, g, vals, [3][]uint32{}, nil, func(graph.ValueID) int { return R }, nil); err != nil {
				t.Fatalf("%d-grams x%d at %d rows: %v", c.N, c.Per, R, err)
			}
			return vals[rows]
		}
		check := func(i int, w, gotRows []uint32) {
			wantRows, _ := NGramRef(w, c.Words, c.Per, c.Reset, -1)
			if !slices.Equal(gotRows, wantRows) {
				t.Fatalf("%d-grams x%d reset %v, window %d %v: rows %v, want %v", c.N, c.Per, c.Reset, i, w, gotRows, wantRows)
			}
			if r, _ := NGramRef(w, c.Words, c.Per, !c.Reset, -1); !slices.Equal(r, gotRows) {
				reset++
			}
			if r, _ := NGramRef(w, c.Words, c.Per, c.Reset, heads-1); heads > 1 && !slices.Equal(r, gotRows) {
				dropped++
			}
		}
		for i, w := range c.Wins {
			check(i, w, run([][]uint32{w}))
		}
		gr := run(c.Wins)
		for i, w := range c.Wins {
			check(i, w, gr[i*heads:(i+1)*heads])
		}
		ran++
	}
	if reset == 0 || dropped == 0 {
		t.Fatalf("the reset ignored parts %d windows and a head dropped %d: the gate cannot see them", reset, dropped)
	}
	t.Logf("%d shapes held; the reset ignored parts %d windows, a head dropped %d", ran, reset, dropped)
}
