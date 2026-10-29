package jit

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/cpu"
)

// SampleCase is one row the sampler's form is held to: its logits, the
// history (Hist[:H] the real ids), the penalty and the form's bounds.
type SampleCase struct {
	Name    string
	V, K, B int
	Logits  []float32
	Hist    []uint32
	Pen     float32
}

// SampleCases are the sampler gates' rows: ragged vocabularies on either
// side of the selection's threshold and of the candidates' count, logits on
// a coarse grid so that ties are many, every history length from none to
// the bound, and histories that repeat an id.
func SampleCases() []SampleCase {
	rng := rand.New(rand.NewSource(77))
	var cs []SampleCase
	for _, sh := range []struct{ v, k, b int }{{37, 7, 16}, {300, 40, 32}, {1531, 40, 32}, {4099, 256, 256}, {51200, 256, 256}, {65536, 256, 256}, {131072, 256, 256}, {151936, 256, 256}} {
		for _, h := range []int{0, 1, 5, sh.b / 2, sh.b} {
			l := make([]float32, sh.v)
			for i := range l {
				l[i] = float32(math.Round(rng.NormFloat64()*24)) / 8
			}
			if sh.v > 100 {
				l[3], l[sh.v-1] = float32(math.Inf(-1)), float32(math.Copysign(0, -1))
			}
			hist := make([]uint32, h)
			for i := range hist {
				if i > 0 && i%3 == 0 {
					hist[i] = hist[i-1]
					continue
				}
				hist[i] = uint32(rng.Intn(sh.v))
			}
			// The best ids of the row in the history too, so the penalty moves
			// what is kept.
			order := sampleOrder(l)
			for i := 0; i < h && i < 4; i++ {
				hist[i] = uint32(order[i])
			}
			cs = append(cs, SampleCase{Name: fmt.Sprintf("v%d/k%d/h%d", sh.v, sh.k, h), V: sh.v, K: sh.k, B: sh.b, Logits: l, Hist: hist, Pen: 1.3})
		}
	}
	return cs
}

// sampleOrder is the row's ids, the larger value first and the lower id on
// a tie.
func sampleOrder(l []float32) []int {
	ids := make([]int, len(l))
	for i := range ids {
		ids[i] = i
	}
	slices.SortStableFunc(ids, func(a, b int) int {
		switch {
		case l[a] > l[b]:
			return -1
		case l[a] < l[b]:
			return 1
		}
		return 0
	})
	return ids
}

// SampleWant is the host sampler's candidates for c: the penalty applied
// once per history entry in order, then the K best. neighbour applies each
// entry's penalty to the next id instead, a violation.
func SampleWant(c SampleCase, neighbour bool) ([]float32, []uint32) {
	l := slices.Clone(c.Logits)
	for _, id := range c.Hist {
		if neighbour {
			id = (id + 1) % uint32(c.V)
		}
		if v := l[id]; v > 0 {
			l[id] = v / c.Pen
		} else {
			l[id] = v * c.Pen
		}
	}
	k := min(c.K, c.V)
	order := sampleOrder(l)[:k]
	vals, ids := make([]float32, k), make([]uint32, k)
	for i, id := range order {
		vals[i], ids[i] = l[id], uint32(id)
	}
	return vals, ids
}

// SampleArgs is c's argument words as the tier writes them: the count, the
// penalty's bits, the ids, and past them an id no row has.
func SampleArgs(c SampleCase) []uint32 {
	w := make([]uint32, 2+c.B)
	w[0], w[1] = uint32(len(c.Hist)), math.Float32bits(c.Pen)
	for i := range c.B {
		w[2+i] = 0xffffffff
		if i < len(c.Hist) {
			w[2+i] = c.Hist[i]
		}
	}
	return w
}

// SampleMismatch is how many of the K places got and want part at, by
// value bits or id.
func SampleMismatch(gv []float32, gi []uint32, wv []float32, wi []uint32) int {
	bad := 0
	for i := range wv {
		if math.Float32bits(gv[i]) != math.Float32bits(wv[i]) || gi[i] != wi[i] {
			bad++
		}
	}
	return bad
}

// runCPUValues runs codes on the CPU over the graph's values: vals holds
// the inputs and receives every value a kernel writes, each followed by
// guard words checked after; every other storage is NaN until written.
func runCPUValues(codes []Code, g *graph.Graph, vals map[graph.ValueID][]uint32, planes [3][]uint32) error {
	return runCPUTables(codes, g, vals, planes, nil)
}

// runCPUTables is runCPUValues with the step's tables by name, each read
// and written in place.
func runCPUTables(codes []Code, g *graph.Graph, vals map[graph.ValueID][]uint32, planes [3][]uint32, tables map[string][]uint32) error {
	return runCPUGridValues(codes, g, vals, planes, tables, nil, nil)
}

// runCPUGridValues is runCPUTables over values of several rows, value v
// rowsOf(v) rows of its width (one where rowsOf is nil), and each weight
// and vector a matrix of byRole binds by its role, its words as the
// format the request named lays them. A scratch holds the most any code
// states of it. A value's or scratch's guard past it holds the widest
// value's most rows, so a form run at more rows than a value holds writes
// its guard and not the process's memory.
func runCPUGridValues(codes []Code, g *graph.Graph, vals map[graph.ValueID][]uint32, planes [3][]uint32, tables map[string][]uint32,
	rowsOf func(graph.ValueID) int, byRole map[jlm.Role][]uint32) error {
	guard := 16
	if rowsOf != nil {
		w, r := 0, 0
		for v := range g.Values {
			w, r = max(w, g.Values[v].Width), max(r, rowsOf(graph.ValueID(v)))
		}
		guard += w * r
	}
	extent := func(v graph.ValueID) int {
		if rowsOf == nil {
			return g.Values[v].Width
		}
		return g.Values[v].Width * rowsOf(v)
	}
	nan := math.Float32bits(float32(math.NaN()))
	poison := func(n int) []uint32 {
		w := make([]uint32, n)
		for i := range w {
			w[i] = nan
		}
		return w
	}
	// A guard is a signalling NaN, which no arithmetic writes: a row
	// computed from poison and stored past the end still shows.
	const fence = 0x7fa00001
	fenced := func(w []uint32) []uint32 {
		for range guard {
			w = append(w, fence)
		}
		return w
	}
	store := map[graph.ValueID][]uint32{}
	for v, w := range vals {
		store[v] = fenced(slices.Clone(w))
	}
	keep := [][]uint32{}
	scratch := map[Operand][]uint32{}
	scratchN := map[Operand]int{}
	for _, c := range codes {
		if c.CPU == nil {
			return fmt.Errorf("%s has no CPU code", c.Name)
		}
		code, err := mapComposed(c.CPU)
		if err != nil {
			return fmt.Errorf("mapping %s: %w", c.Name, err)
		}
		var args cpu.Args
		base := unsafe.Pointer(&args)
		var coords []CallArg
		var moving []movingArg
		for _, a := range c.Args {
			if a.Field < 0 || uintptr(a.Field)+8 > unsafe.Sizeof(args) {
				code.Close()
				return fmt.Errorf("%s: a call word at %d past the argument block", c.Name, a.Field)
			}
			slot := (*uint64)(unsafe.Add(base, a.Field))
			var words []uint32
			o := a.Operand
			switch {
			case a.Coord > 0:
				coords = append(coords, a)
				continue
			case a.Length || a.Value:
				*slot = argValue(a, spaceOf(&c, harnessWorkers))
				continue
			case a.Words != nil:
				words = append([]uint32(nil), a.Words...)
			case o.Kind == OperandValue:
				if store[o.Value] == nil {
					store[o.Value] = fenced(poison(extent(o.Value)))
				}
				words = store[o.Value]
			case (o.Kind == OperandWeight || o.Kind == OperandVector) && len(byRole[o.Role]) > 0:
				words = byRole[o.Role]
			case o.Kind == OperandWeight && o.Region >= 0 && o.Region < 3 && len(planes[o.Region]) > 0:
				words = planes[o.Region]
			case o.Kind == OperandTable && len(tables[o.Name]) > 0:
				words = tables[o.Name]
			case o.Kind == OperandScratch:
				// A pair's second head's scratch is the first's at its offset,
				// as the engine binds it.
				so := o
				so.Copy = 0
				if scratch[so] == nil {
					size := 0
					for _, cc := range codes {
						for _, b := range cc.Binds {
							bo := b.Operand
							bo.Copy = 0
							if bo == so && b.Bytes > 0 {
								size = max(size, int(b.Bytes+3)/4)
							}
						}
					}
					if size == 0 {
						size = 1 << 16
					}
					scratch[so] = fenced(poison(size))
					scratchN[so] = size
				}
				words = scratch[so]
			default:
				code.Close()
				return fmt.Errorf("%s binds %+v, which the sampler does not provide", c.Name, o)
			}
			keep = append(keep, words)
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
		callAll(code, &args, spaceOf(&c, harnessWorkers), coords, moving)
		code.Close()
		for so, w := range scratch {
			for i, x := range w[scratchN[so]:] {
				if x != fence {
					return fmt.Errorf("%s: scratch %s: word %#x written %d past its end", c.Name, so.Name, x, i)
				}
			}
		}
	}
	runtime.KeepAlive(keep)
	// An output no kernel binds was stored over the value its last writer
	// writes (the in-place pass): an input of its shape that nothing reads
	// after.
	for _, out := range g.Out {
		if _, ok := store[out]; ok {
			continue
		}
		for i := len(codes) - 1; i >= 0 && store[out] == nil; i-- {
			for _, b := range codes[i].Binds {
				if b.Writes && b.Operand.Kind == OperandValue && store[b.Operand.Value] != nil {
					store[out] = store[b.Operand.Value]
				}
			}
		}
	}
	for v, w := range store {
		n := extent(v)
		for i, x := range w[n:] {
			if x != fence {
				return fmt.Errorf("value %s: word %#x written %d past its end", g.Values[v].Name, x, i)
			}
		}
		vals[v] = w[:n]
	}
	return nil
}

// SampleNeighbour is s with every history id moved to the next: the
// penalty on the neighbouring id, a violation of the form.
func SampleNeighbour(s *graph.Graph) *graph.Graph {
	g := *s
	g.Values = slices.Clone(s.Values)
	g.Ops = slices.Clone(s.Ops)
	for i, o := range g.Ops {
		if o.Kind != graph.OpMap || o.Attr.Fn != graph.MapEqual {
			continue
		}
		// Each candidate's id less one against the history: the penalty
		// lands on the id after each entry's.
		cand := o.In[0]
		less := graph.ValueID(len(g.Values))
		g.Values = append(g.Values, graph.Value{Name: "cand.less", Width: g.Values[cand].Width})
		g.Ops[i].In = []graph.ValueID{less, o.In[1]}
		g.Ops = slices.Insert(g.Ops, i, graph.Op{Kind: graph.OpMap, In: []graph.ValueID{cand}, Out: []graph.ValueID{less},
			Attr: graph.Attrs{Fn: graph.MapAdd, Const: -1}})
		break
	}
	return &g
}

// SampleTieHigh reports whether two kept places hold one value, the lower
// id first: the ties a reference breaking them to the higher id would put
// the other way.
func SampleTieHigh(ids []uint32, vals []float32) bool {
	for i := 1; i < len(vals); i++ {
		if vals[i] == vals[i-1] && ids[i] > ids[i-1] {
			return true
		}
	}
	return false
}
