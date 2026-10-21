package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// API is the target's kernel language.
type API uint8

const (
	APINone API = iota
	APIPTX
	APISPIRV
	APIMSL
	APIAMDGCN
)

// APIOf is the API backend.Device.API names, APINone for another.
func APIOf(name string) API {
	switch name {
	case "ptx":
		return APIPTX
	case "spirv":
		return APISPIRV
	case "msl":
		return APIMSL
	case "amdgcn":
		return APIAMDGCN
	}
	return APINone
}

// Target is the device a group is planned for: what the passes ask of it.
type Target struct {
	API API
	// ISA is the instruction set, as an id.
	ISA ir.ISA
	// Lanes is the subgroup width the device guarantees, 0 for none.
	Lanes int
}

// IntMMA reports whether the target has an int8 matrix instruction (ptx's
// mma.sync on s8, sm_75 on).
func (t Target) IntMMA() bool { return t.ISA.Family == ir.FamilyPTX && t.ISA.Version >= 75 }

// Step is the kind of step a group runs in.
type Step uint8

const (
	StepDecode Step = iota // one row a sequence
	StepChunk              // a prompt chunk of the scratch's rows
	StepRagged             // a step of several sequences' rows
)

// Facts are what a plan is made from beyond the group's structure: the
// target, the step, and the tier's pins (and a tuner's measured choices),
// each 0 where the pass's rule decides.
type Facts struct {
	Target Target
	Step   Step
	// Over a chunk: tokens and rows a thread, and the k split.
	Tok, RowTile, Split int
	// At one row: the k split and rows a thread.
	DecodeSplit, DecodeRowt int
	// Over a ragged step: tokens and rows a thread, and the k split.
	RagTok, RagRowT, RagSplit int
	// RagGroup runs a ragged step's few tokens in one thread at decode's
	// split, reduced in the workgroup (Schedule.GroupSplit) where the split
	// and the rows allow it.
	RagGroup bool
	// DecodeGroup reduces decode's k split in the workgroup
	// (Schedule.GroupSplit) rather than through planes.
	DecodeGroup bool
	// The warp tile of the int8 matrix instruction: tiles of 16 elements
	// and of 8 rows a warp; NoMMA, the caller's refusal of the instruction.
	MMAMT, MMANT int
	NoMMA        bool
	// A row group's width and combine where the tier states them: RowGroup
	// threads (0 for the rule's), and Tree to combine a reduction through
	// threadgroup memory where the target could butterfly it.
	RowGroup int
	Tree     bool
	// GroupRows takes a lane row from the workgroup and the thread
	// (Schedule.GroupRows).
	GroupRows bool
}

// Planned is one kernel of a plan: its group, its schedule, its buffers by
// the planned group's index (-1 for a split's planes), and its threads.
type Planned struct {
	G       *Row
	S       Schedule
	Bufs    []int
	Threads int
}

// Plan is the kernels group g runs as, for the step and target in f:
//
//   - a contraction -- a group whose one store reads a sum over loop 0 whose
//     body reads loads that do not depend on the row (a weight) beside loads
//     that do (an operand) -- runs flat, a tile of rows and elements a thread
//     and loop 0 split between threads (contractSchedule), the split a
//     rewrite of the group (SplitFold);
//   - a group with a windowed reduction and no stages runs flat, a thread an
//     element, 128 to a workgroup (its windows meet in the workgroup); one
//     with stages runs a workgroup a row, its stages' windows within it;
//   - a group with no reduction runs flat, 128 threads to a workgroup;
//   - a group with a row reduction runs a workgroup a row, 1024 threads.
func (g *Row) Plan(f Facts) ([]Planned, error) {
	all := make([]int, len(g.Bufs))
	for i := range all {
		all[i] = i
	}
	if g.windowed() && len(g.Stages) == 0 {
		// Windows meet across a workgroup's consecutive threads: flat, a
		// thread an element, 128 to a group.
		return []Planned{{G: g, S: Schedule{Group: 128, Flat: true, Target: f.Target}, Bufs: all,
			Threads: g.Rows * g.Width}}, nil
	}
	shuffles, shared := false, false
	for _, n := range g.Nodes {
		shuffles = shuffles || n.Kind == NShuffle
	}
	for i := range g.Bufs {
		shared = shared || g.shared(i)
	}
	if g.reduces() && g.Width <= f.Target.Lanes && g.Width&(g.Width-1) == 0 || g.loopReduces() || shuffles && !(shared && g.Width > f.Target.Lanes) {
		// A row no wider than a subgroup, or a reduction a loop's body reads
		// (a value of the body), runs on one subgroup's lanes, the reduction
		// their butterfly.
		s := Schedule{Group: 128, Flat: true, Lanes: g.Width, Target: f.Target, GroupRows: f.GroupRows}
		if f.RowGroup > 0 {
			s.Group = f.RowGroup
		}
		if shared {
			// A row's arrays are its workgroup's.
			s.Group = max(g.Width, 32)
		}
		return []Planned{{G: g, S: s, Bufs: all, Threads: g.Rows * g.Width}}, nil
	}
	if shared && g.Width <= 1024 {
		// Elements that meet in workgroup arrays: a row a workgroup.
		return []Planned{{G: g, S: Schedule{Group: (g.Width + 31) / 32 * 32, Flat: true, WorkRow: true, Target: f.Target}, Bufs: all,
			Threads: g.Rows * ((g.Width + 31) / 32 * 32)}}, nil
	}
	if L := f.Target.Lanes; L > 1 && g.Width == 1 && g.Rows > 1 && g.foldsReadAfter() {
		// An item whose run-time loops fold what is read after them: its
		// loops' trips over a subgroup's lanes, the folds' lanes combined.
		s := Schedule{Group: L, Flat: true, Lanes: L, LaneLoops: true, Target: f.Target, GroupRows: f.GroupRows}
		if f.RowGroup > 0 {
			s.Group = f.RowGroup
		}
		return []Planned{{G: g, S: s, Bufs: all, Threads: g.Rows * L}}, nil
	}
	if g.product() {
		return g.PlanAt(g.productSchedule(f))
	}
	if !g.contraction() && !g.reduces() && len(g.Stages) == 0 {
		return []Planned{{G: g, S: Schedule{Group: 128, Flat: true, Target: f.Target}, Bufs: all,
			Threads: g.Rows * g.Width}}, nil
	}
	if !g.contraction() {
		s := rowSchedule(g, f.Target)
		if f.RowGroup > 0 {
			s.Group = f.RowGroup
		}
		if f.Tree {
			s.Warp = false
		}
		s.Target = f.Target
		return []Planned{{G: g, S: s, Bufs: all, Threads: g.Rows}}, nil
	}
	s := g.contractSchedule(f)
	s.Target = f.Target
	return g.PlanAt(s)
}

// PlanAt is the kernels a contraction runs as under schedule s: s itself,
// or under a split the rewrite's two groups (SplitFold).
func (g *Row) PlanAt(s Schedule) ([]Planned, error) {
	all := make([]int, len(g.Bufs))
	for i := range all {
		all[i] = i
	}
	threads := g.Rows / max(s.Tile, 1) * (g.Width / max(s.TileCols, 1))
	if s.MMA {
		threads = g.Width / (16 * s.MT) * (g.Rows / (8 * s.NT)) * ir.SubgroupLanes
	}
	if s.Split <= 1 {
		return []Planned{{G: g, S: s, Bufs: all, Threads: threads}}, nil
	}
	if s.GroupSplit {
		r, err := g.GroupFold(s.Split, s.Group)
		if err != nil {
			return nil, err
		}
		return []Planned{{G: r, S: s, Bufs: all, Threads: g.Width / (s.Group / s.Split) * s.Group}}, nil
	}
	first, second, b1, b2, err := g.SplitFold(s.Split)
	if err != nil {
		return nil, err
	}
	return []Planned{
		{G: first, S: s, Bufs: b1, Threads: threads * s.Split},
		{G: second, S: Schedule{Group: s.Group, Flat: true, Target: s.Target}, Bufs: b2, Threads: g.Rows * g.Width},
	}, nil
}

// loopReduces reports whether a reduction reads a loop's iterator or carried value.
func (g *Row) loopReduces() bool {
	for c := range g.Loops {
		for i, v := range g.varies(c) {
			if v && g.Nodes[i].Kind == NReduce {
				return true
			}
		}
	}
	return false
}

func (g *Row) reduces() bool {
	for _, n := range g.Nodes {
		if n.Kind == NReduce {
			return true
		}
	}
	return false
}

// windowed reports whether g holds a windowed reduction (NWindow).
func (g *Row) windowed() bool {
	for _, n := range g.Nodes {
		if n.Kind == NWindow {
			return true
		}
	}
	return false
}

// contraction reports whether g is a contraction: one store, reading a sum
// over loop 0, whose body reads an indexed load that depends on no row and
// one that does.
func (g *Row) contraction() bool {
	if len(g.Stores) != 1 || len(g.Loops) == 0 || g.Q8 != nil || !g.sumsLoop0(g.Stores[0].Val) {
		return false
	}
	for _, nd := range g.Nodes {
		if nd.Kind == NReduce {
			return false
		}
	}
	// A weight is what reads no row -- or the row only as its choice of
	// sheet -- and an operand what reads the row.
	vary := g.varies(0)
	for _, rowDep := range [][]bool{g.rowDeps(), g.selectedRowDeps(), g.rowDepsPast(g.spanStride(), 0)} {
		weight, operand := false, false
		for i, nd := range g.Nodes {
			if nd.Kind != NLoad || !nd.Indexed || !vary[i] {
				continue
			}
			if rowDep[i] {
				operand = true
			} else {
				weight = true
			}
		}
		if weight && operand {
			return true
		}
	}
	return false
}

// contractSchedule is a contraction's schedule, Width outputs by Rows
// tokens over loop 0's trips (sub-blocks of the contracted axis):
//
//   - at one row, the pinned or tuned k split and rows a thread, else the
//     split doubling while the grid holds fewer than two small cards' resident
//     threads (2*30720), at most 32, and it divides the trips;
//   - over several tokens where the int8 matrix instruction takes the
//     contraction (mmaShaped), its warp tile of 4x4 instructions (Facts' pin)
//     halved until it divides;
//   - over a ragged step the int8 matrix instruction does not take
//     (mmaShaped), eight tokens and two rows a thread (four from 64 tokens),
//     each halved until it divides (swept on a Volta-class card);
//   - over a chunk, the scratch's tokens a thread and rows a thread where they
//     divide, else two, else one;
//   - over several tokens, the k split doubling until the grid holds 1<<16
//     threads, at most 64, halved until it divides the trips.
func (g *Row) contractSchedule(f Facts) Schedule {
	s := Schedule{Group: 128, Flat: true}
	trips := g.Loops[0]
	if g.Rows == 1 || f.Step == StepDecode {
		split := f.DecodeSplit
		if split < 1 {
			const slots = 30720 // a small Ampere card's resident threads: 20 SMs x 1536
			split = 1
			for split < 32 && g.Width*split < 2*slots && trips%(split*2) == 0 {
				split *= 2
			}
		}
		for split > 1 && trips%split != 0 {
			split /= 2
		}
		rowt := max(f.DecodeRowt, 1)
		if g.Width%rowt != 0 {
			rowt = 1
		}
		s.TileCols, s.Split = rowt, split
		if gw := min(max(32*split, 128), 1024); f.DecodeGroup && split > 1 && rowt == 1 && gw%split == 0 && g.Width%(gw/split) == 0 {
			s.Group, s.Tile, s.GroupSplit = gw, g.Rows, true
		}
		return s
	}
	if d := g.rowGroup(); d > 1 && g.Rows%d == 0 {
		// A weight selected per aligned group of rows: the group a thread,
		// the selection read once, an element a thread and no split.
		s.Tile, s.TileCols, s.Split = d, 1, 1
		return s
	}
	if !f.NoMMA && g.mmaShaped(f.Target) {
		// The matrix instruction's warp tile, 4x4 unless pinned, each
		// halved until it divides (wider runs out of registers, narrower
		// reads the weights more often: measured on sm_86).
		mt, nt := 4, 4
		if f.MMAMT > 0 && f.MMANT > 0 {
			mt, nt = f.MMAMT, f.MMANT
		}
		for mt > 1 && g.Width%(16*mt) != 0 {
			mt /= 2
		}
		for nt > 1 && g.Rows%(8*nt) != 0 {
			nt /= 2
		}
		s.MMA, s.MT, s.NT = true, mt, nt
		return s
	}
	if f.Step == StepRagged && f.RagGroup {
		split := max(f.DecodeSplit, 1)
		gw := min(max(32*split, 128), 1024)
		switch {
		case split == 1:
			s.Tile = g.Rows
			return s
		case trips%split == 0 && gw%split == 0 && g.Width%(gw/split) == 0:
			s.Group, s.Tile, s.Split, s.GroupSplit = gw, g.Rows, split, true
			return s
		}
	}
	var tok, rowt, split int
	switch {
	case f.Step == StepRagged:
		tok, rowt = min(g.Rows, 8), 2
		if g.Rows >= 64 {
			rowt = 4
		}
		if f.RagTok > 0 && g.Rows%f.RagTok == 0 {
			tok = f.RagTok
		}
		if f.RagRowT > 0 {
			rowt = f.RagRowT
		}
		split = f.RagSplit
	default:
		tok = max(f.Tok, 1)
		rowt = f.RowTile
		if rowt < 1 {
			rowt = 1
			if g.Width%2 == 0 {
				rowt = 2
			}
		}
		split = f.Split
		s.RowStride = g.spanStride()
	}
	for tok > 1 && (g.Rows%(tok*max(s.RowStride, 1)) != 0 || g.SpanGroup > 0 && g.SpanGroup%tok != 0) {
		tok /= 2
	}
	for rowt > 1 && g.Width%rowt != 0 {
		rowt /= 2
	}
	if split < 1 {
		split = 1
		for g.Width/rowt*(g.Rows/tok)*split < 1<<16 && split < 64 {
			split *= 2
		}
	}
	for split > 1 && trips%split != 0 {
		split /= 2
	}
	s.Tile, s.TileCols, s.Split = tok, rowt, split
	if len(g.Ranges) > 0 && g.Ranges[0].ok && g.tripOf(0) >= 0 {
		// A contraction over a run-time range of a coordinate: blocks of
		// eight places, unsplit.
		s.Block, s.Split = keyBlock, 1
	}
	return s
}

// mmaShaped reports whether the int8 matrix instruction can run g on t: an
// integer contraction (its loop's body a chain of NDot4) on a target that
// has the instruction, its outputs a whole number of the instruction's 16
// rows and its tokens of its 8 columns.
func (g *Row) mmaShaped(t Target) bool {
	if !t.IntMMA() || g.Width%16 != 0 || g.Rows%8 != 0 {
		return false
	}
	for _, n := range g.Nodes {
		if n.Kind == NDot4 {
			return true
		}
	}
	return false
}

// rowSchedule is a row group's schedule: a group with a row reduction runs
// 1024 threads whatever its width (every reduction one workgroup's), its
// reductions a shuffle butterfly on a ptx target guaranteeing 32 lanes, a
// tree elsewhere; any other the narrowest power of two over the row, at
// most 256.
func rowSchedule(g *Row, t Target) Schedule {
	for _, n := range g.Nodes {
		if n.Kind == NReduce {
			return Schedule{Group: 1024, Warp: t.API == APIPTX && t.Lanes == ir.SubgroupLanes}
		}
	}
	w := 32
	for w < g.Width && w < 256 {
		w *= 2
	}
	return Schedule{Group: w}
}

// String is the plan's kernels, for a dump.
func (p Planned) String() string {
	return fmt.Sprintf("%d nodes, %+v, %d threads", len(p.G.Nodes), p.S, p.Threads)
}

// product reports a group of no loop and no reduction whose one store
// reads products of an operand that reads the row and not the element by
// one that reads the element and not the row, past what its spans hold
// uniform over a group of rows: a tile of rows by elements a thread reads
// each operand once for the tile.
func (g *Row) product() bool {
	if len(g.Stores) != 1 || len(g.Loops) != 0 || g.reduces() || g.Q8 != nil || len(g.Stages) != 0 {
		return false
	}
	row, col := g.rowDepsPast(g.spanStride(), 0), g.colDeps()
	operand, weight := false, false
	for i, nd := range g.Nodes {
		if nd.Kind != NLoad || !nd.Indexed || g.typeOf(i) == ir.U32 {
			continue
		}
		operand = operand || row[i] && !col[i]
		weight = weight || col[i] && !row[i]
	}
	return operand && weight
}

// productSchedule is a product's tile: the step's rows and elements a
// thread, halved until they divide; rows spaced as its spans' rows are (a
// row's span is the row's over a divisor, every row the divisor apart
// sharing it), elements spaced a tile's count apart where an element's
// operand is gathered through a table (neighbouring threads then read
// neighbouring places), adjacent where it is an affine place.
func (g *Row) productSchedule(f Facts) Schedule {
	s := Schedule{Group: 128, Flat: true, Target: f.Target, RowStride: g.spanStride(), ColStride: 1}
	tok, rowt := max(f.Tok, 1), max(f.RowTile, 1)
	for tok > 1 && (g.Rows%(tok*s.RowStride) != 0 || g.SpanGroup > 0 && g.SpanGroup%tok != 0) {
		tok /= 2
	}
	for rowt > 1 && g.Width%rowt != 0 {
		rowt /= 2
	}
	s.Tile, s.TileCols = tok, rowt
	if rowt > 1 && g.gathersByColumn() {
		s.ColStride = g.Width / rowt
	}
	return s
}

// spanStride is how far apart the rows sharing a span are: d where every
// span's row reads the group's row only as the row over d, else 1.
func (g *Row) spanStride() int {
	d := 1
	for _, sp := range g.Nodes {
		if sp.Kind != NHint {
			continue
		}
		var walk func(n int) int
		walk = func(n int) int {
			nd := g.Nodes[n]
			if q := g.rowQuot(n); q > 1 {
				return q
			}
			if nd.Kind == NRowIdx {
				return -1
			}
			x := 0
			for _, o := range g.operands(n) {
				switch y := walk(o); {
				case y < 0 || x > 0 && y > 0 && y != x:
					return -1
				case y > 0:
					x = y
				}
			}
			return x
		}
		if x := walk(sp.A); x > 0 {
			d = x
		}
	}
	return d
}

// rowDepsPast is rowDeps with the hints' nodes, the row's remainder by
// stride and its quotient by group read as uniform: over rows stride apart,
// and over each aligned group of rows.
func (g *Row) rowDepsPast(stride, group int) []bool {
	return g.rowDepsWith(func(i int) bool {
		nd := g.Nodes[i]
		if nd.Kind == NHint {
			return true
		}
		if nd.Kind != NUBinary || g.Nodes[nd.B].Kind != NUConst {
			return false
		}
		q, m := g.rowQuot(nd.A), g.Nodes[nd.B].At
		return nd.Op == OpRem && q > 0 && m > 0 && stride%(q*m) == 0 || nd.Op == OpDiv && group > 1 && q == 1 && m == group
	})
}

// rowQuot is q where node n is the row over q (1 for the row itself), else 0.
func (g *Row) rowQuot(n int) int {
	nd := g.Nodes[n]
	switch {
	case nd.Kind == NRowIdx:
		return 1
	case nd.Kind == NUBinary && nd.Op == OpDiv && g.Nodes[nd.B].Kind == NUConst && g.Nodes[nd.B].At > 0:
		if q := g.rowQuot(nd.A); q > 0 {
			return q * g.Nodes[nd.B].At
		}
	}
	return 0
}

// rowDepsWith is rowDeps where a node uniform names reads as no row's.
func (g *Row) rowDepsWith(uniform func(int) bool) []bool {
	v := make([]bool, len(g.Nodes))
	for changed := true; changed; {
		changed = false
		for i, nd := range g.Nodes {
			if v[i] || uniform(i) {
				continue
			}
			x := nd.Kind == NRowIdx || nd.Kind == NRowOff
			if nd.Kind == NLoad && !nd.Indexed {
				k := g.Bufs[nd.Buf].Kind
				x = k == BufRow || k == BufRowF16 || k == BufRowScalar || k == BufU32
			}
			if nd.Kind == NFold {
				x = x || v[nd.B] || nd.A >= 0 && v[nd.A]
			}
			for _, o := range g.operands(i) {
				x = x || v[o]
			}
			if x {
				v[i], changed = true, true
			}
		}
	}
	return v
}

// gathersByColumn reports whether a float load's place reads a load that
// reads the element: an operand gathered through a table.
func (g *Row) gathersByColumn() bool {
	col := g.colDeps()
	var via func(n int, seen map[int]bool) bool
	via = func(n int, seen map[int]bool) bool {
		if seen[n] {
			return false
		}
		seen[n] = true
		nd := g.Nodes[n]
		if nd.Kind == NLoad && nd.Indexed && col[n] {
			return true
		}
		for _, o := range g.operands(n) {
			if via(o, seen) {
				return true
			}
		}
		return false
	}
	for i, nd := range g.Nodes {
		if nd.Kind == NLoad && nd.Indexed && g.typeOf(i) != ir.U32 && col[i] && via(nd.Ix, map[int]bool{}) {
			return true
		}
	}
	return false
}

// rowGroup is d where a load's place reads the row as the row over d (a
// selection per aligned group of rows), else 1.
func (g *Row) rowGroup() int {
	for _, nd := range g.Nodes {
		if nd.Kind != NLoad || !nd.Indexed {
			continue
		}
		if x := g.Nodes[nd.Ix]; x.Kind == NUBinary && x.Op == OpDiv && g.Nodes[x.A].Kind == NRowIdx && g.Nodes[x.B].Kind == NUConst && g.Nodes[x.B].At > 1 {
			return g.Nodes[x.B].At
		}
	}
	return 1
}

// selectedRowDeps is rowDeps with a selection read by the row -- a word
// loaded at the row, or at the row over its group -- read as the row's
// choice of sheet, not as the row's own value.
func (g *Row) selectedRowDeps() []bool {
	d := g.rowGroup()
	return g.rowDepsWith(func(i int) bool {
		nd := g.Nodes[i]
		if nd.Kind == NUBinary && nd.Op == OpDiv && d > 1 && g.Nodes[nd.A].Kind == NRowIdx && g.Nodes[nd.B].Kind == NUConst && g.Nodes[nd.B].At == d {
			return true
		}
		if nd.Kind != NLoad || !nd.Indexed || g.typeOf(i) != ir.U32 {
			return false
		}
		x := g.Nodes[nd.Ix]
		return x.Kind == NRowIdx || x.Kind == NUBinary && x.Op == OpDiv && g.Nodes[x.A].Kind == NRowIdx && g.Nodes[x.B].Kind == NUConst && g.Nodes[x.B].At == d
	})
}

// foldsReadAfter reports whether a fold of a loop run for a run-time count,
// a maximum or a sum, is read after its loop.
func (g *Row) foldsReadAfter() bool {
	for c := range g.Loops {
		if g.tripOf(c) < 0 {
			continue
		}
		body := g.bodyOf(c)
		for f, nd := range g.Nodes {
			if nd.Kind != NFold || nd.At != c {
				continue
			}
			if _, ok := g.foldRed(f); ok && g.readAfter(f, body) {
				return true
			}
		}
	}
	return false
}

// keyBlock is the places a range loop's block takes (Schedule.Block).
const keyBlock = 8

// sumsLoop0 reports whether value n reads a fold over loop 0 whose step
// adds into it: a sum, not a running maximum or a selection.
func (g *Row) sumsLoop0(n int) bool {
	seen := map[int]bool{}
	var walk func(n int) bool
	walk = func(n int) bool {
		if seen[n] {
			return false
		}
		seen[n] = true
		nd := g.Nodes[n]
		if nd.Kind == NFold && nd.At == 0 {
			if nd.A < 0 {
				return false
			}
			st := g.Nodes[nd.A]
			return st.Kind == NDot4 || st.Kind == NBinary && (st.Op == OpAdd || st.Op == OpFma) || st.Kind == NUBinary && st.Op == OpAdd
		}
		for _, o := range g.operands(n) {
			if walk(o) {
				return true
			}
		}
		return false
	}
	return walk(n)
}
