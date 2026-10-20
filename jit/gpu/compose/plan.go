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
	// The warp tile of the int8 matrix instruction: tiles of 16 elements
	// and of 8 rows a warp; NoMMA, the caller's refusal of the instruction.
	MMAMT, MMANT int
	NoMMA        bool
	// A row group's width and combine where the tier states them: RowGroup
	// threads (0 for the rule's), and Tree to combine a reduction through
	// threadgroup memory where the target could butterfly it.
	RowGroup int
	Tree     bool
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
	if g.reduces() && g.Width <= f.Target.Lanes && g.Width&(g.Width-1) == 0 || g.loopReduces() || shuffles {
		// A row no wider than a subgroup, or a reduction a loop's body reads
		// (a value of the body), runs on one subgroup's lanes, the reduction
		// their butterfly.
		s := Schedule{Group: 128, Flat: true, Lanes: g.Width, Target: f.Target}
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
	if len(g.Stores) != 1 || len(g.Loops) == 0 || g.Q8 != nil {
		return false
	}
	vary, rowDep := g.varies(0), g.rowDeps()
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
	for _, nd := range g.Nodes {
		if nd.Kind == NReduce {
			return false
		}
	}
	return weight && operand
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
		rowt = 1
		for _, r := range []int{f.RowTile, 2} {
			if r > 1 && g.Width%r == 0 {
				rowt = r
				break
			}
		}
		split = f.Split
	}
	for g.Rows%tok != 0 {
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
