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
//   - a group with a windowed reduction runs flat, a thread an element, 128
//     to a workgroup (its windows meet in the workgroup);
//   - any other group runs a workgroup a row, its width the narrowest power of
//     two over the row, at most 256, or 1024 with a row reduction.
func (g *Row) Plan(f Facts) ([]Planned, error) {
	all := make([]int, len(g.Bufs))
	for i := range all {
		all[i] = i
	}
	if g.windowed() {
		// Windows meet across a workgroup's consecutive threads: flat, a
		// thread an element, 128 to a group.
		return []Planned{{G: g, S: Schedule{Group: 128, Flat: true, Target: f.Target}, Bufs: all,
			Threads: g.Rows * g.Width}}, nil
	}
	if !g.contraction() {
		s := rowSchedule(g)
		s.Target = f.Target
		return []Planned{{G: g, S: s, Bufs: all, Threads: g.Rows}}, nil
	}
	s := g.contractSchedule(f)
	s.Target = f.Target
	threads := g.Rows / max(s.Tile, 1) * (g.Width / max(s.TileCols, 1))
	if s.Split <= 1 {
		return []Planned{{G: g, S: s, Bufs: all, Threads: threads}}, nil
	}
	first, second, b1, b2, err := g.SplitFold(s.Split)
	if err != nil {
		return nil, err
	}
	return []Planned{
		{G: first, S: s, Bufs: b1, Threads: threads * s.Split},
		{G: second, S: Schedule{Group: s.Group, Flat: true, Target: f.Target}, Bufs: b2, Threads: g.Rows * g.Width},
	}, nil
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
//   - over a ragged step on a target with no int8 matrix instruction, eight
//     tokens and two rows a thread (four from 64 tokens), each halved until
//     it divides (swept on a Volta-class card);
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
	var tok, rowt, split int
	switch {
	case f.Step == StepRagged && !f.Target.IntMMA():
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

// rowSchedule is a row group's schedule: the narrowest power-of-two group
// over the row, at most 256, or 1024 for a group with a row reduction.
func rowSchedule(g *Row) Schedule {
	top := 256
	for _, n := range g.Nodes {
		if n.Kind == NReduce {
			top = 1024
		}
	}
	w := 32
	for w < g.Width && w < top {
		w *= 2
	}
	return Schedule{Group: w}
}

// String is the plan's kernels, for a dump.
func (p Planned) String() string {
	return fmt.Sprintf("%d nodes, %+v, %d threads", len(p.G.Nodes), p.S, p.Threads)
}
