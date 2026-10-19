package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The paged decode attention, formed in compose. Its schedule is the decode
// plan (kernels.ChooseDecodePlan), which follows the step's key count, so the
// kernels are formed per plan in the prologue -- pagedPrep chooses the plan
// before the submission, and pagedBuild forms its variant here once -- and an
// attention op's generated launch is the call's variant: the executor runs
// the plan's launches, each bound to the step's buffers through generic
// arguments (q, the pools, the causal count, the table, the descriptors, the
// sinks, the partials and the staged planes).

// formAttendPlan is plan's kernels formed and compiled through the one
// generated-kernel cache, as launches: FlashDecodeKV's schedule
// (compose.Attend), or the staged scores, softmax and accumulate; and the
// merge of their partials where there are any, a thread an element or, where
// items is set, a 32-lane group a (row, head). lanes is what the device
// promises a subgroup (1 or 32): the staged forms take one lane where it
// promises less. Callers hold g.mu, outside any submission.
func (g *devTier) formAttendPlan(plan kernels.DecodePlan, lanes int, items bool) ([]genLaunch, error) {
	s := plan.Shape
	var out []genLaunch
	form := func(enc []uint32, build func() (*ir.Kernel, error), groups int, args ...genArg) error {
		k, width := g.genKern(enc, build)
		if k == nil {
			return fmt.Errorf("a formed attention kernel: %s", g.LastErr)
		}
		l := genLaunch{k: k, groups: groups, width: width, n: uint8(len(args)), whole: true}
		copy(l.args[:], args)
		out = append(out, l)
		return nil
	}
	merged := s.Splits > 1
	switch plan.Path {
	case kernels.PathStaged:
		merged = true
		// A row-major latent row is hundreds of floats a key: a warp across
		// its dimensions where the device promises one. Head rows keep a
		// thread a slot.
		scoreLanes := 1
		if s.MLA > 0 && lanes == ir.SubgroupLanes {
			scoreLanes = ir.SubgroupLanes
		}
		sc := &compose.StagedScores{S: s, Lanes: scoreLanes}
		sm := &compose.StagedSoftmax{S: s, Lanes: lanes}
		ac := &compose.StagedAcc{S: s}
		if err := form(sc.Encode(), sc.Emit, sc.Groups(), gaIn0, gaKPool, gaN, gaPlane, gaTab, gaADesc); err != nil {
			return nil, err
		}
		if err := form(sm.Encode(), sm.Emit, sm.Groups(), gaPlane, gaN, gaPlane2, gaAPart, gaTab, gaADesc); err != nil {
			return nil, err
		}
		if err := form(ac.Encode(), ac.Emit, ac.Groups(), gaPlane2, gaVPool, gaN, gaAPart, gaTab, gaADesc); err != nil {
			return nil, err
		}
	default:
		a := &compose.Attend{S: s}
		dst := gaOut0
		if merged {
			dst = gaAPart
		}
		args := []genArg{gaIn0, gaKPool, gaVPool, gaN, dst, gaTab, gaADesc}
		if s.Sink {
			args = append(args, gaSink)
		}
		if err := form(a.Encode(), a.Emit, a.Groups(), args...); err != nil {
			return nil, err
		}
	}
	if !merged {
		return out, nil
	}
	// The merge and its partials run at the value's width: a head, or a
	// latent attention's value (the row's prefix), not the row the scores
	// read.
	m := &compose.Merge{S: mergeShape(s), Wide: !items}
	args := []genArg{gaAPart, gaOut0}
	if s.Sink {
		args = append(args, gaSink)
	}
	if err := form(m.Encode(), m.Emit, m.Groups(), args...); err != nil {
		return nil, err
	}
	return out, nil
}

// attendBufs is what a plan's launches read, resolved by the caller for the
// layer and the call: the query, the history's pools and table, the
// descriptor set, the output and the sinks; the scratch's causal count,
// partials and staged planes.
type attendBufs struct {
	q, k, v, n, out, tab, desc, sink, part, sc, wt backend.Buf
}

// runPlan launches ls with their arguments bound to ab. The argument lists
// are arrays on the stack: a warm step allocates nothing.
func runPlan(lc *launcher, ls []genLaunch, ab *attendBufs) {
	var bufs [8]backend.Buf
	for i := range ls {
		l := &ls[i]
		for j := 0; j < int(l.n); j++ {
			switch l.args[j] {
			case gaIn0:
				bufs[j] = ab.q
			case gaOut0:
				bufs[j] = ab.out
			case gaKPool:
				bufs[j] = ab.k
			case gaVPool:
				bufs[j] = ab.v
			case gaN:
				bufs[j] = ab.n
			case gaTab:
				bufs[j] = ab.tab
			case gaADesc:
				bufs[j] = ab.desc
			case gaSink:
				bufs[j] = ab.sink
			case gaAPart:
				bufs[j] = ab.part
			case gaPlane:
				bufs[j] = ab.sc
			case gaPlane2:
				bufs[j] = ab.wt
			}
		}
		*lc.memoK, *lc.memoSrc = nil, nil
		if *lc.err == nil {
			*lc.err = lc.launch(l.k, l.groups, l.width, bufs[:l.n]...)
		}
	}
}

// genAttend is a decode attention over the paged history: one launch that
// runs the call's plan (genLaunch.plan), whose kernels the prologue formed
// for the step's key count. nil where the history is not paged (a non-causal
// segment's contiguous cache) or the op is over a prompt's rows, whose
// attention is the prefill's.
func (g *devTier) genAttend(o *devOp, bs *blockScratch, rows int) []genLaunch {
	if rows != 1 || bs.pkv == nil || !g.pagedPlan(&bs.p) || len(o.In) != 1 || len(o.Out) != 1 {
		return nil
	}
	return []genLaunch{{plan: true}}
}
