package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// MatVec is a matvec group: Rows outputs of a weight in format T over K,
// against one token's int8 operand (a float format: against the float row),
// each output passed through Epi. The format's arithmetic over a sub-block is
// its op fragment (kernels.SubDot); MatVec is the loop nest around it, by its
// schedule: Split segments of k a row, reduced in the workgroup (Group) or
// one segment a thread (Split 1), Rowt rows a thread.
//
// Parameters: pQS, then pD (a float format: pX, the float row), pSC, pA,
// pAX, pOut, then the epilogue's.
type MatVec struct {
	T       kernels.Quant
	K, Rows int
	Split   int
	Group   bool
	Rowt    int
	Center  kernels.Center
	Epi     *Epi
}

// Encode is the group's structure and schedule, for the kernel cache's key.
func (m *MatVec) Encode() []uint32 {
	e := []uint32{3, uint32(m.T), uint32(m.K), uint32(m.Rows), uint32(m.Split), b2u(m.Group), uint32(m.Rowt),
		b2u(m.Center.NoCenter), uint32(m.Center.MinTok)}
	if m.Epi != nil {
		e = append(e, m.Epi.Encode()...)
	}
	return e
}

// Width is the workgroup the group launches at; Threads its whole grid.
func (m *MatVec) Width() int {
	if m.Group {
		return kernels.GroupSplitWidth(max(m.Split, 1))
	}
	return 128
}

// Threads is the grid: a thread a row tile and segment.
func (m *MatVec) Threads() int { return m.Rows / max(m.Rowt, 1) * max(m.Split, 1) }

// Emit forms the kernel.
func (m *MatVec) Emit() (*ir.Kernel, error) {
	split, rowt := max(m.Split, 1), max(m.Rowt, 1)
	switch {
	case m.Group && (split < 2 || rowt > 1 || !kernels.GroupSplitOK(m.Rows, split)):
		return nil, fmt.Errorf("compose: MatVec: an in-group split of %d over %d rows at %d rows a thread", split, m.Rows, rowt)
	case !m.Group && split > 1:
		return nil, fmt.Errorf("compose: MatVec: a split of %d meets in a partial plane, which no epilogue reads", split)
	case m.Rows%rowt != 0:
		return nil, fmt.Errorf("compose: MatVec: %d rows at %d a thread", m.Rows, rowt)
	}
	if m.Epi != nil {
		if err := m.Epi.Check(); err != nil {
			return nil, err
		}
	}
	group := m.Width()
	b := ir.New("composed_matvec", [3]int{group, 1, 1})
	fl := kernels.IsFloat(m.T)
	pQS := b.Param("pQS", ir.U32)
	var pD, pX ir.Value
	if fl {
		pX = b.Param("pX", ir.F32)
	} else {
		pD = b.Param("pD", ir.U32)
	}
	pSC := b.Param("pSC", ir.U32)
	pA := b.Param("pA", ir.U32)
	pAX := b.Param("pAX", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	if m.Epi != nil {
		m.Epi.Params(b)
	}
	var shPart ir.Value
	if m.Group {
		shPart = b.Shared("part", ir.F32, group) // before every loop (ir.Validate)
	}
	s := kernels.MatVecShape{T: m.T, K: m.K, Rows: m.Rows, Split: split, Rowt: rowt, GroupSplit: m.Group, Center: m.Center}
	env, nsub, err := kernels.NewSubDotEnv(b, s)
	if err != nil {
		return nil, err
	}
	if nsub%split != 0 {
		return nil, fmt.Errorf("compose: MatVec: a split of %d over %d sub-blocks", split, nsub)
	}
	spg := nsub / split

	// A thread a row tile and k segment, the row fastest so adjacent threads
	// read adjacent rows of the transposed payload; an in-group split lays its
	// group out as Split segments of rpg rows.
	orows := b.Const(ir.U32, int64(m.Rows))
	nrow2 := m.Rows / rowt
	trows := orows
	if rowt > 1 {
		trows = b.Const(ir.U32, int64(nrow2))
	}
	one := env.One
	var segLocal, rowLocal ir.Value
	rpg := 0
	if m.Group {
		rpg = group / split
		tid := b.TID()
		rowLocal = b.Rem(ir.U32, tid, b.Const(ir.U32, int64(rpg)))
		segLocal = b.Div(ir.U32, tid, b.Const(ir.U32, int64(rpg)))
	}
	flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(nrow2*split-1)))
	orow := b.Rem(ir.U32, flat, trows)
	splitSeg := b.Div(ir.U32, flat, trows)
	if rowt > 1 {
		orow = b.Mul(ir.U32, orow, b.Const(ir.U32, int64(rowt)))
	}
	if m.Group {
		orow = b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.Const(ir.U32, int64(rpg))), rowLocal)
		orow = b.Min(ir.U32, orow, b.Sub(ir.U32, orows, one))
		splitSeg = segLocal
	}
	env.PA, env.PAX, env.PX = pA, pAX, pX
	env.BankRows, env.WSel, env.WRowD, env.WRowSC, env.WRows = orows, orow, orow, orow, m.Rows

	words := env.PW + env.HW
	segSub := b.Mul(ir.U32, splitSeg, b.Const(ir.U32, int64(spg)))
	wStart := b.Add(ir.U32, orow, b.Mul(ir.U32, segSub, b.Mul(ir.U32, b.Const(ir.U32, int64(words)), orows)))
	aStart := b.Mul(ir.U32, segSub, b.Const(ir.U32, int64(env.AWords)))
	if fl {
		wStart = b.Add(ir.U32, b.Mul(ir.U32, orow, b.Const(ir.U32, int64(env.FW))),
			b.Mul(ir.U32, segSub, b.Const(ir.U32, int64(env.FSub))))
	}

	b.Loop(int64(spg))
	accs := make([]ir.Value, rowt)
	for j := range accs {
		accs[j] = b.Phi(ir.F32, env.ZeroF)
	}
	wIdx := b.Phi(ir.U32, wStart)
	aIdx := b.Phi(ir.U32, aStart)
	subIdx := b.Phi(ir.U32, segSub)
	iaccs := make([]ir.Value, rowt)
	for j := range iaccs {
		iaccs[j] = b.Const(ir.I32, 0)
	}
	stream := kernels.SubDot(b, env, wIdx, aIdx, subIdx)
	w := stream(pQS, pD, pSC, accs, iaccs)
	b.SetPhi(wIdx, w)
	b.SetPhi(aIdx, b.Add(ir.U32, aIdx, b.Const(ir.U32, int64(env.AWords))))
	b.SetPhi(subIdx, b.Add(ir.U32, subIdx, one))
	b.EndLoop()

	store := func(v, at, row ir.Value) {
		if m.Epi != nil {
			v = m.Epi.Apply(b, v, at, row)
		}
		b.Store(pOut, at, v, 0)
	}
	if m.Group {
		// The segments meet in the workgroup, summed in segment order; every
		// thread of a row stores the same bits (RULE 13).
		b.Store(shPart, b.TID(), accs[0], 0)
		b.Barrier()
		sum := b.Load(ir.F32, shPart, rowLocal, 0)
		for sg := 1; sg < split; sg++ {
			sum = b.Add(ir.F32, sum, b.Load(ir.F32, shPart, rowLocal, int64(sg*rpg)))
		}
		store(sum, orow, orow)
		return b.Done(), nil
	}
	for j := 0; j < rowt; j++ {
		at := orow
		if j > 0 {
			at = b.Add(ir.U32, orow, b.Const(ir.U32, int64(j)))
		}
		store(accs[j], at, at)
	}
	return b.Done(), nil
}
