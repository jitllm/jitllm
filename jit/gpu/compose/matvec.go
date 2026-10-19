package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// MatVec is a matvec group: Rows outputs of a weight in format T over K,
// against a chunk of NTok tokens' int8 operand (a float format: against the
// float rows), each output passed through Epi. The format's arithmetic over a
// sub-block is its op fragment (kernels.SubDot); MatVec is the loop nest
// around it, by its schedule: Tok tokens and Rowt rows a thread, Split
// segments of k a row, reduced in the workgroup (Group) or written as
// partial planes for SplitSum. Outputs are token-major, Rows a token.
//
// Parameters: pQS, then pD (a float format: pX, the float row), pSC, pA,
// pAX, pOut, then the epilogue's.
type MatVec struct {
	T         kernels.Quant
	K, Rows   int
	NTok, Tok int
	// Experts is a bank of that many sheets of Rows x K, and Slots the
	// selected sheets one launch runs (pSel holds their ids, at one token);
	// SlotAct gives each slot its own operand vector (a down bank's).
	Experts, Slots int
	SlotAct        bool
	Split          int
	Group          bool
	Rowt           int
	Center         kernels.Center
	Epi            *Epi
}

// Encode is the group's structure and schedule, for the kernel cache's key.
func (m *MatVec) Encode() []uint32 {
	e := []uint32{3, uint32(m.T), uint32(m.K), uint32(m.Rows), uint32(m.NTok), uint32(m.Tok), uint32(m.Experts), uint32(m.Slots),
		b2u(m.SlotAct), uint32(m.Split), b2u(m.Group), uint32(m.Rowt),
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

// Threads is the grid: a thread a row tile, token group and segment.
func (m *MatVec) Threads() int {
	return m.Rows / max(m.Rowt, 1) * m.groups() * max(m.Split, 1)
}

// tok is the tokens a thread; groups the token groups.
func (m *MatVec) tok() int {
	if max(m.NTok, 1) == 1 {
		return 1
	}
	return max(m.Tok, 1)
}

func (m *MatVec) groups() int { return m.lanes() / m.tok() }

// lanes is the output's slots: a chunk's tokens, or a bank's selected sheets.
func (m *MatVec) lanes() int {
	if max(m.NTok, 1) > 1 {
		return m.NTok
	}
	if m.Experts > 1 {
		return max(m.Slots, 1)
	}
	return 1
}

// Emit forms the kernel.
func (m *MatVec) Emit() (*ir.Kernel, error) {
	split, rowt := max(m.Split, 1), max(m.Rowt, 1)
	ntok, tok, groups := max(m.NTok, 1), m.tok(), m.groups()
	switch {
	case ntok%tok != 0:
		return nil, fmt.Errorf("compose: MatVec: %d tokens at %d a thread", ntok, tok)
	case m.Group && groups > 1 && m.Experts <= 1:
		return nil, fmt.Errorf("compose: MatVec: an in-group split over %d token groups", groups)
	case m.Experts > 1 && ntok > 1:
		return nil, fmt.Errorf("compose: MatVec: a bank over a chunk is the grouped run's")
	case m.Group && (split < 2 || rowt > 1 || !kernels.GroupSplitOK(m.Rows, split)):
		return nil, fmt.Errorf("compose: MatVec: an in-group split of %d over %d rows at %d rows a thread", split, m.Rows, rowt)
	case !m.Group && split > 1 && m.Epi != nil:
		// The segments meet in a partial plane (pOut, Split planes of Rows);
		// the epilogue is the reduction's (SplitSum).
		return nil, fmt.Errorf("compose: MatVec: a split of %d writes partials; its epilogue is the reduction's", split)
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
	var pSel ir.Value
	if m.Experts > 1 {
		pSel = b.Param("pSel", ir.U32) // the selected sheets' ids, a slot each
	}
	if m.Epi != nil {
		m.Epi.Params(b)
	}
	var shPart ir.Value
	if m.Group {
		shPart = b.Shared("part", ir.F32, group*tok) // before every loop (ir.Validate)
	}
	s := kernels.MatVecShape{T: m.T, K: m.K, Rows: m.Rows, NTok: m.NTok, Tok: tok, Split: split, Rowt: rowt,
		GroupSplit: m.Group, Center: m.Center, Experts: m.Experts, Slots: m.Slots, SlotAct: m.SlotAct}
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
	flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(nrow2*groups*split-1)))
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
	// The output's slot: a chunk's token group (its first token), or a
	// bank's selected sheet; an in-group split over a bank takes it from the
	// workgroup, Rows/rpg workgroups a slot.
	lanes := m.lanes()
	slot := ir.Value(0)
	switch {
	case m.Group && m.Experts > 1:
		rbk := b.Const(ir.U32, int64(m.Rows/rpg))
		c := b.CTAID()
		slot = b.Min(ir.U32, b.Div(ir.U32, c, rbk), b.Const(ir.U32, int64(lanes-1)))
		orow = b.Add(ir.U32, b.Mul(ir.U32, b.Rem(ir.U32, c, rbk), b.Const(ir.U32, int64(rpg))), rowLocal)
	case lanes > 1 && groups > 1:
		rest := splitSeg
		ngrp := b.Const(ir.U32, int64(groups))
		g := b.Rem(ir.U32, rest, ngrp)
		splitSeg = b.Div(ir.U32, rest, ngrp)
		slot = g
		if tok > 1 {
			slot = b.Mul(ir.U32, g, b.Const(ir.U32, int64(tok)))
		}
	case lanes > 1 || m.Experts > 1:
		slot = b.Const(ir.U32, 0)
	}
	if env.SlotAct {
		env.AxS = b.Mul(ir.U32, slot, b.Const(ir.U32, int64(env.NB)))
		env.AxU = b.Mul(ir.U32, slot, b.Const(ir.U32, int64(m.K/16)))
	}
	env.PA, env.PAX, env.PX = pA, pAX, pX
	wrow, wrowD, wrowSC := orow, orow, orow
	var eSel ir.Value
	if m.Experts > 1 {
		// A bank is Experts sheets back to back in each plane, each plane its
		// own words a sheet: the slot's sheet adds a base to each.
		nq, nd, nsc, err := kernels.PackedWords(m.T, m.Rows, m.K)
		if err != nil {
			return nil, err
		}
		eSel = b.Load(ir.U32, pSel, slot, 0)
		base := func(words int) ir.Value {
			if words == 0 {
				return orow
			}
			return b.Add(ir.U32, b.Mul(ir.U32, eSel, b.Const(ir.U32, int64(words))), orow)
		}
		wrow, wrowD, wrowSC = base(nq), base(nd), base(nsc)
		env.DExpAdd = func(v ir.Value) ir.Value {
			return b.Add(ir.U32, b.Mul(ir.U32, eSel, b.Const(ir.U32, int64(nd))), v)
		}
	}
	env.BankRows, env.WSel, env.WRowD, env.WRowSC, env.WRows = orows, orow, wrowD, wrowSC, m.Rows

	words := env.PW + env.HW
	segSub := b.Mul(ir.U32, splitSeg, b.Const(ir.U32, int64(spg)))
	wStart := b.Add(ir.U32, wrow, b.Mul(ir.U32, segSub, b.Mul(ir.U32, b.Const(ir.U32, int64(words)), orows)))
	aStart := b.Mul(ir.U32, segSub, b.Const(ir.U32, int64(env.AWords)))
	if env.SlotAct {
		aStart = b.Add(ir.U32, aStart, b.Mul(ir.U32, slot, b.Const(ir.U32, int64(m.K/4))))
	}
	if fl {
		rowF := orow
		if m.Experts > 1 {
			rowF = b.Add(ir.U32, b.Mul(ir.U32, eSel, orows), orow)
		}
		wStart = b.Add(ir.U32, b.Mul(ir.U32, rowF, b.Const(ir.U32, int64(env.FW))),
			b.Mul(ir.U32, segSub, b.Const(ir.U32, int64(env.FSub))))
	}

	b.Loop(int64(spg))
	accs := make([]ir.Value, rowt*tok)
	for j := range accs {
		accs[j] = b.Phi(ir.F32, env.ZeroF)
	}
	wIdx := b.Phi(ir.U32, wStart)
	aIdx := b.Phi(ir.U32, aStart)
	subIdx := b.Phi(ir.U32, segSub)
	iaccs := make([]ir.Value, rowt*tok)
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
		for t := 0; t < tok; t++ {
			b.Store(shPart, b.TID(), accs[t], int64(t*group))
		}
		b.Barrier()
		for t := 0; t < tok; t++ {
			sum := b.Load(ir.F32, shPart, rowLocal, int64(t*group))
			for sg := 1; sg < split; sg++ {
				sum = b.Add(ir.F32, sum, b.Load(ir.F32, shPart, rowLocal, int64(t*group+sg*rpg)))
			}
			at := orow
			if lanes > 1 {
				at = b.Add(ir.U32, b.Mul(ir.U32, slot, orows), orow)
			}
			if t > 0 {
				at = b.Add(ir.U32, at, b.Const(ir.U32, int64(t*m.Rows)))
			}
			store(sum, at, orow)
		}
		return b.Done(), nil
	}
	// Token-major outputs, Rows a token; a split's partials at its segment's
	// plane of Rows*NTok.
	base := orow
	if lanes > 1 {
		base = b.Add(ir.U32, b.Mul(ir.U32, slot, orows), orow)
	}
	if split > 1 {
		base = b.Add(ir.U32, base, b.Mul(ir.U32, splitSeg, b.Const(ir.U32, int64(m.Rows*lanes))))
	}
	for j := 0; j < rowt; j++ {
		row := orow
		if j > 0 {
			row = b.Add(ir.U32, orow, b.Const(ir.U32, int64(j)))
		}
		for t := 0; t < tok; t++ {
			at := base
			if off := t*m.Rows + j; off > 0 {
				at = b.Add(ir.U32, base, b.Const(ir.U32, int64(off)))
			}
			store(accs[j*tok+t], at, row)
		}
	}
	return b.Done(), nil
}

// SplitSum is a split matvec's reduction former: Rows outputs, each the sum
// of its Split partials in segment order from zero (pIn, Split planes of
// Rows), passed through Epi. Parameters: pIn, pOut, then the epilogue's.
type SplitSum struct {
	Rows, Split int
	// Width is an output row's width where Rows spans several tokens' rows
	// (Rows a token's row count times the tokens): an epilogue's per-row
	// loads read at the element's place in its row. 0 is Rows.
	Width int
	Epi   *Epi
}

// Encode is the group's structure, for the kernel cache's key.
func (r *SplitSum) Encode() []uint32 {
	e := []uint32{4, uint32(r.Rows), uint32(r.Split), uint32(r.Width)}
	if r.Epi != nil {
		e = append(e, r.Epi.Encode()...)
	}
	return e
}

// Emit forms the kernel: a thread a row, at the width Width.
func (r *SplitSum) Emit() (*ir.Kernel, error) {
	if r.Rows < 1 || r.Split < 2 {
		return nil, fmt.Errorf("compose: SplitSum: %d rows in %d segments", r.Rows, r.Split)
	}
	if r.Epi != nil {
		if err := r.Epi.Check(); err != nil {
			return nil, err
		}
	}
	b := ir.New("composed_splitsum", [3]int{SplitSumWidth, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	if r.Epi != nil {
		r.Epi.Params(b)
	}
	nrows := b.Const(ir.U32, int64(r.Rows))
	flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	row := b.Min(ir.U32, flat, b.Sub(ir.U32, nrows, b.Const(ir.U32, 1)))
	zero := b.ConstF32(0)
	b.Loop(int64(r.Split))
	acc := b.Phi(ir.F32, zero)
	idx := b.Phi(ir.U32, row)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Load(ir.F32, pIn, idx, 0)))
	b.SetPhi(idx, b.Add(ir.U32, idx, nrows))
	b.EndLoop()
	v := acc
	if r.Epi != nil {
		rr := row
		if r.Width > 0 && r.Width != r.Rows {
			rr = b.Rem(ir.U32, row, b.Const(ir.U32, int64(r.Width)))
		}
		v = r.Epi.Apply(b, v, row, rr)
	}
	b.Store(pOut, row, v, 0)
	return b.Done(), nil
}

// SplitSumWidth is SplitSum's workgroup.
const SplitSumWidth = 128
