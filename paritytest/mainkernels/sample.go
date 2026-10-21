package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// The sampler's device half: the repeat penalty and the top-k selection, so a
// sampled token reads back k candidates rather than the vocabulary's logits.
//
//	SamplePenalty   a penalized copy of the logits
//	SampleTopK      each group's k best of its slice, then (one group) the k best
//	                of those: the candidates, best first
//
// What is selected is exactly what the host sampler selects
// (engine/model.Sampler, nn.SampleOrder): the k highest values, higher first
// and the lower id on a tie, over the penalized row. Selection only compares,
// and the penalty is one IEEE multiply or correctly rounded divide per history
// entry, so the candidates are bit for bit the host's over the same logits; the
// softmax over k, the cuts and the draw stay with the host sampler's own
// kernels, whose exp a device's does not reproduce.

// SampleGroup is the top-k kernels' workgroup and SampleSlice how many logits
// one group of the first pass reduces.
const (
	SampleGroup = 256
	SampleSlice = SampleGroup * 8
)

// SampleArgsWords is the argument block's length for a history of h tokens:
// the count, the penalty's bits, and the ids.
func SampleArgsWords(h int) int { return 2 + h }

// SamplePenalty writes pOut = pIn with llama.cpp's repeat penalty applied once
// per history entry, in history order, as the host's scatter does: a positive
// value divided by the penalty, anything else multiplied. pArgs is [count,
// penalty bits, ids...]. One thread per logit; launch SampleGroups(n) groups
// of SampleGroup.
func SamplePenalty(n int) (*ir.Kernel, error) {
	if n < 1 {
		return nil, fmt.Errorf("kernels: SamplePenalty: n=%d", n)
	}
	b := ir.New("samplepenalty", [3]int{SampleGroup, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pArgs := b.Param("pArgs", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	zero := b.Const(ir.U32, 0)
	cnt := b.Load(ir.U32, pArgs, zero, 0)
	pen := b.Bitcast(ir.F32, b.Load(ir.U32, pArgs, zero, 1))
	fz := b.ConstF32(0)
	v0 := b.Load(ir.F32, pIn, i, 0)
	b.LoopN(cnt)
	j := b.Phi(ir.U32, zero)
	v := b.Phi(ir.F32, v0)
	id := b.Load(ir.U32, pArgs, j, 2)
	hit := b.Lt(ir.U32, zero, eqU32(b, id, i))
	pos := b.Lt(ir.F32, fz, v)
	nv := b.Select(ir.F32, pos, divRN(b, v, pen), b.Mul(ir.F32, v, pen))
	b.SetPhi(v, b.Select(ir.F32, hit, nv, v))
	b.SetPhi(j, b.Add(ir.U32, j, b.Const(ir.U32, 1)))
	b.EndLoop()
	b.Store(pOut, i, v, 0)
	return b.Done(), nil
}

// SampleGroups is how many groups SamplePenalty takes over n logits, and
// SampleSlices how many the first top-k pass takes.
func SampleGroups(n int) int { return (n + SampleGroup - 1) / SampleGroup }
func SampleSlices(n int) int { return (n + SampleSlice - 1) / SampleSlice }

// SampleTopK writes, for each group g, the k best of elements
// [g*per, min(n, g*per+per)) of pV to pOutV/pOutI[g*k ..], best first. The
// first pass runs it over the logits with per = SampleSlice and the positions
// as ids (ids false); the second over the first's n = groups*k candidates with
// per = n, one group, and ids read from pI (ids true). "Best" is the host
// order: a higher value, or an equal one with a lower id.
//
// Each of the k rounds finds the best element strictly below the last round's
// winner in that order, so no element is taken twice. A slice with fewer than
// k elements pads its tail with (-Inf, 0xFFFFFFFF), below every real element;
// the caller asks for k below the vocabulary, so a pad never reaches the k
// kept. Parameters: pV, pI (u32, read only when ids), pOutV, pOutI.
func SampleTopK(n, per, k int, ids bool) (*ir.Kernel, error) {
	if n < 1 || per < 1 || k < 1 || k > per {
		return nil, fmt.Errorf("kernels: SampleTopK: n=%d per=%d k=%d", n, per, k)
	}
	const g = SampleGroup
	name := "sampletopk"
	if ids {
		name = "sampletopkids"
	}
	b := ir.New(name, [3]int{g, 1, 1})
	pV := b.Param("pV", ir.F32)
	pI := b.Param("pI", ir.U32)
	pOutV := b.Param("pOutV", ir.F32)
	pOutI := b.Param("pOutI", ir.U32)
	sv := b.Shared("sv", ir.F32, g)
	si := b.Shared("si", ir.U32, g)

	u := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	zero, one := u(0), u(1)
	tid, grp := b.TID(), b.CTAID()
	base := b.Mul(ir.U32, grp, u(int64(per)))
	none := u(0xFFFFFFFF)
	ninf := b.Bitcast(ir.F32, u(0xFF800000))
	flag := func(p ir.Value) ir.Value { return b.Select(ir.U32, p, one, zero) }
	// ahead(a, b): a comes before b in the host order.
	ahead := func(av, ai, bv, bi ir.Value) ir.Value {
		return b.Lt(ir.U32, zero, b.Select(ir.U32, b.Lt(ir.F32, bv, av), one,
			b.Select(ir.U32, b.Lt(ir.F32, av, bv), zero, flag(b.Lt(ir.U32, ai, bi)))))
	}
	and := func(x, y ir.Value) ir.Value { return b.Lt(ir.U32, zero, b.And(ir.U32, flag(x), flag(y))) }
	or := func(x, y ir.Value) ir.Value { return b.Lt(ir.U32, zero, b.Add(ir.U32, flag(x), flag(y))) }
	outBase := b.Mul(ir.U32, grp, u(int64(k)))
	top := u(g - 1)
	last := u(int64(n - 1))

	b.Loop(int64(k))
	r := b.Phi(ir.U32, zero)
	pv := b.Phi(ir.F32, ninf)
	pi := b.Phi(ir.U32, none)
	first := b.Lt(ir.U32, r, one)

	b.Loop(int64((per + g - 1) / g))
	j := b.Phi(ir.U32, tid)
	bv := b.Phi(ir.F32, ninf)
	bi := b.Phi(ir.U32, none)
	gi := b.Add(ir.U32, base, j)
	inRange := and(b.Lt(ir.U32, j, u(int64(per))), b.Lt(ir.U32, gi, u(int64(n))))
	at := b.Min(ir.U32, gi, last)
	cv := b.Load(ir.F32, pV, at, 0)
	ci := at
	if ids {
		ci = b.Load(ir.U32, pI, at, 0)
	}
	// Below the last winner: what the earlier rounds took is out.
	below := or(first, ahead(pv, pi, cv, ci))
	take := and(and(inRange, below), ahead(cv, ci, bv, bi))
	b.SetPhi(bv, b.Select(ir.F32, take, cv, bv))
	b.SetPhi(bi, b.Select(ir.U32, take, ci, bi))
	b.SetPhi(j, b.Add(ir.U32, j, u(g)))
	b.EndLoop()

	b.Store(sv, tid, bv, 0)
	b.Store(si, tid, bi, 0)
	b.Barrier()
	for s := g / 2; s >= 1; s /= 2 {
		mv := b.Load(ir.F32, sv, tid, 0)
		mi := b.Load(ir.U32, si, tid, 0)
		o := b.Min(ir.U32, b.Add(ir.U32, tid, u(int64(s))), top)
		ov := b.Load(ir.F32, sv, o, 0)
		oi := b.Load(ir.U32, si, o, 0)
		w := ahead(ov, oi, mv, mi)
		rv := b.Select(ir.F32, w, ov, mv)
		ri := b.Select(ir.U32, w, oi, mi)
		b.Barrier()
		b.Store(sv, tid, rv, 0)
		b.Store(si, tid, ri, 0)
		b.Barrier()
	}
	wv := b.Load(ir.F32, sv, zero, 0)
	wi := b.Load(ir.U32, si, zero, 0)
	// Every thread has read the winner before the next round overwrites it.
	b.Barrier()
	b.Store(pOutV, b.Add(ir.U32, outBase, r), wv, 0)
	b.Store(pOutI, b.Add(ir.U32, outBase, r), wi, 0)
	b.SetPhi(r, b.Add(ir.U32, r, one))
	b.SetPhi(pv, wv)
	b.SetPhi(pi, wi)
	b.EndLoop()
	return b.Done(), nil
}
