package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// SubgroupProbeOuts is how many t-typed results SubgroupProbe writes per
// thread, and SubgroupProbeMeta how many u32 ones.
const (
	SubgroupProbeOuts = 5
	SubgroupProbeMeta = 2
)

// SubgroupProbe exercises every subgroup collective (ir/subgroup.go) in one
// kernel, for a gate that holds each target's lowering to a host reference.
//
// Workgroups of two subgroups, so a collective that leaked across the
// subgroup boundary reads the wrong half. Thread i of the launch (group*64 +
// tid) reads pIn[i] and pSrc[i] and writes, at pOut[i*5 ..]: the subgroup's
// sum, maximum and minimum of pIn, pIn broadcast from lane (7*sg + 3) & 31
// where sg is the subgroup's index in the workgroup (uniform), and pIn
// shuffled from lane pSrc[i] (per lane, any value: the op masks it to 31); and
// at pMeta[i*2 ..]: the lane index and the subgroup width.
func SubgroupProbe(t ir.Type) (*ir.Kernel, error) {
	switch t {
	case ir.F32, ir.I32, ir.U32:
	default:
		return nil, fmt.Errorf("kernels: SubgroupProbe: %s is not a 32-bit scalar", t)
	}
	const width = 2 * ir.SubgroupLanes
	b := ir.New("subgroupprobe_"+t.String(), [3]int{width, 1, 1})
	pIn := b.Param("pIn", t)
	pSrc := b.Param("pSrc", ir.U32)
	pOut := b.Param("pOut", t)
	pMeta := b.Param("pMeta", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), c(width)), b.TID())
	x := b.Load(t, pIn, i, 0)
	src := b.Load(ir.U32, pSrc, i, 0)
	sg := b.SubgroupIndex()
	from := b.Add(ir.U32, b.Mul(ir.U32, sg, c(7)), c(3))

	o := b.Mul(ir.U32, i, c(SubgroupProbeOuts))
	b.Store(pOut, o, b.SubgroupReduce(t, x, ir.ReduceAdd), 0)
	b.Store(pOut, o, b.SubgroupReduce(t, x, ir.ReduceMax), 1)
	b.Store(pOut, o, b.SubgroupReduce(t, x, ir.ReduceMin), 2)
	b.Store(pOut, o, b.SubgroupBroadcast(t, x, from), 3)
	b.Store(pOut, o, b.Shuffle(t, x, src), 4)

	m := b.Mul(ir.U32, i, c(SubgroupProbeMeta))
	b.Store(pMeta, m, b.LaneID(), 0)
	b.Store(pMeta, m, b.SubgroupSize(), 1)
	k := b.Done()
	if err := k.Validate(); err != nil {
		return nil, err
	}
	return k, nil
}
