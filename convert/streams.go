package convert

import (
	"github.com/jitllm/jitllm/format/jlm"
)

// StateStreams states the residual's streams in c where the model carries
// several: Kimi-K3's bank of checkpoints, a stream for the running residual
// and one per block of AttnResBlock (modeling_kimi_linear.py's block
// attention residuals). Each block's sublayer reads the bank banked before
// it and the running residual, mixed by the sublayer's score vector; the
// first block of a group banks its input into the group's stream, and its
// attention's output is the running residual rather than added to it.
// stores says block li (-1 the dense tensors) stores role.
func StateStreams(c *jlm.Config, stores func(r jlm.Role, li int32) bool) {
	if c.AttnResBlock == 0 || c.NLayer == 0 {
		return
	}
	n, blk := int(c.NLayer), int(c.AttnResBlock)
	final := (n + blk - 1) / blk
	s := jlm.Streams{N: uint8(1 + final), Blocks: make([]jlm.BlockStreams, n)}
	for li := range n {
		b := &s.Blocks[li]
		bank := (li + blk - 1) / blk
		checkpoint := li%blk == 0
		if checkpoint {
			b.Bank = uint8(1 + li/blk)
			b.Attn.Join = &jlm.Program{NIn: 2, Out: 1}
		}
		b.Attn = bankRead(c, b.Attn, bankStreams(bank), jlm.RoleAttnResScore, stores(jlm.RoleAttnResScore, int32(li)))
		ffn := bankStreams(li/blk + 1)
		if checkpoint {
			ffn[b.Bank-1] = jlm.StreamIn
		}
		b.FFN = bankRead(c, b.FFN, ffn, jlm.RoleFFNResScore, stores(jlm.RoleFFNResScore, int32(li)))
	}
	s.Head = bankRead(c, s.Head, bankStreams(final), jlm.RoleOutputResScore, stores(jlm.RoleOutputResScore, jlm.DenseBlock))
	c.Streams = s
}

// sourceStores says s holds role for block li (-1 the dense tensors).
func sourceStores(s *jlm.Source) func(jlm.Role, int32) bool {
	held := map[[2]int32]bool{}
	for _, t := range s.Tensors {
		held[[2]int32{int32(t.Role), t.Block}] = true
	}
	return func(r jlm.Role, li int32) bool { return held[[2]int32{int32(r), li}] }
}

// bankStreams is streams 1 to n, each read as itself.
func bankStreams(n int) []int8 {
	from := make([]int8, n)
	for k := range from {
		from[k] = int8(1 + k)
	}
	return from
}

// bankRead is u reading the bank from and the running residual through a
// mix scored by role, where the block stores it: each row RMS-normed with
// no weight, scored by the sum of role's vector times it, the scores'
// softmax weighting the rows. A block that stores no score reads the
// running residual.
func bankRead(c *jlm.Config, u jlm.SubStreams, from []int8, role jlm.Role, stored bool) jlm.SubStreams {
	if !stored {
		return u
	}
	u.From = append(from, jlm.StreamRun)
	ns := len(u.From)
	p := &jlm.Program{NIn: uint16(ns)}
	op := func(o jlm.TowerOp) uint16 {
		p.Ops = append(p.Ops, o)
		return uint16(ns + len(p.Ops) - 1)
	}
	scores := make([]uint16, ns)
	for j := range ns {
		nv := op(jlm.TowerOp{Code: jlm.TOpRMSNorm, In: []uint16{uint16(j)}, Eps: c.RMSEps})
		sv := op(jlm.TowerOp{Code: jlm.TOpScale, In: []uint16{nv}, Bind: []jlm.Role{role}})
		scores[j] = op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{sv}})
	}
	sc := op(jlm.TowerOp{Code: jlm.TOpConcat, In: scores})
	mx := op(jlm.TowerOp{Code: jlm.TOpMax, In: []uint16{sc}})
	var e jlm.ExprBuilder
	e.Fn(jlm.FnExp, e.Fn(jlm.FnSub, jlm.Operand(0), jlm.Operand(1)))
	ex := op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{sc, mx}, Width: int32(ns), Expr: &jlm.Expr{Terms: e.T}})
	sum := op(jlm.TowerOp{Code: jlm.TOpSum, In: []uint16{ex}})
	var d jlm.ExprBuilder
	d.Fn(jlm.FnDiv, jlm.Operand(0), jlm.Operand(1))
	pr := op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{ex, sum}, Width: int32(ns), Expr: &jlm.Expr{Terms: d.T}})
	w := int32(c.NEmbd)
	var acc uint16
	for j := range ns {
		pj := op(jlm.TowerOp{Code: jlm.TOpSlice, In: []uint16{pr}, Offset: int32(j), Width: 1})
		var m jlm.ExprBuilder
		if j == 0 {
			m.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1))
			acc = op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{pj, uint16(j)}, Width: w, Expr: &jlm.Expr{Terms: m.T}})
			continue
		}
		m.Fn(jlm.FnMul, jlm.Operand(0), jlm.Operand(1))
		part := op(jlm.TowerOp{Code: jlm.TOpStated, In: []uint16{pj, uint16(j)}, Width: w, Expr: &jlm.Expr{Terms: m.T}})
		acc = op(jlm.TowerOp{Code: jlm.TOpAdd, In: []uint16{acc, part}})
	}
	p.Out = acc
	u.Read = p
	return u
}
