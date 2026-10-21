package tier

import (
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// genSegments is a segmented launch's one kernel (compose.EmitSegments):
// each matvec of o formed as genMatVecNodes forms one at decode or over a
// ragged step's few rows, the int8 operand they share the launch's first
// parameters. Callers hold g.mu.
func (g *devTier) genSegments(o *devOp, prog *devProgram, rows int) []genLaunch {
	l := g.curLayer
	if l == nil || len(o.Ws) < 2 || len(o.Ws) > maxSegs || len(o.Out) != len(o.Ws) {
		return nil
	}
	live, toks := 1, 1
	if g.genRagLive > 0 {
		live, toks = g.genRagLive, rows
	} else if rows != 1 {
		return nil
	}
	split := max((*l.ms[o.Ws[0]]).split, 1)
	var segs []compose.Segment
	args := []genArg{gaOpA, gaOpAX}
	enc := []uint32{encSegments}
	blocks := 0
	for si, wi := range o.Ws {
		m := *l.ms[wi]
		rb := &rowBuild{r: compose.Row{Width: m.rows, Rows: live}}
		seg := gaSeg0 + genArg(4*si)
		w := compose.MatVecWeight{T: m.q, K: m.k, Rows: m.rows, QS: rb.buf(compose.BufU32, seg), D: rb.buf(compose.BufU32, seg+1), SC: -1}
		if m.q.Info().PerSuper > 1 {
			w.SC = rb.buf(compose.BufU32, seg+2)
		}
		a := compose.MatVecOperand{Toks: toks, A: rb.buf(compose.BufU32, gaOpA), AX: rb.buf(compose.BufRow, gaOpAX)}
		v, err := rb.r.MatVec(w, a)
		if err != nil {
			return nil
		}
		if m.bias != nil {
			v = rb.bin(compose.OpAdd, v, rb.load(compose.BufVec, seg+3))
		}
		rb.store(v, gaOut0+genArg(si))
		f := compose.Facts{Target: g.composeTarget(), Step: compose.StepDecode, DecodeSplit: split, DecodeGroup: true}
		if g.genRagLive > 0 {
			f.Step, f.RagGroup = compose.StepRagged, true
		}
		ps, err := rb.r.Plan(f)
		if err != nil || len(ps) != 1 || ps[0].S.Split != split && split > 1 {
			return nil
		}
		p := ps[0]
		share := make([]int, len(rb.args))
		for j, x := range rb.args {
			share[j] = -1
			switch x {
			case gaOpA:
				share[j] = 0
			case gaOpAX:
				share[j] = 1
			default:
				args = append(args, x)
			}
		}
		n := (p.Threads + p.S.Group - 1) / p.S.Group
		segs = append(segs, compose.Segment{G: p.G, S: p.S, Blocks: n, Share: share})
		enc = append(enc, uint32(n))
		enc = append(enc, p.G.Encode(p.S)...)
		blocks += n
	}
	if len(args) > len(genLaunch{}.args) || kernels.IsFloat((*l.ms[o.Ws[0]]).q) {
		return nil
	}
	k, _ := g.genKern(enc, func() (*ir.Kernel, error) { return compose.EmitSegments(segs, 2) })
	if k == nil {
		return nil
	}
	gl := genLaunch{k: k, groups: blocks, width: segs[0].S.Group, n: uint8(len(args))}
	copy(gl.args[:], args)
	return []genLaunch{gl}
}
