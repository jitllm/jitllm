package tier

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Clef's joint schema head on the device (nn.ClefHeadDevice): the request's
// final residual rows go up, the head runs as a program of mainkernels.DH*
// launches on the device that holds the model's output projection -- where
// the backbone's last rows came from -- and one logit per option comes back.
// The head's matrices are uploaded once per head (f32, as the model hands
// them) and charged against the device's budget; the kernels are compiled per
// request shape, before the submission, and kept.

// dhHead is a head's buffers on one device.
type dhHead struct {
	h     *nn.ClefHead
	bufs  map[*float32]backend.Buf // by the first element of the host slice
	bytes uint64
}

// dhState is the device's Clef state: the resident heads, the compiled
// kernels and the programs of the request shapes seen last.
type dhState struct {
	heads map[*nn.ClefHead]*dhHead
	kerns map[dhKey]backend.Kernel
	progs []*dhProg
}

type dhKey struct {
	name string
	a    [10]int
	f    [3]float32
	b    [2]bool
}

// ClefHead runs the head on the device holding the output projection.
func (g *GPU) ClefHead(h *nn.ClefHead, in *nn.ClefHeadInput, out []float32) bool {
	return false
}

// ClefHead is the session's device's.
func (s *gpuSession) ClefHead(h *nn.ClefHead, in *nn.ClefHeadInput, out []float32) bool {
	defer s.leave(s.step())
	return s.g.ClefHead(h, in, out)
}

// dhUpload puts h's matrices and vectors on the device, once.
func (g *devTier) dhUpload(h *nn.ClefHead) (*dhHead, bool) {
	if g.dh.heads == nil {
		g.dh.heads = map[*nn.ClefHead]*dhHead{}
	}
	if r, ok := g.dh.heads[h]; ok {
		return r, true
	}
	var all [][]float32
	add := func(vs ...[]float32) {
		for _, v := range vs {
			if len(v) != 0 {
				all = append(all, v)
			}
		}
	}
	add(h.OutNorm, h.OutNormB, h.HidW, h.HidB, h.PMem, h.PQ, h.POQ, h.PG, h.POC, h.POL, h.SumW, h.SumB, h.FieldW, h.FieldB,
		h.OptW, h.OptB, h.Types, h.Sc, h.ScB, h.ScOut, h.ScOutB)
	for i := range h.Blocks {
		b := &h.Blocks[i]
		add(b.SelfW, b.SelfB, b.SQ, b.SK, b.SV, b.SO, b.SBQ, b.SBK, b.SBV, b.SBO, b.CrossW, b.CrossB, b.KVW, b.KVB,
			b.CQ, b.CK, b.CV, b.CO, b.CBQ, b.CBK, b.CBV, b.CBO, b.FFNW, b.FFNB, b.Up, b.BUp, b.Down, b.BDown)
	}
	var need uint64
	for _, v := range all {
		need += uint64(4 * len(v))
	}
	// The head is the model's, as a block is: a card short of room pages
	// blocks out for it (they fault back in when next run), and refuses only
	// when nothing it holds can go.
	g.kvCompact(need, nil)
	if !g.room(need) && !g.reclaimFor(need) {
		g.LastErr = fmt.Sprintf("the decision head needs %d bytes and nothing can be paged out (%d of %d used)",
			need, g.used, g.limit)
		return nil, false
	}
	r := &dhHead{h: h, bufs: map[*float32]backend.Buf{}}
	for _, v := range all {
		if _, ok := r.bufs[&v[0]]; ok {
			continue
		}
		b, err := g.dev.Alloc(4 * len(v))
		if err == nil {
			err = b.Write(f32b(v))
		}
		if err != nil {
			if b != nil {
				b.Free()
			}
			r.free()
			g.LastErr = fmt.Sprintf("the decision head: %v", err)
			return nil, false
		}
		r.bufs[&v[0]] = b
		r.bytes += uint64(4 * len(v))
	}
	g.charge(r.bytes)
	g.dh.heads[h] = r
	return r, true
}

func (r *dhHead) free() {
	for _, b := range r.bufs {
		b.Free()
	}
	r.bufs = nil
}

// dhClose gives back every head and kernel. Callers hold g.mu.
func (g *devTier) dhClose() {
	for _, r := range g.dh.heads {
		g.refund(r.bytes)
		r.free()
	}
	for _, k := range g.dh.kerns {
		if k != nil {
			k.Close()
		}
	}
	for _, p := range g.dh.progs {
		p.free()
	}
	g.dh = dhState{}
}

// dhKern is the compiled kernel for key. Callers hold g.mu.
func (g *devTier) dhKern(key dhKey, build func() (*ir.Kernel, error)) (backend.Kernel, error) {
	if k, ok := g.dh.kerns[key]; ok {
		if k == nil {
			return nil, fmt.Errorf("%s does not compile here", key.name)
		}
		return k, nil
	}
	if g.dh.kerns == nil {
		g.dh.kerns = map[dhKey]backend.Kernel{}
	}
	ker, err := build()
	var k backend.Kernel
	if err == nil {
		k, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.dh.kerns[key] = nil
		return nil, fmt.Errorf("%s: %w", key.name, err)
	}
	g.dh.kerns[key] = k
	return k, nil
}

// dhNewBuf is a buffer of n floats, poisoned (RULE 13). Callers hold g.mu.
func (g *devTier) dhNewBuf(n int) (backend.Buf, error) {
	n = max(n, 1)
	b, err := g.dev.Alloc(4 * n)
	if err != nil {
		return nil, err
	}
	poison := make([]float32, n)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := b.Write(f32b(poison)); err != nil {
		b.Free()
		return nil, err
	}
	return b, nil
}

// dhStep is one launch of the head's program.
type dhStep struct {
	k       backend.Kernel
	threads int
	bufs    []backend.Buf
}

// dhProg is one request shape's program: its launches, the buffers they
// run against (its own, so another shape's growth never moves them), and
// the inputs' constant parts. Kept per shape (dhProgs), so a warm request
// writes its inputs, launches and reads back, allocating nothing.
type dhProg struct {
	key                            dhShape
	prog                           []dhStep
	own                            []backend.Buf
	resid, pool, lex, owner, qtype backend.Buf
	zero, mask, logits             backend.Buf
	sel                            [3]backend.Buf
	selB                           [3][]byte
	zeroB                          []byte
	in                             *nn.ClefHeadInput
	out                            []float32
	err                            error
	run                            func(backend.Session)
}

// dhShape keys a program: the head and the request's rows, questions and
// options.
type dhShape struct {
	h         *nn.ClefHead
	n, nq, no int
}

// dhProgs is how many request shapes a device keeps programs for; the oldest
// goes first.
const dhProgs = 8

func (p *dhProg) free() {
	for _, b := range p.own {
		b.Free()
	}
	p.own = nil
}

// dhRun is a program's submission: the inputs up, the launches, the logits
// back.
func (p *dhProg) dhRun(s backend.Session) {
	in := p.in
	w := func(b backend.Buf, x []byte) {
		if p.err == nil {
			p.err = s.Write(b, x)
		}
	}
	w(p.resid, f32b(in.Resid))
	w(p.pool, f32b(in.Pool))
	w(p.lex, f32b(in.Lex))
	w(p.owner, u32b(in.Owner))
	w(p.qtype, u32b(in.QType))
	w(p.mask, f32b(in.Mask))
	w(p.zero, p.zeroB)
	for i := range p.sel {
		w(p.sel[i], p.selB[i])
	}
	for i := range p.prog {
		if p.err != nil {
			return
		}
		st := &p.prog[i]
		p.err = s.Launch(st.k, (st.threads+127)/128, 128, st.bufs...)
	}
	if p.err == nil {
		p.err = s.Read(p.logits, f32b(p.out))
	}
}
