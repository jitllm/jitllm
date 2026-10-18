package tier

import "github.com/jitllm/jitllm/format/jlm"

// The device program's rewrites (docs/design/composer.md, "The device
// program"): each replaces a pattern of device ops with one launch of a
// fused kernel, where the kernel exists for this block and scratch, and is
// recorded by name. They are the block encoder's decode fusions, stated over
// the program's ops and value uses instead of inside the encoder.

// Fused kinds, past the program's base kinds.
const (
	dkNormQuant    devKind = iota + numBaseKinds // norm and the quantize of its row (bs.rmsQ)
	dkAddNorm                                    // residual add and the norm of the sum (bs.addRms)
	dkAddNormQuant                               // the two, and the quantize (bs.addRmsQ)
	dkMatVecRes                                  // a matvec carrying the residual add (mv.res)
	dkMatVecGate                                 // up's matvec writing act(gate)*up (mv.gated)
	dkActQuant                                   // act(gate)*up and its quantize (bs.quantAct)
	dkQKV                                        // q, k and v in one launch (layer.qkv)
	dkRopeQKV                                    // q's rotary, k's into its page and v's copy (ropeQKV)
	dkGate2                                      // gate and up in one launch writing act(gate)*up (mv.gate2)
	dkGate2In                                    // the FFN norm, gate and up in one launch (mv.gate2In)
)

func init() {
	devKindName = append(devKindName[:numBaseKinds:numBaseKinds],
		"norm_quant", "add_norm", "add_norm_quant", "matvec_res", "matvec_gate", "act_quant", "qkv", "rope_qkv",
		"gate2", "gate2_in")
}

// devRewrite is one rewrite: its name and whether it pays and is legal for
// a block on a scratch (the fused kernel exists).
type devRewrite struct {
	name  string
	apply func(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool
}

// devRewrites is the order they are tried: each to a fixed point before the
// next, the order the encoder's decode takes them in.
var devRewrites = []devRewrite{
	{"qkv-grouped", rwQKV},
	{"rope-qkv-paged", rwRopeQKV},
	{"matvec-residual", rwMatVecRes},
	{"gate2in", rwGate2In},
	{"gate2", rwGate2},
	{"matvec-gate", rwMatVecGate},
	{"act-quant", rwActQuant},
	{"add-norm-quant", rwAddNormQuant},
	{"add-norm", rwAddNorm},
	{"norm-quant", rwNormQuant},
}

// fused is a block's program with every rewrite that applies on bs, and the
// names of those that did.
func (p *devProgram) fused(l *layer, bs *blockScratch, g *devTier) (*devProgram, []string) {
	q := p.clone()
	var took []string
	for _, rw := range devRewrites {
		n := 0
		for rw.apply(q, l, bs, g) {
			n++
		}
		if n > 0 {
			took = append(took, rw.name)
		}
	}
	if err := q.check(); err != nil {
		// A rewrite that broke a type is a bug in the rewrite; the unfused
		// program is still right.
		return p, []string{"refused: " + err.Error()}
	}
	return q, took
}

func (p *devProgram) clone() *devProgram {
	q := &devProgram{vals: append([]devType(nil), p.vals...), slots: append([]devSlot(nil), p.slots...),
		names: append([]string(nil), p.names...)}
	for _, o := range p.ops {
		o.Src = append([]int(nil), o.Src...)
		o.In = append([]devVal(nil), o.In...)
		o.Out = append([]devVal(nil), o.Out...)
		q.ops = append(q.ops, o)
	}
	return q
}

// uses is how many ops read v.
func (p *devProgram) uses(v devVal) int {
	n := 0
	for _, o := range p.ops {
		for _, x := range o.In {
			if x == v {
				n++
			}
		}
	}
	return n
}

// next is the index of the first op after i that reads v, or -1.
func (p *devProgram) next(i int, v devVal) int {
	for j := i + 1; j < len(p.ops); j++ {
		for _, x := range p.ops[j].In {
			if x == v {
				return j
			}
		}
	}
	return -1
}

// replace puts op o at i and drops the op at j (j > i).
func (p *devProgram) replace(i, j int, o devOp) {
	p.ops[i] = o
	p.ops = append(p.ops[:j], p.ops[j+1:]...)
}

func cat(a, b []int) []int { return append(append([]int(nil), a...), b...) }

// rwQKV: q, k and v's matvecs over one operand, as the grouped launch.
func rwQKV(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if l.qkv == nil || g.NoFuse {
		return false
	}
	for i := 0; i+2 < len(p.ops); i++ {
		a, b, c := p.ops[i], p.ops[i+1], p.ops[i+2]
		if a.Kind == dkMatVec && b.Kind == dkMatVec && c.Kind == dkMatVec &&
			a.Role == jlm.RoleAttnQ && b.Role == jlm.RoleAttnK && c.Role == jlm.RoleAttnV &&
			a.In[0] == b.In[0] && b.In[0] == c.In[0] {
			p.ops[i] = devOp{Kind: dkQKV, Src: cat(cat(a.Src, b.Src), c.Src), In: a.In, Out: []devVal{a.Out[0], b.Out[0], c.Out[0]}}
			p.ops = append(p.ops[:i+1], p.ops[i+3:]...)
			return true
		}
	}
	return false
}

// rwRopeQKV: q's rotary and the history write, one paged launch.
func rwRopeQKV(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	pk := bs.pkv
	if pk == nil || pk.ropeQKV == nil || g.NoFuse || bs.copyQ != nil {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkRopeQ {
			continue
		}
		for j := i + 1; j < len(p.ops); j++ {
			if w := p.ops[j]; w.Kind == dkKVWrite {
				p.replace(i, j, devOp{Kind: dkRopeQKV, Src: cat(o.Src, w.Src), In: []devVal{o.In[0], w.In[0], w.In[1]},
					Out: []devVal{o.Out[0], w.Out[0]}})
				return true
			}
		}
	}
	return false
}

// rwMatVecRes: a matvec whose output only a residual add reads, as the
// matvec carrying the add.
func rwMatVecRes(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	for i, o := range p.ops {
		if o.Kind != dkMatVec || (o.Role != jlm.RoleAttnOut && o.Role != jlm.RoleFFNDown) {
			continue
		}
		m := l.mvo
		if o.Role == jlm.RoleFFNDown {
			m = l.mvd
		}
		if m.res == nil || g.TableSplit && m.alt != nil || p.uses(o.Out[0]) != 1 {
			continue
		}
		j := p.next(i, o.Out[0])
		if j < 0 || p.ops[j].Kind != dkAdd {
			continue
		}
		add := p.ops[j]
		resid := add.In[0]
		if resid == o.Out[0] {
			resid = add.In[1]
		}
		p.replace(i, j, devOp{Kind: dkMatVecRes, Src: cat(o.Src, add.Src), In: []devVal{o.In[0], resid}, Out: add.Out, Role: o.Role})
		return true
	}
	return false
}

// rwMatVecGate: up's matvec and the act_mul over the gate, as up's Gate
// variant writing the product.
func rwMatVecGate(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if l.mvu.gated == nil || g.TableSplit && l.mvu.alt != nil {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkMatVec || o.Role != jlm.RoleFFNUp || p.uses(o.Out[0]) != 1 {
			continue
		}
		j := p.next(i, o.Out[0])
		if j < 0 || p.ops[j].Kind != dkActMul || p.ops[j].In[1] != o.Out[0] {
			continue
		}
		am := p.ops[j]
		p.replace(i, j, devOp{Kind: dkMatVecGate, Src: cat(o.Src, am.Src), In: []devVal{o.In[0], am.In[0]}, Out: am.Out, Role: o.Role})
		return true
	}
	return false
}

// rwActQuant: act_mul and the quantize of its product, one launch.
func rwActQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if bs.quantAct == nil || g.NoOperandFuse {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkActMul || i+1 >= len(p.ops) {
			continue
		}
		q := p.ops[i+1]
		if q.Kind == dkQuant && q.In[0] == o.Out[0] {
			p.replace(i, i+1, devOp{Kind: dkActQuant, Src: cat(o.Src, q.Src), In: o.In, Out: []devVal{o.Out[0], q.Out[0]}})
			return true
		}
	}
	return false
}

// addNorm finds a residual add whose sum a norm reads next.
func addNorm(p *devProgram) (int, int, bool) {
	for i, o := range p.ops {
		if o.Kind != dkAdd || i+1 >= len(p.ops) {
			continue
		}
		if n := p.ops[i+1]; n.Kind == dkNorm && n.In[0] == o.Out[0] {
			return i, i + 1, true
		}
	}
	return 0, 0, false
}

// rwAddNormQuant: the add, the norm of the sum and its quantize, one launch.
func rwAddNormQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if bs.addRmsQ == nil || bs.normVar != nil {
		return false
	}
	i, j, ok := addNorm(p)
	if !ok || j+1 >= len(p.ops) || p.ops[j+1].Kind != dkQuant || p.ops[j+1].In[0] != p.ops[j].Out[0] {
		return false
	}
	a, n, q := p.ops[i], p.ops[j], p.ops[j+1]
	p.ops[i] = devOp{Kind: dkAddNormQuant, Src: cat(cat(a.Src, n.Src), q.Src), In: a.In,
		Out: []devVal{a.Out[0], n.Out[0], q.Out[0]}, Role: n.Role}
	p.ops = append(p.ops[:i+1], p.ops[j+2:]...)
	return true
}

// rwAddNorm: the add and the norm of the sum, one launch.
func rwAddNorm(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if bs.addRms == nil || bs.normVar != nil {
		return false
	}
	i, j, ok := addNorm(p)
	if !ok {
		return false
	}
	a, n := p.ops[i], p.ops[j]
	p.replace(i, j, devOp{Kind: dkAddNorm, Src: cat(a.Src, n.Src), In: a.In, Out: []devVal{a.Out[0], n.Out[0]}, Role: n.Role})
	return true
}

// rwNormQuant: a norm and the quantize of its row, one launch.
func rwNormQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if bs.rmsQ == nil || bs.normVar != nil {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkNorm || i+1 >= len(p.ops) {
			continue
		}
		if q := p.ops[i+1]; q.Kind == dkQuant && q.In[0] == o.Out[0] {
			p.replace(i, i+1, devOp{Kind: dkNormQuant, Src: cat(o.Src, q.Src), In: o.In, Out: []devVal{o.Out[0], q.Out[0]}, Role: o.Role})
			return true
		}
	}
	return false
}

// gateUp finds gate's matvec followed by up's over the same operand, and the
// act_mul reading them; -1s where the program has none.
func gateUp(p *devProgram) (gi, ui, ai int) {
	for i := 0; i+2 < len(p.ops); i++ {
		g, u, a := p.ops[i], p.ops[i+1], p.ops[i+2]
		if g.Kind == dkMatVec && g.Role == jlm.RoleFFNGate && u.Kind == dkMatVec && u.Role == jlm.RoleFFNUp &&
			g.In[0] == u.In[0] && a.Kind == dkActMul && a.In[0] == g.Out[0] && a.In[1] == u.Out[0] &&
			p.uses(g.Out[0]) == 1 && p.uses(u.Out[0]) == 1 {
			return i, i + 1, i + 2
		}
	}
	return -1, -1, -1
}

// rwGate2In: the FFN norm of a residual a fused matvec wrote, its quantize,
// gate and up, as one launch building its own operand (the gate2in recipe,
// where the card's recipe chose it).
func rwGate2In(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if !g.actInOn() || g.ffnFormOf(l) != FFNGate2In || l.mvu.gate2In == nil ||
		!g.gateUp2On(l, 1, false, false, false, false) {
		return false
	}
	gi, _, ai := gateUp(p)
	if gi < 2 {
		return false
	}
	n, q := p.ops[gi-2], p.ops[gi-1]
	if n.Kind != dkNorm || n.Role != jlm.RoleFFNNorm || q.Kind != dkQuant || q.In[0] != n.Out[0] ||
		p.uses(n.Out[0]) != 1 || p.uses(q.Out[0]) != 2 {
		return false
	}
	// Behind a fused residual only: otherwise the norm's launch is the add.
	src := -1
	for k := 0; k < gi-2; k++ {
		for _, x := range p.ops[k].Out {
			if x == n.In[0] {
				src = k
			}
		}
	}
	if src < 0 || p.ops[src].Kind != dkMatVecRes {
		return false
	}
	a := p.ops[ai]
	o := devOp{Kind: dkGate2In, Src: cat(cat(cat(n.Src, q.Src), cat(p.ops[gi].Src, p.ops[gi+1].Src)), a.Src),
		In: []devVal{n.In[0]}, Out: a.Out, Role: jlm.RoleFFNUp}
	p.ops[gi-2] = o
	p.ops = append(p.ops[:gi-1], p.ops[ai+1:]...)
	return true
}

// rwGate2: gate and up over one operand and their act_mul, one launch (Gate2,
// where the card picked it).
func rwGate2(p *devProgram, l *layer, bs *blockScratch, g *devTier) bool {
	if !g.gateUp2On(l, 1, false, false, false, false) {
		return false
	}
	gi, ui, ai := gateUp(p)
	if gi < 0 {
		return false
	}
	a := p.ops[ai]
	o := devOp{Kind: dkGate2, Src: cat(cat(p.ops[gi].Src, p.ops[ui].Src), a.Src), In: p.ops[gi].In, Out: a.Out, Role: jlm.RoleFFNUp}
	p.ops[gi] = o
	p.ops = append(p.ops[:gi+1], p.ops[ai+1:]...)
	return true
}
