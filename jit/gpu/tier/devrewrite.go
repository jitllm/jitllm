package tier

import (
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

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
	dkQKVIn                                      // the norm, its quantize and q, k and v, one launch (layer.qkvIn)
	dkQKVSep                                     // q, k and v each normalising and quantizing (mv.kernIn)
	dkResIn                                      // a matvec quantizing its own operand, with the residual (mv.resIn)
	dkFFNIn                                      // gate and up normalising and quantizing the residual (mv.kernIn, gatedIn)
	dkDownAct                                    // down forming act(gate)*up as it quantizes, with the residual (mv.resActIn)
	dkNormF16                                    // a norm writing its row and the next GEMM's binary16 operand (rmsF16For)
	dkAddNormF16                                 // the residual add, its norm and the operand (rmsF16For, add)
	dkActF16                                     // act(gate)*up written as down's binary16 operand only (actF16For)
	dkQKVGemm                                    // q, k and v as one binary16 grid over one operand (qkvGemm)
	numKinds
)

// String names a kind for the program's dump; the executor and the rewrites
// switch on the kinds themselves. TestEveryDevKindHasAName holds it to the
// enum.
func (k devKind) String() string {
	switch k {
	case dkNorm:
		return "norm"
	case dkQuant:
		return "quant"
	case dkMatVec:
		return "matvec"
	case dkRopeQ:
		return "rope_q"
	case dkKVWrite:
		return "kv_write"
	case dkAttend:
		return "attend"
	case dkAdd:
		return "add"
	case dkActMul:
		return "act_mul"
	case dkHeadNorm:
		return "head_norm"
	case dkPostNorm:
		return "post_norm"
	case dkNormQuant:
		return "norm_quant"
	case dkAddNorm:
		return "add_norm"
	case dkAddNormQuant:
		return "add_norm_quant"
	case dkMatVecRes:
		return "matvec_res"
	case dkMatVecGate:
		return "matvec_gate"
	case dkActQuant:
		return "act_quant"
	case dkQKV:
		return "qkv"
	case dkRopeQKV:
		return "rope_qkv"
	case dkGate2:
		return "gate2"
	case dkGate2In:
		return "gate2_in"
	case dkQKVIn:
		return "qkv_in"
	case dkQKVSep:
		return "qkv_sep"
	case dkResIn:
		return "matvec_res_in"
	case dkFFNIn:
		return "ffn_in"
	case dkDownAct:
		return "down_act"
	case dkNormF16:
		return "norm_f16"
	case dkAddNormF16:
		return "add_norm_f16"
	case dkActF16:
		return "act_f16"
	case dkQKVGemm:
		return "qkv_gemm"
	}
	return ""
}

// devRewrite is one rewrite: its name and whether it pays and is legal for
// a block on a scratch at R rows, of nseq sequences on a ragged step (0
// otherwise): the fused kernel exists for that shape.
type devRewrite struct {
	name  string
	apply func(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool
}

// newDevRewrites is a device's rewrites, in the order they are tried: each
// to a fixed point before the next, the order the encoder takes them in. A
// program's rewrites taken are a bit each (rwSet), by position here: the
// f16 lowering first (a prompt chunk of binary16 GEMMs takes it and its
// writers alone), then the rest. Each device holds its own list (devShared).
func newDevRewrites() []devRewrite {
	return []devRewrite{
		{"f16-operands", nil},
		{"qkv-gemm", rwQKVGemm},
		{"add-norm-f16", rwAddNormF16},
		{"norm-f16", rwNormF16},
		{"act-f16", rwActF16},
		{"qkv-grouped", rwQKV},
		{"qkv-in", rwQKVIn},
		{"qkv-sep", rwQKVSep},
		{"rope-qkv-paged", rwRopeQKV},
		{"matvec-residual", rwMatVecRes},
		{"o-in", rwOIn},
		{"ffn-in", rwFFNIn},
		{"gate2in", rwGate2In},
		{"gate2", rwGate2},
		{"matvec-gate", rwMatVecGate},
		{"down-in", rwDownIn},
		{"down-act", rwDownAct},
		{"act-quant", rwActQuant},
		{"add-norm-quant", rwAddNormQuant},
		{"add-norm", rwAddNorm},
		{"norm-quant", rwNormQuant},
	}
}

// rwF16End is where the f16 lowering's rewrites end in devRewrites.
const rwF16End = 5

// rwSet is the rewrites a program took, a bit each by devRewrites' order.
type rwSet uint64

// names is s by the names of rws, the list it was taken from.
func (s rwSet) names(rws []devRewrite) string {
	var b strings.Builder
	for i, rw := range rws {
		if s&(1<<i) != 0 {
			if b.Len() > 0 {
				b.WriteString(", ")
			}
			b.WriteString(rw.name)
		}
	}
	return b.String()
}

// fused is a block's program with every rewrite that applies on bs, the
// rewrites it took, and the type error that refused them all, if one did.
func (p *devProgram) fused(l *layer, bs *blockScratch, g *devTier, R, nseq int) (*devProgram, rwSet, error) {
	q := p.clone()
	var took rwSet
	rws := g.rewrites
	lo, hi := rwF16End, len(rws)
	if R > 1 && nseq == 0 && g.voltaBlock(l, R, bs.tok) {
		q.toF16()
		took |= 1
		lo, hi = 1, rwF16End
	}
	if err := q.check(); err != nil {
		return p, 0, err
	}
	// Each application is held to the check on its own: one that breaks a
	// type or reads what it writes (RULE 13) is undone, and that rewrite
	// stops; the others still apply.
	var refused error
	for i := lo; i < hi; i++ {
		for {
			prev := q.clone()
			if !rws[i].apply(q, l, bs, g, R, nseq) {
				break
			}
			if err := q.check(); err != nil {
				q, refused = prev, err
				break
			}
			took |= 1 << i
		}
	}
	return q, took, refused
}

func (p *devProgram) clone() *devProgram {
	q := &devProgram{f16: p.f16, vals: append([]devType(nil), p.vals...), slots: append([]devSlot(nil), p.slots...),
		names: append([]string(nil), p.names...)}
	for _, o := range p.ops {
		o.Src = append([]int(nil), o.Src...)
		o.In = append([]devVal(nil), o.In...)
		o.Out = append([]devVal(nil), o.Out...)
		o.Need = append([]devType(nil), o.Need...)
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
func rwQKV(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	switch {
	case nseq > 0:
		// A ragged step's segments: q, k and v grouped over its rows.
		if g.TableSplit || l.vFromK {
			return false
		}
		if _, ok := g.groupQKV(l, nseq, R); !ok {
			return false
		}
	case R != 1:
		return false // a one-row variant
	case l.qkv == nil || g.NoFuse || g.TableSplit ||
		l.qkvBias != [3]bool{l.mvq.bias != nil, l.mvk.bias != nil, l.mvv.bias != nil}:
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
func rwRopeQKV(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
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
func rwMatVecRes(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 && nseq == 0 {
		return false // a one-row variant, or a ragged step's grouped one
	}
	for i, o := range p.ops {
		if o.Kind != dkMatVec || (o.Role != jlm.RoleAttnOut && o.Role != jlm.RoleFFNDown) {
			continue
		}
		m := l.mvo
		if o.Role == jlm.RoleFFNDown {
			m = l.mvd
		}
		if nseq > 0 {
			if _, ok := g.groupKern(m, nseq, R, groupRes, 0); !ok {
				continue
			}
		} else if m.res == nil || g.TableSplit && m.alt != nil {
			continue
		}
		if p.uses(o.Out[0]) != 1 || m.bias != nil {
			continue // the residual rides the bias slot
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
func rwMatVecGate(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	switch {
	case nseq > 0:
		if _, ok := g.groupKern(l.mvu, nseq, R, groupGate, bs.p.Act); !ok {
			return false
		}
	case R != 1:
		return false // a one-row variant, or a ragged step's grouped one
	case l.mvu.gated == nil || g.TableSplit && l.mvu.alt != nil:
		return false
	}
	if l.mvu.bias != nil || l.mvg.bias != nil {
		return false // the gated launch carries neither bias
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
func rwActQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
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
func rwAddNormQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
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
func rwAddNorm(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
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
func rwNormQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
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
func rwGate2In(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
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
	if src < 0 || p.ops[src].Kind != dkMatVecRes && p.ops[src].Kind != dkResIn {
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
func rwGate2(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
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

// attnPro reports that the attention half's operand prologue may run: the
// norm and quantize of the block input fold into what reads them.
func attnPro(l *layer, bs *blockScratch, g *devTier) bool {
	return g.actInOn() && l.nAttnB == nil && l.nAttn != nil && bs.normVar == nil && !g.TableSplit
}

// normQuantBefore reports that ops i-2 and i-1 are a norm of role and the
// quantize of its row, read by nothing but the ops from i on.
func normQuantBefore(p *devProgram, i int, role jlm.Role, uses int) bool {
	if i < 2 {
		return false
	}
	n, q := p.ops[i-2], p.ops[i-1]
	return n.Kind == dkNorm && n.Role == role && q.Kind == dkQuant && q.In[0] == n.Out[0] &&
		p.uses(n.Out[0]) == 1 && p.uses(q.Out[0]) == uses
}

// rwQKVIn: the attention norm, its quantize and the grouped q/k/v, as the
// grouped launch normalising and quantizing the block input itself.
func rwQKVIn(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
	if l.qkvIn == nil || !attnPro(l, bs, g) ||
		l.qkvBias != [3]bool{l.mvq.bias != nil, l.mvk.bias != nil, l.mvv.bias != nil} {
		return false
	}
	for i, o := range p.ops {
		if o.Kind == dkQKV && normQuantBefore(p, i, jlm.RoleAttnNorm, 1) {
			n := p.ops[i-2]
			p.ops[i-2] = devOp{Kind: dkQKVIn, Src: cat(cat(n.Src, p.ops[i-1].Src), o.Src), In: n.In, Out: o.Out}
			p.ops = append(p.ops[:i-1], p.ops[i+1:]...)
			return true
		}
	}
	return false
}

// rwQKVSep: the attention norm, its quantize and three q/k/v matvecs, as
// three launches each normalising and quantizing the block input.
func rwQKVSep(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
	if l.qkv != nil || l.mvq.kernIn == nil || l.mvk.kernIn == nil || l.mvv.kernIn == nil || !attnPro(l, bs, g) {
		return false
	}
	for i := 2; i+2 < len(p.ops); i++ {
		a, b, c := p.ops[i], p.ops[i+1], p.ops[i+2]
		if a.Kind == dkMatVec && a.Role == jlm.RoleAttnQ && b.Kind == dkMatVec && b.Role == jlm.RoleAttnK &&
			c.Kind == dkMatVec && c.Role == jlm.RoleAttnV && normQuantBefore(p, i, jlm.RoleAttnNorm, 3) {
			n := p.ops[i-2]
			p.ops[i-2] = devOp{Kind: dkQKVSep, Src: cat(cat(n.Src, p.ops[i-1].Src), cat(cat(a.Src, b.Src), c.Src)),
				In: n.In, Out: []devVal{a.Out[0], b.Out[0], c.Out[0]}}
			p.ops = append(p.ops[:i-1], p.ops[i+3:]...)
			return true
		}
	}
	return false
}

// resIn rewrites quantize + matvec_res of role into the matvec quantizing
// its own operand, where m's prologue variant exists.
func resIn(p *devProgram, role jlm.Role, ok func(src devVal) bool) bool {
	for i := 1; i < len(p.ops); i++ {
		o, q := p.ops[i], p.ops[i-1]
		if o.Kind == dkMatVecRes && o.Role == role && q.Kind == dkQuant && o.In[0] == q.Out[0] &&
			p.uses(q.Out[0]) == 1 && ok(q.In[0]) {
			p.ops[i-1] = devOp{Kind: dkResIn, Src: cat(q.Src, o.Src), In: []devVal{q.In[0], o.In[1]}, Out: o.Out, Role: role}
			p.ops = append(p.ops[:i], p.ops[i+1:]...)
			return true
		}
	}
	return false
}

// rwOIn: the attention output's quantize and its matvec with the residual,
// as the matvec quantizing its own operand.
func rwOIn(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
	if !g.actInOn() || l.mvo.resIn == nil || l.mvo.bias != nil || l.mvo.k != bs.p.NHead*bs.p.HeadDim || g.TableSplit && l.mvo.alt != nil {
		return false
	}
	return resIn(p, jlm.RoleAttnOut, func(devVal) bool { return true })
}

// rwFFNIn: the FFN norm, its quantize, gate and up (gated or not), as gate
// and up normalising and quantizing the residual themselves.
func rwFFNIn(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
	if !g.actInOn() || l.mvg.kernIn == nil || (l.mvu.gatedIn == nil && l.mvu.kernIn == nil) || l.mvg.bias != nil || l.mvu.bias != nil ||
		l.nFFNB != nil || l.nFFN == nil || bs.normVar != nil || g.TableSplit && (l.mvg.alt != nil || l.mvu.alt != nil) {
		return false
	}
	gi, ui, ai := gateUp(p)
	if gi < 0 || !normQuantBefore(p, gi, jlm.RoleFFNNorm, 2) {
		return false
	}
	n, gt, up, am := p.ops[gi-2], p.ops[gi], p.ops[ui], p.ops[ai]
	src := cat(cat(n.Src, p.ops[gi-1].Src), cat(gt.Src, up.Src))
	if l.mvu.gatedIn != nil {
		// Up forms act(gate)*up in its own store: the act_mul goes too.
		p.ops[gi-2] = devOp{Kind: dkFFNIn, Src: cat(src, am.Src), In: n.In, Out: []devVal{gt.Out[0], am.Out[0]}, Role: jlm.RoleFFNUp}
		p.ops = append(p.ops[:gi-1], p.ops[ai+1:]...)
		return true
	}
	p.ops[gi-2] = devOp{Kind: dkFFNIn, Src: src, In: n.In, Out: []devVal{gt.Out[0], up.Out[0]}}
	p.ops = append(p.ops[:gi-1], p.ops[ui+1:]...)
	return true
}

// gatedProduct says v is act(gate)*up written by a gated launch.
func gatedProduct(p *devProgram, v devVal) bool {
	for _, o := range p.ops {
		for _, x := range o.Out {
			if x == v {
				return o.Kind == dkMatVecGate || o.Kind == dkGate2 || o.Kind == dkGate2In ||
					o.Kind == dkFFNIn && o.Role == jlm.RoleFFNUp
			}
		}
	}
	return false
}

func downOK(l *layer, bs *blockScratch, g *devTier) bool {
	return g.actInOn() && l.mvd.bias == nil && l.mvd.k == bs.p.NFFN && !(g.TableSplit && l.mvd.alt != nil)
}

// rwDownIn: the quantize of a gated product and down with the residual, as
// down quantizing its own operand.
func rwDownIn(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
	if l.mvd.resIn == nil || !downOK(l, bs, g) {
		return false
	}
	return resIn(p, jlm.RoleFFNDown, func(src devVal) bool { return gatedProduct(p, src) })
}

// rwDownAct: act_mul over a prologue's gate and up, its quantize and down
// with the residual, as down forming the product as it quantizes.
func rwDownAct(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false // a one-row variant
	}
	if l.mvd.resActIn == nil || !downOK(l, bs, g) {
		return false
	}
	for i := 2; i < len(p.ops); i++ {
		am, q, d := p.ops[i-2], p.ops[i-1], p.ops[i]
		if am.Kind == dkActMul && q.Kind == dkQuant && q.In[0] == am.Out[0] && d.Kind == dkMatVecRes &&
			d.Role == jlm.RoleFFNDown && d.In[0] == q.Out[0] && p.uses(am.Out[0]) == 1 && p.uses(q.Out[0]) == 1 {
			p.ops[i-2] = devOp{Kind: dkDownAct, Src: cat(cat(am.Src, q.Src), d.Src), In: []devVal{am.In[0], am.In[1], d.In[1]},
				Out: d.Out, Role: jlm.RoleFFNDown}
			p.ops = append(p.ops[:i-1], p.ops[i+1:]...)
			return true
		}
	}
	return false
}

// toF16 makes the program's matvecs binary16 GEMMs over the float rows: each
// int8 quantize goes and its readers read the row it quantized.
func (p *devProgram) toF16() {
	p.f16 = true
	for i := 0; i < len(p.ops); {
		o := p.ops[i]
		if o.Kind != dkQuant {
			i++
			continue
		}
		for j := range p.ops {
			for k, x := range p.ops[j].In {
				if x == o.Out[0] {
					p.ops[j].In[k] = o.In[0]
				}
			}
		}
		p.ops = append(p.ops[:i], p.ops[i+1:]...)
	}
}

// twinLayout is the binary16 layout m's batched twin at R rows reads.
func twinLayout(g *devTier, bs *blockScratch, m mv, R int) (devLayout, bool) {
	bm, ok := g.batchMV(m, R, bs.tok)
	if !ok || bm.act == nil {
		return 0, false
	}
	if bm.actT {
		return layoutTokF16, true
	}
	return layoutFrag, true
}

// readsOperand adds the operand a binary16 writer wrote as the GEMM op j's
// second input, which the GEMM reads in its twin's layout: the check holds
// the writer's layout to it.
func readsOperand(p *devProgram, g *devTier, bs *blockScratch, j int, m mv, R int, op devVal) bool {
	lay, ok := twinLayout(g, bs, m, R)
	if !ok {
		return false
	}
	o := &p.ops[j]
	need := make([]devType, len(o.In)+1)
	need[len(o.In)] = devType{Elem: elemF16, Layout: lay, Width: p.vals[op].Width}
	o.In = append(o.In, op)
	o.Need = need
	return true
}

// rwNormF16: a norm whose first reader is a GEMM, writing that GEMM's
// binary16 operand too (kernels.RMSNormF16TRows, token-major).
func rwNormF16(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	for i, o := range p.ops {
		if o.Kind != dkNorm || len(o.Out) != 1 {
			continue
		}
		j := p.next(i, o.Out[0])
		if j < 0 || p.ops[j].Kind != dkMatVec && p.ops[j].Kind != dkQKVGemm {
			continue
		}
		m, _ := mvOfLayer(l, p.ops[j].Role) // q's for the grid, whose operand is q's
		nk, conv := g.rmsF16For(bs, m, R, false)
		if nk == nil {
			continue
		}
		op := p.val(p.names[o.Out[0]]+".f16", devType{Elem: elemF16, Layout: layoutTokF16, Width: p.vals[o.Out[0]].Width}, slotF16)
		p.ops[i] = devOp{Kind: dkNormF16, Src: o.Src, In: o.In, Out: []devVal{o.Out[0], op}, Role: o.Role, K: nk, Conv: conv}
		if !readsOperand(p, g, bs, j, m, R, op) {
			return false
		}
		return true
	}
	return false
}

// rwAddNormF16: a residual add, the norm of the sum and the next GEMM's
// binary16 operand, one launch (kernels.RMSNormF16TRows with add).
func rwAddNormF16(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	i, n, ok := addNorm(p)
	if !ok {
		return false
	}
	a, no := p.ops[i], p.ops[n]
	j := p.next(n, no.Out[0])
	if j < 0 || p.ops[j].Kind != dkMatVec {
		return false
	}
	m, _ := mvOfLayer(l, p.ops[j].Role)
	nk, conv := g.rmsF16For(bs, m, R, true)
	if nk == nil {
		return false
	}
	op := p.val(p.names[no.Out[0]]+".f16", devType{Elem: elemF16, Layout: layoutTokF16, Width: p.vals[no.Out[0]].Width}, slotF16)
	p.replace(i, n, devOp{Kind: dkAddNormF16, Src: cat(a.Src, no.Src), In: a.In, Out: []devVal{a.Out[0], no.Out[0], op},
		Role: no.Role, K: nk, Conv: conv})
	return readsOperand(p, g, bs, j-1, m, R, op)
}

// rwActF16: act_mul whose one reader is down's GEMM, written as down's
// binary16 operand alone (kernels.ActMulF16T): its float row is never written.
func rwActF16(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	for i, o := range p.ops {
		if o.Kind != dkActMul || p.uses(o.Out[0]) != 1 {
			continue
		}
		j := p.next(i, o.Out[0])
		if j < 0 || p.ops[j].Kind != dkMatVec || p.ops[j].Role != jlm.RoleFFNDown {
			continue
		}
		ak, conv, threads := g.actF16For(bs, l.mvd, R)
		if ak == nil {
			continue
		}
		op := p.val(p.names[o.Out[0]]+".f16", devType{Elem: elemF16, Layout: layoutTokF16, Width: p.vals[o.Out[0]].Width}, slotF16)
		p.ops[i] = devOp{Kind: dkActF16, Src: o.Src, In: o.In, Out: []devVal{o.Out[0], op}, K: ak, Conv: conv, N: threads}
		return readsOperand(p, g, bs, j, l.mvd, R, op)
	}
	return false
}

// mvOfLayer is the matvec and weight of role in l.
func mvOfLayer(l *layer, role jlm.Role) (mv, *resident) {
	switch role {
	case jlm.RoleAttnK:
		return l.mvk, l.wk
	case jlm.RoleAttnV:
		return l.mvv, l.wv
	case jlm.RoleAttnOut:
		return l.mvo, l.wo
	case jlm.RoleFFNGate:
		return l.mvg, l.gate
	case jlm.RoleFFNUp:
		return l.mvu, l.up
	case jlm.RoleFFNDown:
		return l.mvd, l.down
	}
	return l.mvq, l.wq
}

// rwQKVGemm: q, k and v's GEMMs over one float row, as one binary16 grid
// (GemmVoltaSegments) reading one token-major operand.
func rwQKVGemm(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if nseq > 0 || l.vFromK {
		return false
	}
	for i := 0; i+2 < len(p.ops); i++ {
		a, b, c := p.ops[i], p.ops[i+1], p.ops[i+2]
		if a.Kind != dkMatVec || b.Kind != dkMatVec || c.Kind != dkMatVec || a.Role != jlm.RoleAttnQ ||
			b.Role != jlm.RoleAttnK || c.Role != jlm.RoleAttnV || a.In[0] != b.In[0] || b.In[0] != c.In[0] {
			continue
		}
		qs, ok := g.qkvGemm(l, R, false)
		if !ok {
			return false
		}
		p.ops[i] = devOp{Kind: dkQKVGemm, Src: cat(cat(a.Src, b.Src), c.Src), In: a.In,
			Out: []devVal{a.Out[0], b.Out[0], c.Out[0]}, Role: jlm.RoleAttnQ, QS: qs}
		p.ops = append(p.ops[:i+1], p.ops[i+3:]...)
		return true
	}
	return false
}

// forms is the op-span forms a fused program runs, under the encoder's names
// (BlockForms): q/k/v, gate/up and down.
func (p *devProgram) forms() (qkv qkvForm, gu ffnGateUpForm, down ffnDownForm) {
	qkv, gu, down = qkvThree, gateUpTwo, downFormPlain
	for _, o := range p.ops {
		switch o.Kind {
		case dkQKV, dkQKVIn:
			qkv = qkvGrouped
		case dkQKVSep:
			qkv = qkvSeparate
		case dkQKVGemm:
			qkv = qkvGEMM
		case dkGate2In:
			gu = gateUpGate2In
		case dkGate2:
			gu = gateUpGate2
		case dkFFNIn:
			gu = gateUpIn
		case dkDownAct:
			down = downFormAct
		case dkResIn:
			if o.Role == jlm.RoleFFNDown {
				down = downFormIn
			}
		}
	}
	return qkv, gu, down
}
