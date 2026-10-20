package tier

import (
	"slices"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/backend"

	"github.com/jitllm/jitllm/engine/nn"
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
	dkGatherHeads                                // a chunk's rows reordered head-major (headMajor)
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
	case dkRecurrent:
		return "recurrent"
	case dkCopy:
		return "copy"
	case dkKVCopy:
		return "kv_copy"
	case dkAct:
		return "act"
	case dkQTemp:
		return "q_temp"
	case dkSplitQG:
		return "split_qg"
	case dkPLESlice:
		return "ple_slice"
	case dkRopeKRow:
		return "rope_k_row"
	case dkStreamRoute:
		return "stream_route"
	case dkStreamPredict:
		return "stream_predict"
	case dkStreamCorrect:
		return "stream_correct"
	case dkStreamScale:
		return "stream_scale"
	case dkStreamFinish:
		return "stream_finish"
	case dkGaussTopK:
		return "gauss_topk"
	case dkStreamNorm:
		return "stream_norm"
	case dkStreamGateMix:
		return "stream_gate_mix"
	case dkPlaceWeights:
		return "place_weights"
	case dkStreamPlace:
		return "stream_place"
	case dkIdxNorm:
		return "idx_norm"
	case dkIdxRope:
		return "idx_rope"
	case dkIdxAssemble:
		return "idx_assemble"
	case dkIdxScore:
		return "idx_score"
	case dkSelectBlocks:
		return "select_blocks"
	case dkGatherKV:
		return "gather_kv"
	case dkAttendGathered:
		return "attend_gathered"
	case dkStreamDot:
		return "stream_dot"
	case dkGatedVal:
		return "gated_val"
	case dkNGramConv:
		return "ngram_conv"
	case dkNGramAdd:
		return "ngram_add"
	case dkStreamScore:
		return "stream_score"
	case dkStreamWeighted:
		return "stream_weighted"
	case dkIdxWrite:
		return "idx_write"
	case dkIdxAttendLatent:
		return "idx_attend_latent"
	case dkRowBias:
		return "row_bias"
	case dkStreamFlatten:
		return "stream_flatten"
	case dkFlatNorm:
		return "flat_norm"
	case dkMixLogits:
		return "mix_logits"
	case dkSinkhorn:
		return "sinkhorn"
	case dkStreamCollect:
		return "stream_collect"
	case dkRowRMS:
		return "row_r_m_s"
	case dkTurn:
		return "turn"
	case dkCacheWrite:
		return "cache_write"
	case dkPoolEntries:
		return "pool_entries"
	case dkEntryWrite:
		return "entry_write"
	case dkEntrySelect:
		return "entry_select"
	case dkGatherEntries:
		return "gather_entries"
	case dkGroupedMatVec:
		return "grouped_mat_vec"
	case dkClampAt:
		return "clamp_at"
	case dkDWConvRows:
		return "dw_conv_rows"
	case dkAffineRows:
		return "affine_rows"
	case dkRelAttend:
		return "rel_attend"
	case dkScale:
		return "scale"
	case dkClamp:
		return "clamp"
	case dkRouterLogits:
		return "router_logits"
	case dkRouterBias:
		return "router_bias"
	case dkGroupMask:
		return "group_mask"
	case dkTopK:
		return "topk"
	case dkRouteWeights:
		return "route_weights"
	case dkExpertMatVec:
		return "expert_matvec"
	case dkExpertBias:
		return "expert_bias"
	case dkCombine:
		return "combine"
	case dkGateLogit:
		return "gate_logit"
	case dkGatedAdd:
		return "gated_add"
	case dkExpertFetch:
		return "expert_fetch"
	case dkGroupByExpert:
		return "group_by_expert"
	case dkSlice:
		return "slice"
	case dkView:
		return "view"
	case dkRopeTail:
		return "rope_tail"
	case dkLatentWrite:
		return "latent_write"
	case dkConcatHeads:
		return "concat_heads"
	case dkLatentAttend:
		return "latent_attend"
	case dkExpertMatVecE:
		return "expert_matvec_e"
	case dkConv1d:
		return "conv1d"
	case dkScan:
		return "scan"
	case dkGatedNorm:
		return "gated_norm"
	case dkMul:
		return "mul"
	case dkConvBias:
		return "conv_bias"
	case dkSSMNorm:
		return "ssm_norm"
	case dkGatherHeads:
		return "gather_heads"
	case dkMap:
		return "map"
	case dkReduce:
		return "reduce"
	case dkSelectK:
		return "select_k"
	case dkRank:
		return "rank"
	case dkTake:
		return "take"
	case dkTile:
		return "tile"
	case dkConcat:
		return "concat"
	case dkRelayout:
		return "relayout"
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
// to a fixed point before the next. A program's rewrites taken are a bit
// each (rwSet), by position here. Each device holds its own list (devShared).
func newDevRewrites() []devRewrite {
	return []devRewrite{
		{"rope-qkv-paged", rwRopeQKV},
		{"matvec-residual", rwMatVecRes},
		{"matvec-gate", rwMatVecGate},
		{"expert-epilogue", rwExpertEpilogue},
		{"act-quant", rwActQuant},
		{"add-norm-quant", rwAddNormQuant},
		{"add-norm", rwAddNorm},
		{"norm-quant", rwNormQuant},
		{"route", rwRoute},
	}
}

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
	lo, hi := 0, len(rws)
	if R > 1 {
		q.headMajor()
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
	// The whole value, then its slices: a flag added to the program is
	// carried without a line here (a clone that dropped experts ran a
	// mixture's rows on one row's kernels).
	q := new(devProgram)
	*q = *p
	q.vals = append([]devType(nil), p.vals...)
	q.slots = append([]devSlot(nil), p.slots...)
	q.names = append([]string(nil), p.names...)
	q.pad = append([]int(nil), p.pad...)
	q.ops = make([]devOp, 0, len(p.ops))
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

// ropeQKVFuses: every head turned whole on every layer, k and v the whole
// cached row, nothing between the rotary and the history.
func ropeQKVFuses(p *nn.LayerPlan) bool {
	return p.NRot > 0 && !partialRotary(p) && !p.NoPosEnc && !p.RopeSplit && !p.QKNormPost &&
		p.NKVHead*p.HeadDim == p.KVRow()
}

// rwRopeQKV: q's rotary and the history write, one paged launch.
func rwRopeQKV(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if bs.pkv == nil || g.NoFuse || !ropeQKVFuses(&bs.p) {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkRopeQ {
			continue
		}
		for j := i + 1; j < len(p.ops); j++ {
			w := p.ops[j]
			if w.Kind != dkKVWrite || p.slots[w.In[0]] == slotKAll {
				continue // an assembled row (an indexer's key in it) writes alone
			}
			// The fused launch goes where the history write was, after
			// every op its key and value come from; nothing between may
			// read the rotated q it now writes later.
			if p.next(i, o.Out[0]) < j {
				return false
			}
			p.ops[j] = devOp{Kind: dkRopeQKV, Src: cat(o.Src, w.Src), In: []devVal{o.In[0], w.In[0], w.In[1]},
				Out: []devVal{o.Out[0], w.Out[0]}}
			p.ops = append(p.ops[:i], p.ops[i+1:]...)
			return true
		}
	}
	return false
}

// rwMatVecRes: a matvec whose output only a residual add reads, as the
// matvec carrying the add.
func rwMatVecRes(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if residScaled(&bs.p) {
		return false // a scaled residual: the fused kernel carries no scale
	}
	if R != 1 {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkMatVec || (o.Role != jlm.RoleAttnOut && o.Role != jlm.RoleFFNDown) {
			continue
		}
		m := l.mvo
		if o.Role == jlm.RoleFFNDown {
			m = l.mvd
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
		p.replace(i, j, devOp{Kind: dkMatVecRes, Src: cat(o.Src, add.Src), In: []devVal{o.In[0], resid}, Out: add.Out, Role: o.Role, W: o.W})
		return true
	}
	return false
}

// rwMatVecGate: up's matvec and the act_mul over the gate, as up's Gate
// variant writing the product.
func rwMatVecGate(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false
	}
	if l.mvu.split > 1 && !l.mvu.group {
		return false // the gate rides only a matvec that writes its final row
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
		p.replace(i, j, devOp{Kind: dkMatVecGate, Src: cat(o.Src, am.Src), In: []devVal{o.In[0], am.In[0]}, Out: am.Out, Role: o.Role, W: o.W, Act: am.Act})
		return true
	}
	return false
}

// rwActQuant: act_mul and the quantize of its product, one launch.
func rwActQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if ungatedFFN(&bs.p) || bs.p.NFFN == 0 || g.NoOperandFuse {
		return false
	}
	for i, o := range p.ops {
		// The dense FFN's activation: an expert's or a shared expert's has
		// its own width and quantize layout.
		if o.Kind != dkActMul || p.slots[o.Out[0]] != slotAct || i+1 >= len(p.ops) {
			continue
		}
		q := p.ops[i+1]
		if q.Kind == dkQuant && q.In[0] == o.Out[0] {
			p.replace(i, i+1, devOp{Kind: dkActQuant, Src: cat(o.Src, q.Src), In: o.In, Out: []devVal{o.Out[0], q.Out[0]}, Act: o.Act, A: o.A})
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
		if n := p.ops[i+1]; n.Kind == dkNorm && p.slots[n.Out[0]] == slotH && n.In[0] == o.Out[0] {
			return i, i + 1, true
		}
	}
	return 0, 0, false
}

// rwAddNormQuant: the add, the norm of the sum and its quantize, one launch.
func rwAddNormQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if residScaled(&bs.p) {
		return false // a scaled residual: the fused kernel carries no scale
	}
	if lnBlock(&bs.p) {
		return false
	}
	i, j, ok := addNorm(p)
	if !ok {
		return false
	}
	k := rowQuant(p, j)
	if k < 0 {
		return false
	}
	a, n, q := p.ops[i], p.ops[j], p.ops[k]
	fused := devOp{Kind: dkAddNormQuant, Src: cat(cat(a.Src, n.Src), q.Src), In: a.In,
		Out: []devVal{a.Out[0], n.Out[0], q.Out[0]}, Role: n.Role, V: n.V, VB: n.VB}
	p.ops = append(p.ops[:k], p.ops[k+1:]...)
	p.ops[i] = fused
	p.ops = append(p.ops[:i+1], p.ops[j+1:]...)
	return true
}

// rowQuant is the first quantize of op i's row after it, where it may run
// with op i: no op between writes the row or reads or writes what the
// quantize writes; -1 for none.
func rowQuant(p *devProgram, i int) int {
	v := p.ops[i].Out[0]
	for k := i + 1; k < len(p.ops); k++ {
		o := p.ops[k]
		if o.Kind == dkQuant && len(o.In) > 0 && o.In[0] == v {
			q := o.Out[0]
			for m := i + 1; m < k; m++ {
				if slices.Contains(p.ops[m].Out, v) || slices.Contains(p.ops[m].In, q) || slices.Contains(p.ops[m].Out, q) {
					return -1
				}
			}
			return k
		}
	}
	return -1
}

// rwAddNorm: the add and the norm of the sum, one launch.
func rwAddNorm(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if residScaled(&bs.p) {
		return false // a scaled residual: the fused kernel carries no scale
	}
	if lnBlock(&bs.p) {
		return false
	}
	i, j, ok := addNorm(p)
	if !ok {
		return false
	}
	a, n := p.ops[i], p.ops[j]
	p.replace(i, j, devOp{Kind: dkAddNorm, Src: cat(a.Src, n.Src), In: a.In, Out: []devVal{a.Out[0], n.Out[0]}, Role: n.Role, V: n.V, VB: n.VB})
	return true
}

// rwNormQuant: a norm and the quantize of its row, one launch.
func rwNormQuant(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if lnBlock(&bs.p) {
		return false
	}
	for i, o := range p.ops {
		// The block row's norm: a latent's has its own width and kernels.
		if o.Kind != dkNorm || p.slots[o.Out[0]] != slotH {
			continue
		}
		if k := rowQuant(p, i); k >= 0 {
			q := p.ops[k]
			p.ops = append(p.ops[:k], p.ops[k+1:]...)
			p.ops[i] = devOp{Kind: dkNormQuant, Src: cat(o.Src, q.Src), In: o.In, Out: []devVal{o.Out[0], q.Out[0]}, Role: o.Role, V: o.V, VB: o.VB}
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

// resIn rewrites quantize + matvec_res of role into the matvec quantizing
// its own operand, where m's prologue variant exists.
func resIn(p *devProgram, role jlm.Role, ok func(src devVal) bool) bool {
	for i := 1; i < len(p.ops); i++ {
		o, q := p.ops[i], p.ops[i-1]
		if o.Kind == dkMatVecRes && o.Role == role && q.Kind == dkQuant && o.In[0] == q.Out[0] &&
			p.uses(q.Out[0]) == 1 && ok(q.In[0]) {
			p.ops[i-1] = devOp{Kind: dkResIn, Src: cat(q.Src, o.Src), In: []devVal{q.In[0], o.In[1]}, Out: o.Out, Role: role, W: o.W}
			p.ops = append(p.ops[:i], p.ops[i+1:]...)
			return true
		}
	}
	return false
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

// forms is the op-span forms a fused program runs, under the encoder's names
// (BlockForms): q/k/v, and gate/up and down where the block has a dense FFN
// (dense false for a mixture's, which runs whole).
func (p *devProgram) forms() (qkv qkvForm, gu ffnGateUpForm, down ffnDownForm, dense bool) {
	qkv, gu, down, dense = qkvThree, gateUpTwo, downFormPlain, true
	for _, o := range p.ops {
		switch o.Kind {
		case dkExpertMatVec:
			dense = false
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
	return qkv, gu, down, dense
}

// rwExpertEpilogue: an expert bank's matvec and the op that follows it in
// its own epilogue, where prep built the bank's fused kernel (fuseMoE) --
// its bias, and on up the activation act(gate)*up, which reads the gate rows
// and writes the activations. One row: rows run the grouped kernels.
func rwExpertEpilogue(p *devProgram, l *layer, bs *blockScratch, g *devTier, R, nseq int) bool {
	if R != 1 {
		return false
	}
	for i, o := range p.ops {
		if o.Kind != dkExpertMatVec {
			continue
		}
		var k backend.Kernel
		switch o.Role {
		case jlm.RoleExpGateBank:
			k = l.moeGate
		case jlm.RoleExpUpBank:
			k = l.moeUp
		case jlm.RoleExpDownBank:
			k = l.moeDown
		}
		if k == nil || p.uses(o.Out[0]) != 1 {
			continue
		}
		j := p.next(i, o.Out[0])
		if j < 0 {
			continue
		}
		f := devOp{Kind: dkExpertMatVecE, Src: o.Src, In: []devVal{o.In[0], o.In[1]}, Out: o.Out, Role: o.Role, W: o.W, K: k}
		b := p.ops[j]
		if b.Kind == dkExpertBias && b.In[0] == o.Out[0] {
			f.Src, f.Out, f.V = cat(f.Src, b.Src), b.Out, b.V
			if o.Role != jlm.RoleExpUpBank {
				p.replace(i, j, f)
				return true
			}
			if p.uses(b.Out[0]) != 1 {
				continue
			}
			if j = p.next(j, b.Out[0]); j < 0 {
				continue
			}
		}
		if o.Role != jlm.RoleExpUpBank {
			continue // only up's kernel carries the activation without a bias
		}
		a := p.ops[j]
		if a.Kind != dkActMul || len(a.In) != 2 || a.In[1] != f.Out[0] {
			continue
		}
		// up's activation: act(gate)*up into the activations, the gate rows
		// read in the epilogue.
		f.Src, f.In, f.Out = cat(f.Src, a.Src), append(f.In, a.In[0]), a.Out
		p.ops[i] = f
		drop := []int{j}
		if b.Kind == dkExpertBias && b.In[0] == o.Out[0] {
			drop = []int{p.next(i, o.Out[0]), j}
		}
		for n := len(drop) - 1; n >= 0; n-- {
			if d := drop[n]; d > i {
				p.ops = append(p.ops[:d], p.ops[d+1:]...)
			}
		}
		return true
	}
	return false
}

// headMajor lowers, on a chunk of rows, the order a selected matvec over
// heads reads (devOp.Heads): its float rows gathered head-major by an op of
// their own, and its int8 operand quantized from those, so the order is an
// edge of the program rather than a buffer a launch wrote on the side. The
// gathered rows of the program's first such matvec take the pool's first
// slot, of the next its second.
func (p *devProgram) headMajor() {
	pool := slotPool0
	for j := 0; j < len(p.ops); j++ {
		o := p.ops[j]
		if o.Heads <= 0 || len(o.In) != 3 || o.Kind != dkExpertMatVec {
			continue
		}
		src := o.In[2]
		k := -1
		for i := 0; i < j; i++ {
			if q := p.ops[i]; q.Kind == dkQuant && len(q.Out) > 0 && q.Out[0] == o.In[0] && q.In[0] == src {
				k = i
			}
		}
		if k < 0 {
			continue // no quantize of its rows to reorder: the check refuses what reads it
		}
		t := p.vals[src]
		h := p.val(p.names[src]+".heads", devType{Elem: elemF32, Layout: layoutHeads, Width: t.Width}, pool)
		pool++
		p.ops[k].In[0] = h
		p.ops[j].In[2] = h
		g := devOp{Kind: dkGatherHeads, Src: o.Src, In: []devVal{src}, Out: []devVal{h}, Heads: o.Heads}
		p.ops = append(p.ops[:k], append([]devOp{g}, p.ops[k:]...)...)
		j++
	}
}
