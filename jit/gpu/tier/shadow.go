//go:build jitllmtest

package tier

import (
	"fmt"
	"maps"
	"math"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The shadow (ShadowForTest; a jitllmtest build's alone, see shadow_off.go)
// asks the composer for
// slices of each block the tier has just run, as the call sites will once
// the composer serves them: it runs the composer's kernels on the slice's
// inputs, read from the tier's buffers, into its own buffers and compares
// them with what the tier's own kernels wrote. What the composer cannot
// form, and what does not match, is the work list before the switch. The
// composer may split or fuse a slice its own way; only the slice's float
// edges are compared.

// ShadowLog is what a shadowed tier compared, refused and could not form.
type ShadowLog struct {
	// Both the composer's slice and the tier's quantize their activations,
	// and an f32 reduction-order difference upstream flips a code
	// downstream, so a fixed bound on their distance is not one: a compare
	// measures the slice's int8 band, the tier's distance from the same
	// slice with float activations (no int8 form, the weights decoded),
	// formed by the composer from the same ops, and holds the composer's
	// distance from the tier to Ratio of that band, plus Floor.
	Ratio, Floor float64
	// MaxBand caps the band: the tier's distance from the float slice.
	MaxBand float64
	// Swap binds these two roles' weights, each with its own format, the
	// wrong way round in the composer's slice: a slice that forms and runs
	// and computes the wrong thing, a violation the compare must see.
	Swap [2]jlm.Role
	// WrongHistory has the composer's slice state the keys' pool laid out by
	// position rather than by dimension: a slice that forms and runs and
	// addresses the wrong elements.
	WrongHistory bool
	// Perturb has the composer's slice read this role's vector at twice its
	// values, and PerturbTable this step's table: violations for a slice
	// with one weight or none.
	Perturb      jlm.Role
	PerturbTable string
	// Compared counts compared slices by name.
	Compared map[string]int
	// Worst is the largest NMSE between a composer's slice and the tier's;
	// WorstTier and WorstComposer each one's largest from the float slice;
	// WorstRatio the largest composer-to-tier distance as a fraction of its
	// slice's band.
	Worst, WorstTier, WorstComposer, WorstRatio float64
	// PoolKeys is the largest NMSE between the keys the composer appended
	// and the tier's at the same pages.
	PoolKeys float64
	// FloatBound is the largest NMSE between the tier and the float slice
	// where the tier runs a block's contractions on binary16 operands
	// (voltaOnly): the composer has no matrix-unit form, so only its float
	// slice is formed there, and the block is counted in Gaps.
	FloatBound float64
	// Refused counts slices not shadowed, by reason; Unformed slices the
	// composer could not form, by its error; Gaps slices compared at the
	// tier's precision because the composer lacks the tier's form; Failed
	// compares over the bound.
	Refused  map[string]int
	Unformed map[string]int
	Gaps     map[string]int
	// GapSlices counts the slices of Gaps by slice name.
	GapSlices map[string]int
	Failed    []string
	// FailedSlices counts, by slice name, the compared slices with at least
	// one edge over the bound: a violation must fail every slice it touches.
	// Errors are runs, reads and binds that went wrong, never a compare.
	FailedSlices map[string]int
	Errors       []string
}

func count(m *map[string]int, k string) {
	if *m == nil {
		*m = map[string]int{}
	}
	(*m)[k]++
}

// shadowDefault is the log a tier opened after ShadowForTest shadows into.
var shadowDefault *ShadowLog

// ShadowForTest makes every device opened after it shadow its blocks into
// log, and returns the function that stops it. Not safe against a concurrent
// open.
func ShadowForTest(log *ShadowLog) (restore func()) {
	old := shadowDefault
	shadowDefault = log
	return func() { shadowDefault = old }
}

// openShadow is a new device's shadow, nil unless ShadowForTest set a log.
func openShadow() *shadowTier {
	if shadowDefault == nil {
		return nil
	}
	return newShadow(shadowDefault)
}

// shadowTier is one tier's shadow: the slices the composer formed, by
// shape, and the inputs captured from the block running now.
type shadowTier struct {
	log   *ShadowLog
	forms map[jit.Key]*shadowForm
	// float is the composer with no int8 form: the float slice's.
	float *jit.Registry
	// li is the block whose post-attention inputs x and a hold, -1 for none.
	li   int
	x, a []float32
	// rows is the step's rows: a slice's values, inputs and outputs are that
	// many rows of their widths.
	rows int
	// half says the block running now takes its contractions on binary16
	// operands.
	half bool
	// pools are the shadow's own history, a block's keys and values laid
	// out as the tier's pools are, by block: written only by the composer's
	// appends, never the tier's.
	pools map[int]*shadowPools
}

// shadowPools is a block's two histories: the composer's slice's (k, v)
// and its float reference's (rk, rv), each appended to by its own arm.
type shadowPools struct {
	k, v, rk, rv backend.Buf
	n            int // pages
}

func (sp *shadowPools) free() {
	for _, b := range []backend.Buf{sp.k, sp.v, sp.rk, sp.rv} {
		if b != nil {
			b.Free()
		}
	}
}

// step is a step of R rows beginning.
func (sh *shadowTier) step(R int) { sh.rows = R }

// block is a block beginning, half whether it takes its contractions on
// binary16 operands.
func (sh *shadowTier) block(half bool) { sh.half = half }

type sliceKind uint8

const (
	sliceAttnIn sliceKind = iota + 1
	slicePostAttn
	sliceHead
	sliceQuery
	sliceKeyAttend
)

var sliceNames = [...]string{sliceAttnIn: "attention in", slicePostAttn: "post-attention", sliceHead: "head", sliceQuery: "query",
	sliceKeyAttend: "key+attend"}

// shadowSlice is one call site's slice of the block that just ran: its
// generic ops, and the block's weights and vectors under their roles, its
// weights' roles in order.
type shadowSlice struct {
	kind    sliceKind
	graph   *graph.Graph
	formats map[jlm.Role]jlm.Type
	weights map[jlm.Role]*resident
	vectors map[jlm.Role]backend.Buf
	// tables are the step's tables the slice reads, by the composer's name,
	// and their widths in floats.
	tables     map[string]backend.Buf
	tableWidth map[string]int
	// refTables are tables the float reference reads in place of tables':
	// its own history, which its own appends fill.
	refTables map[string]backend.Buf
	// history and step are the slice's history layout and its step's keys
	// and page, for a slice with history ops.
	history    jit.History
	keys, page int
	roles      []jlm.Role
}

// shadowForm is a slice's kernels compiled on the device with the buffers
// the composer's own operands bind: the inputs uploaded, the outputs, the
// values between its ops, its scratch and constants.
type shadowForm struct {
	codes []formedCode
	sargs [][]shadowArg
	bufs  []backend.Buf
	in    []int // the slice's inputs' buffers, in graph.In order
	out   []int // its outputs', in graph.Out order
	width []int // its outputs' widths
}

// shadowArg is what one kernel parameter binds: one of the form's buffers,
// or a region of the block's weight or vector of a role.
type shadowArg struct {
	buf    int
	role   jlm.Role
	region int
	vector bool
	table  string
}

const shadowOwn = -1

func newShadow(log *ShadowLog) *shadowTier {
	return &shadowTier{log: log, forms: map[jit.Key]*shadowForm{}, li: -1}
}

func (sh *shadowTier) close() {
	for k, f := range sh.forms {
		if f != nil {
			f.close()
		}
		delete(sh.forms, k)
	}
	for li, sp := range sh.pools {
		sp.free()
		delete(sh.pools, li)
	}
}

func (f *shadowForm) close() {
	for _, c := range f.codes {
		c.k.Close()
	}
	for _, b := range f.bufs {
		b.Free()
	}
	f.codes, f.bufs = nil, nil
}

// shadowBlock is "" for a block whose attention is the plain one the
// shadow states: an RMSNorm and three projections of its own, attention,
// the output projection. Anything else names itself, not shadowed yet.
func shadowBlock(l *layer, li int, p *nn.LayerPlan) string {
	switch {
	case l.linear || l.ds4 != nil || l.k3 != nil || l.q4e != nil || l.altRouter != nil || l.mla || l.conv != nil:
		return "not an attention block's plain residual"
	case l.clampB != nil:
		return "clamped activations"
	case l.nAttnB != nil || p.LayerNorm:
		return "LayerNorm"
	case l.wq == nil || l.wk == nil || l.wv == nil || l.wo == nil || l.vFromK || l.kvSrc != li:
		return "not three projections of its own"
	}
	return ""
}

// shadowFFN is "" for a block whose post-attention half is the plain slice:
// o, the residual add, an RMSNorm, a gated MLP and its add.
func shadowFFN(l *layer, p *nn.LayerPlan) string {
	switch {
	case l.noFFN:
		return "no FFN"
	case l.router != nil || l.gate == nil || l.up == nil || l.down == nil:
		return "not a gated dense FFN"
	case l.ungated || p.Parallel:
		return "ungated or parallel FFN"
	case l.nPostAttn != nil || l.nPostFFN != nil || l.nSubAttn != nil || l.nSubFFN != nil || l.nResid != nil:
		return "a norm beside the residual"
	case l.nFFNB != nil:
		return "LayerNorm"
	case l.withAttn || l.pleGate != nil || l.outScale != nil:
		return "a branch joining the residual"
	case p.ResidualScale != 0 && p.ResidualScale != 1:
		return "scaled residual"
	}
	return ""
}

var shadowActs = map[kernels.ActKind]graph.Act{
	kernels.ActSiLU:      graph.ActSiLU,
	kernels.ActGELU:      graph.ActGELUTanh,
	kernels.ActGELUErf:   graph.ActGELUErf,
	kernels.ActQuickGELU: graph.ActQuickGELU,
	kernels.ActReLU:      graph.ActReLU,
}

// jlmType is a resident's format as the container names it: a packed type,
// or a float weight the device keeps verbatim, row-major, as the container
// does.
func jlmType(q kernels.Quant) (jlm.Type, bool) {
	switch q {
	case kernels.Float32:
		return jlm.TypeF32, true
	case kernels.Float16:
		return jlm.TypeF16, true
	case kernels.BFloat16:
		return jlm.TypeBF16, true
	}
	for _, t := range quant.PackedTypes {
		if k, ok := kernels.QuantOf(t); ok && k == q {
			return jlm.TypeOf(t)
		}
	}
	return 0, false
}

// setFormats fills the slice's formats from its weights in roles' order.
func (sl *shadowSlice) setFormats(roles ...jlm.Role) error {
	sl.roles = roles
	sl.formats = map[jlm.Role]jlm.Type{}
	for _, r := range roles {
		w := sl.weights[r]
		t, ok := jlmType(w.t)
		if !ok {
			return fmt.Errorf("format %d has no container type", w.t)
		}
		sl.formats[r] = t
	}
	return nil
}

func biasOf(on bool, r jlm.Role) []jlm.Role {
	if on {
		return []jlm.Role{r}
	}
	return nil
}

// attnIn is the attention's input slice: n = RMSNorm(x), then q, k and v,
// each W n plus its bias where it has one, held to the tier's q, k and v
// before anything after the projections reads them.
func (sh *shadowTier) attnIn(g *devTier, lc *launcher, s backend.Session, l *layer, li int, p *nn.LayerPlan, x, q, k, v backend.Buf, qdim, kvDim int, why string) {
	if why == "" {
		why = shadowBlock(l, li, p)
	}
	if why != "" {
		count(&sh.log.Refused, why)
		return
	}
	bias := [3]bool{l.mvq.bias != nil, l.mvk.bias != nil, l.mvv.bias != nil}
	sl := &shadowSlice{
		kind:    sliceAttnIn,
		weights: map[jlm.Role]*resident{jlm.RoleAttnQ: l.wq, jlm.RoleAttnK: l.wk, jlm.RoleAttnV: l.wv},
		vectors: map[jlm.Role]backend.Buf{jlm.RoleAttnNorm: l.nAttn, jlm.RoleAttnQBias: l.mvq.bias,
			jlm.RoleAttnKBias: l.mvk.bias, jlm.RoleAttnVBias: l.mvv.bias},
	}
	if err := sl.setFormats(jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV); err != nil {
		count(&sh.log.Unformed, err.Error())
		return
	}
	b := graph.NewSlice()
	n := b.Norm(b.In("x", p.NEmbd), jlm.RoleAttnNorm, p.RMSEps)
	b.Out(b.MatVec(n, jlm.RoleAttnQ, qdim, biasOf(bias[0], jlm.RoleAttnQBias)...),
		b.MatVec(n, jlm.RoleAttnK, kvDim, biasOf(bias[1], jlm.RoleAttnKBias)...),
		b.MatVec(n, jlm.RoleAttnV, kvDim, biasOf(bias[2], jlm.RoleAttnVBias)...))
	sl.graph = b.G
	in, err := readF32(s, x, sh.rows*p.NEmbd, nil)
	if err != nil {
		count(&sh.log.Refused, "capture: "+err.Error())
		return
	}
	sh.check(g, lc, s, li, sl, [][]float32{in}, []backend.Buf{q, k, v}, nil)
}

// capture reads the block's input row and its attention output, the inputs
// of the post-attention slice, before the tier's kernels overwrite them.
func (sh *shadowTier) capture(g *devTier, s backend.Session, l *layer, li int, p *nn.LayerPlan, x, a backend.Buf, qdim int, why string) {
	sh.li = -1
	if why == "" {
		why = shadowBlock(l, li, p)
	}
	if why == "" {
		why = shadowFFN(l, p)
	}
	if why != "" {
		count(&sh.log.Refused, why)
		return
	}
	var err error
	if sh.x, err = readF32(s, x, sh.rows*p.NEmbd, sh.x); err == nil {
		sh.a, err = readF32(s, a, sh.rows*qdim, sh.a)
	}
	if err != nil {
		count(&sh.log.Refused, "capture: "+err.Error())
		return
	}
	sh.li = li
}

// postAttention is the post-attention half: x2 = x + Wo a, then
// x2 + down(act(gate(n)) * up(n)) where n = RMSNorm(x2), held to the
// tier's: mid its x2, ffn the FFN's output where the tier holds the add for
// the next block's norm (foldAddNorm), want the row it ends with where it
// does not. An edge the tier has not written is nil and not compared.
func (sh *shadowTier) postAttention(g *devTier, lc *launcher, s backend.Session, l *layer, li int, p *nn.LayerPlan, qdim int, mid, ffn, want backend.Buf) {
	if sh.li != li {
		return
	}
	sh.li = -1
	act, ok := shadowActs[l.act]
	if !ok {
		count(&sh.log.Unformed, fmt.Sprintf("activation %d has no generic op", l.act))
		return
	}
	if l.mvg.bias != nil {
		count(&sh.log.Unformed, "a gate bias has no role")
		return
	}
	nffn := l.gate.nrows
	bias := [4]bool{l.mvo.bias != nil, false, l.mvu.bias != nil, l.mvd.bias != nil}
	sl := &shadowSlice{
		kind:    slicePostAttn,
		weights: map[jlm.Role]*resident{jlm.RoleAttnOut: l.wo, jlm.RoleFFNGate: l.gate, jlm.RoleFFNUp: l.up, jlm.RoleFFNDown: l.down},
		vectors: map[jlm.Role]backend.Buf{jlm.RoleFFNNorm: l.nFFN, jlm.RoleAttnOutBias: l.mvo.bias,
			jlm.RoleFFNUpBias: l.mvu.bias, jlm.RoleFFNDownBias: l.mvd.bias},
	}
	if err := sl.setFormats(jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown); err != nil {
		count(&sh.log.Unformed, err.Error())
		return
	}
	b := graph.NewSlice()
	x := b.In("x", p.NEmbd)
	a := b.In("attn", qdim)
	x2 := b.Add(x, b.MatVec(a, jlm.RoleAttnOut, p.NEmbd, biasOf(bias[0], jlm.RoleAttnOutBias)...))
	n := b.Norm(x2, jlm.RoleFFNNorm, p.RMSEps)
	gt := b.MatVec(n, jlm.RoleFFNGate, nffn)
	up := b.MatVec(n, jlm.RoleFFNUp, nffn, biasOf(bias[2], jlm.RoleFFNUpBias)...)
	down := b.MatVec(b.ActMul(act, gt, up), jlm.RoleFFNDown, p.NEmbd, biasOf(bias[3], jlm.RoleFFNDownBias)...)
	b.Out(x2, down, b.Add(x2, down))
	sl.graph = b.G
	// Where the tier fuses the FFN's add, its FFN output is x - x2, recovered
	// here to half an ulp of x: the FFN is held to its own magnitude, not
	// the residual's.
	var hosts [][]float32
	if want != nil {
		tx, err := readF32(s, want, sh.rows*p.NEmbd, nil)
		if err == nil {
			var tx2 []float32
			if tx2, err = readF32(s, mid, sh.rows*p.NEmbd, nil); err == nil {
				for i := range tx {
					tx[i] -= tx2[i]
				}
				hosts = [][]float32{nil, tx, nil}
			}
		}
		if err != nil {
			sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, post-attention: %v", li, err))
			return
		}
	}
	sh.check(g, lc, s, li, sl, [][]float32{sh.x, sh.a}, []backend.Buf{mid, ffn, want}, hosts)
}

// query is the query's way to the scores: its per-head norm (or one over
// the row, where the plan's norm is wide) where the block has one, then the
// rotary by the step's table, held to the tier's rotated q.
func (sh *shadowTier) query(g *devTier, lc *launcher, s backend.Session, l *layer, li int, p *nn.LayerPlan, q, norm, cs, qr backend.Buf, qdim int, why string) {
	switch {
	case why != "":
	case p.QKL2Norm:
		why = "an L2 query norm"
	case p.QKNormPost:
		why = "a query norm after the rotary"
	case p.RopeSplit:
		why = "a rotary turned by several tables"
	}
	if why != "" {
		count(&sh.log.Refused, "query: "+why)
		return
	}
	sl := &shadowSlice{kind: sliceQuery, vectors: map[jlm.Role]backend.Buf{jlm.RoleAttnQNorm: norm},
		tables: map[string]backend.Buf{"rope.cs": cs}, tableWidth: map[string]int{"rope.cs": sh.rows * p.NRot}}
	layout := graph.RopeNorm
	if p.RopeNeox {
		layout = graph.RopeNeox
	}
	b := graph.NewSlice()
	x := b.In("q", qdim)
	switch {
	case norm != nil && p.QKNorm && p.QKNormWide:
		x = b.Norm(x, jlm.RoleAttnQNorm, p.RMSEps)
	case norm != nil:
		x = b.HeadNorm(x, jlm.RoleAttnQNorm, p.HeadDim, p.RMSEps)
	}
	b.Out(b.Rope(x, p.HeadDim, p.NRot, p.RopeBase, layout))
	sl.graph = b.G
	in, err := readF32(s, q, sh.rows*qdim, nil)
	if err != nil {
		count(&sh.log.Refused, "capture: "+err.Error())
		return
	}
	sh.check(g, lc, s, li, sl, [][]float32{in}, []backend.Buf{qr}, nil)
}

// poolsFor is the shadow's pools for block li, as many pages as the tier's
// pool l holds, its earlier pages kept where the tier's grew.
func (sh *shadowTier) poolsFor(g *devTier, s backend.Session, li int, l *kvLayerPool, page int) (*shadowPools, error) {
	if sh.pools == nil {
		sh.pools = map[int]*shadowPools{}
	}
	sp := sh.pools[li]
	if sp != nil && sp.n >= l.n {
		return sp, nil
	}
	kw, vw := l.geom.kWords(page), l.geom.vWords(page)
	np := &shadowPools{n: l.n}
	var old [4]backend.Buf
	if sp != nil {
		old = [4]backend.Buf{sp.k, sp.v, sp.rk, sp.rv}
	}
	for i, dst := range []*backend.Buf{&np.k, &np.v, &np.rk, &np.rv} {
		words := kw
		if i%2 == 1 {
			words = vw
		}
		b, err := g.dev.Alloc(max(l.n*words*4, 4))
		if err != nil {
			np.free()
			return nil, err
		}
		*dst = b
		w := make([]float32, l.n*words)
		if sp != nil {
			if err := s.Read(old[i], f32b(w[:sp.n*words])); err != nil {
				np.free()
				return nil, err
			}
		}
		if err := s.Write(b, f32b(w)); err != nil {
			np.free()
			return nil, err
		}
	}
	if sp != nil {
		sp.free()
	}
	sh.pools[li] = np
	return np, nil
}

// keyAttend is the attention core: k's per-head norm where the block has
// one and its rotary where it turns, the append of k and v to the history,
// and q's attention over it -- held to the tier's attention output, and the
// keys and values it appended held to the words the tier's own writers put
// in its pools at the same pages, which is the layout the request states
// (jit.History): keys by dimension, values binary16 where the tier keeps
// them so.
func (sh *shadowTier) keyAttend(g *devTier, lc *launcher, s backend.Session, l *layer, li int, p *nn.LayerPlan, pl *kvLayerPool,
	k, v, q, kNorm, cs, sinks, rows, qOff, out backend.Buf, page, keys int, roped bool, why string) {
	qdim, kvDim := p.NHead*p.HeadDim, p.NKVHead*p.HeadDim
	switch {
	case why != "":
	case pl == nil:
		why = "no pages"
	case l.vFromK || p.VNorm:
		why = "v made from k"
	case p.QKL2Norm || p.QKNormPost || p.RopeSplit:
		why = "a key norm or rotary the slice does not state"
	case pl.geom.mla:
		why = "a latent history"
	}
	if why != "" {
		count(&sh.log.Refused, "key+attend: "+why)
		return
	}
	sp, err := sh.poolsFor(g, s, li, pl, page)
	if err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, key+attend: %v", li, err))
		return
	}
	layout := graph.RopeNorm
	if p.RopeNeox {
		layout = graph.RopeNeox
	}
	sl := &shadowSlice{kind: sliceKeyAttend,
		vectors:    map[jlm.Role]backend.Buf{jlm.RoleAttnKNorm: kNorm, jlm.RoleAttnSinks: sinks},
		tables:     map[string]backend.Buf{"rope.cs": cs, "kv.k": sp.k, "kv.v": sp.v, "pages": pl.tab, "rows": rows, "q.off": qOff},
		tableWidth: map[string]int{"rope.cs": sh.rows * p.NRot},
		refTables:  map[string]backend.Buf{"kv.k": sp.rk, "kv.v": sp.rv},
		history:    jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: pl.geom.f16}},
		keys:       keys, page: page,
	}
	b := graph.NewSlice()
	kx := b.In("k", kvDim)
	vx := b.In("v", kvDim)
	qx := b.In("q", qdim)
	if kNorm != nil {
		if p.QKNorm && p.QKNormWide {
			kx = b.Norm(kx, jlm.RoleAttnKNorm, p.RMSEps)
		} else {
			kx = b.HeadNorm(kx, jlm.RoleAttnKNorm, p.HeadDim, p.RMSEps)
		}
	}
	if roped {
		kx = b.Rope(kx, p.HeadDim, p.NRot, p.RopeBase, layout)
	}
	n := b.KVAppend(kx, vx, 0)
	a := graph.Attrs{NHead: p.NHead, NKVHead: p.NKVHead, HeadDim: p.HeadDim, Scale: float64(g.scoreScale(p)), Window: p.Window(li),
		Chunked: p.SWAChunked, Softcap: float64(p.AttnSoftcap)}
	var sk []jlm.Role
	if sinks != nil {
		sk = append(sk, jlm.RoleAttnSinks)
	}
	b.Out(b.Attend(qx, n, a, sk...))
	sl.graph = b.G
	var ins [][]float32
	for _, x := range [][2]any{{k, kvDim}, {v, kvDim}, {q, qdim}} {
		f, err := readF32(s, x[0].(backend.Buf), sh.rows*x[1].(int), nil)
		if err != nil {
			count(&sh.log.Refused, "capture: "+err.Error())
			return
		}
		ins = append(ins, f)
	}
	sh.check(g, lc, s, li, sl, ins, []backend.Buf{out}, nil)
	sh.comparePools(s, li, pl, sp, rows, page, kvDim)
}

// comparePools holds the keys and values the composer's append wrote at each
// of the step's rows to the words the tier wrote at the same page and
// position: they must be the same words, or the composer addresses another
// layout than the tier's.
func (sh *shadowTier) comparePools(s backend.Session, li int, pl *kvLayerPool, sp *shadowPools, rows backend.Buf, page, kvDim int) {
	desc := make([]uint32, sh.rows*4)
	if err := s.Read(rows, u32b(desc)); err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, pools: %v", li, err))
		return
	}
	tabTop := 0
	for r := range sh.rows {
		tabTop = max(tabTop, int(desc[r*4])+int(desc[r*4+3])/page+1)
	}
	tab := make([]uint32, tabTop)
	if err := s.Read(pl.tab, u32b(tab)); err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, pools: %v", li, err))
		return
	}
	kw, vw := pl.geom.kWords(page), pl.geom.vWords(page)
	type at struct{ pid, off int }
	var ats []at
	top := 0
	for r := range sh.rows {
		pos := int(desc[r*4+3])
		pid := int(tab[int(desc[r*4])+pos/page])
		ats = append(ats, at{pid, pos % page})
		top = max(top, pid+1)
	}
	read := func(b backend.Buf, words int) ([]uint32, error) {
		w := make([]uint32, top*words)
		return w, s.Read(b, u32b(w))
	}
	tk, err := read(pl.k, kw)
	var ck, tv, cv []uint32
	if err == nil {
		ck, err = read(sp.k, kw)
	}
	if err == nil {
		tv, err = read(pl.v, vw)
	}
	if err == nil {
		cv, err = read(sp.v, vw)
	}
	if err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, pools: %v", li, err))
		return
	}
	var kt, kc []float32
	diffV := 0
	for _, a := range ats {
		for d := range kvDim {
			// The keys by dimension: [kvRow][page] within a page.
			i := a.pid*kw + d*page + a.off
			kt, kc = append(kt, math.Float32frombits(tk[i])), append(kc, math.Float32frombits(ck[i]))
		}
		vrow := vw / page
		for j := range vrow {
			i := a.pid*vw + a.off*vrow + j
			if tv[i] != cv[i] {
				diffV++
			}
		}
	}
	e := nmse(kc, kt)
	sh.log.PoolKeys = max(sh.log.PoolKeys, e)
	if !(e <= sh.log.Floor) || diffV > 0 {
		count(&sh.log.FailedSlices, "pools")
		sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, pools: the appended keys are %.3g from the tier's, %d value words differ", li, e, diffV))
	}
	count(&sh.log.Compared, "pools")
}

// head is the output head: logits = W RMSNorm(x), held to the tier's logits
// before any softcap.
func (sh *shadowTier) head(g *devTier, lc *launcher, s backend.Session, p *nn.LayerPlan, x, norm backend.Buf, w *resident, logits backend.Buf, why string) {
	switch {
	case why != "":
	case norm == nil:
		why = "a head with no norm"
	case sh.rows != 1:
		why = "a prompt chunk's head"
	}
	if why != "" {
		count(&sh.log.Refused, "head: "+why)
		return
	}
	sl := &shadowSlice{
		kind:    sliceHead,
		weights: map[jlm.Role]*resident{jlm.RoleOutput: w},
		vectors: map[jlm.Role]backend.Buf{jlm.RoleOutputNorm: norm},
	}
	if err := sl.setFormats(jlm.RoleOutput); err != nil {
		count(&sh.log.Unformed, err.Error())
		return
	}
	b := graph.NewSlice()
	b.Out(b.MatVec(b.Norm(b.In("x", p.NEmbd), jlm.RoleOutputNorm, p.RMSEps), jlm.RoleOutput, w.nrows))
	sl.graph = b.G
	in, err := readF32(s, x, sh.rows*p.NEmbd, nil)
	if err != nil {
		count(&sh.log.Refused, "capture: "+err.Error())
		return
	}
	sh.check(g, lc, s, -1, sl, [][]float32{in}, []backend.Buf{logits}, nil)
}

// check forms sl with and without the int8 form, runs both on ins, and holds
// each output to the tier's: its buffer in wants, or where hosts has the
// output's rows, those.
func (sh *shadowTier) check(g *devTier, lc *launcher, s backend.Session, li int, sl *shadowSlice, ins [][]float32, wants []backend.Buf, hosts [][]float32) {
	name := sliceNames[sl.kind]
	if sh.half && sl.kind != sliceQuery {
		sh.checkHalf(g, lc, s, li, sl, ins, wants, hosts)
		return
	}
	v, done, err := sh.violate(g, s, sl)
	if err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, %s: %v", li, name, err))
		return
	}
	defer done()
	ref := sl
	if sl.refTables != nil {
		r := *sl
		r.tables = maps.Clone(sl.tables)
		maps.Copy(r.tables, sl.refTables)
		ref = &r
	}
	arms := [2]*shadowSlice{v, ref}
	var outs [2][][]float32
	failed := false
	for i, float := range [2]bool{false, true} {
		f, err := sh.form(g, arms[i], float)
		if err != nil {
			count(&sh.log.Unformed, name+": "+err.Error())
			return
		}
		if outs[i], err = sh.run(lc, s, f, arms[i], ins); err != nil {
			sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, %s: %v", li, name, err))
			return
		}
	}
	for j, want := range wants {
		tier, err := tierRows(s, want, hosts, j, len(outs[0][j]))
		if err != nil {
			sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, %s: %v", li, name, err))
			return
		}
		if tier == nil {
			continue
		}
		// Every distance is over the float slice's norm: over the tier's own,
		// a tier row far from both would read near one against the composer
		// and pass inside a band its own distance made huge.
		got, ref := outs[0][j], outs[1][j]
		e, et, ec := nmseOver(got, tier, ref), nmse(tier, ref), nmse(got, ref)
		sh.log.Worst = max(sh.log.Worst, e)
		sh.log.WorstTier, sh.log.WorstComposer = max(sh.log.WorstTier, et), max(sh.log.WorstComposer, ec)
		// A slice with no int8 form has no band above the floor, and a
		// ratio to it measures rounding.
		if et > sh.log.Floor {
			sh.log.WorstRatio = max(sh.log.WorstRatio, e/et)
		}
		if !(e <= sh.log.Ratio*et+sh.log.Floor) {
			failed = true
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, %s, output %d: the composer's slice is %.3g from the tier's, whose int8 band here is %.3g (the composer's own %.3g)",
				li, name, j, e, et, ec))
		}
		// The composer's float slice is the band's reference; it is gated
		// against the oracle on its own (TestServedMatVecIsComposed's float
		// arm), and the tier's int8 arithmetic may not be further from it
		// than MaxBand.
		if !(et <= sh.log.MaxBand) {
			failed = true
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, %s, output %d: the tier is %.3g from the float slice, over the largest band %.3g",
				li, name, j, et, sh.log.MaxBand))
		}
	}
	count(&sh.log.Compared, name)
	if failed {
		count(&sh.log.FailedSlices, name)
	}
}

// tierRows is the tier's output j: hosts[j] where it holds the rows, else
// want read back; nil where the tier wrote neither.
func tierRows(s backend.Session, want backend.Buf, hosts [][]float32, j, n int) ([]float32, error) {
	if j < len(hosts) && hosts[j] != nil {
		return hosts[j], nil
	}
	if want == nil {
		return nil, nil
	}
	return readF32(s, want, n, nil)
}

// vectorWidth is the width the graph binds role's vector at.
func vectorWidth(gr *graph.Graph, role jlm.Role) int {
	for _, o := range gr.Ops {
		for _, w := range o.Weight {
			if w.Role == role {
				return w.K
			}
		}
	}
	return 0
}

// violate is sl as the log's violation has the composer's slice bind it,
// and what frees what that took; sl itself where the log names none.
func (sh *shadowTier) violate(g *devTier, s backend.Session, sl *shadowSlice) (*shadowSlice, func(), error) {
	out, frees := sl, []backend.Buf(nil)
	done := func() {
		for _, b := range frees {
			b.Free()
		}
	}
	if a, b := sh.log.Swap[0], sh.log.Swap[1]; sl.weights[a] != nil && sl.weights[b] != nil {
		v := *out
		v.weights = maps.Clone(out.weights)
		v.weights[a], v.weights[b] = out.weights[b], out.weights[a]
		if err := v.setFormats(sl.roles...); err != nil {
			return nil, done, err
		}
		out = &v
	}
	if r := sh.log.Perturb; r != 0 && sl.vectors[r] != nil {
		buf, err := doubled(g, s, sl.vectors[r], vectorWidth(sl.graph, r))
		if err != nil {
			return nil, done, err
		}
		frees = append(frees, buf)
		v := *out
		v.vectors = maps.Clone(out.vectors)
		v.vectors[r] = buf
		out = &v
	}
	if sh.log.WrongHistory && sl.history.Keys.ByDim {
		v := *out
		v.history.Keys.ByDim = false
		out = &v
	}
	if n := sh.log.PerturbTable; n != "" && sl.tables[n] != nil {
		buf, err := doubled(g, s, sl.tables[n], sl.tableWidth[n])
		if err != nil {
			return nil, done, err
		}
		frees = append(frees, buf)
		v := *out
		v.tables = maps.Clone(out.tables)
		v.tables[n] = buf
		out = &v
	}
	return out, done, nil
}

// checkHalf holds the tier's binary16 contractions to the float slice at
// FloatBound, and counts the slice as a gap: the composer forms no
// matrix-unit contraction, so its own form is not what the tier runs.
func (sh *shadowTier) checkHalf(g *devTier, lc *launcher, s backend.Session, li int, sl *shadowSlice, ins [][]float32, wants []backend.Buf, hosts [][]float32) {
	name := sliceNames[sl.kind]
	v, done, err := sh.violate(g, s, sl)
	if err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, %s: %v", li, name, err))
		return
	}
	defer done()
	f, err := sh.form(g, v, true)
	if err != nil {
		count(&sh.log.Unformed, name+": "+err.Error())
		return
	}
	ref, err := sh.run(lc, s, f, v, ins)
	failed := false
	if err != nil {
		sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, %s: %v", li, name, err))
		return
	}
	for j, want := range wants {
		tier, err := tierRows(s, want, hosts, j, len(ref[j]))
		if tier == nil && err == nil {
			continue
		}
		if err != nil {
			sh.log.Errors = append(sh.log.Errors, fmt.Sprintf("block %d, %s: %v", li, name, err))
			return
		}
		et := nmse(tier, ref[j])
		sh.log.WorstTier = max(sh.log.WorstTier, et)
		if !(et <= sh.log.FloatBound) {
			failed = true
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, %s, output %d: the tier's binary16 contraction is %.3g from the float slice, bound %.3g", li, name, j, et, sh.log.FloatBound))
		}
	}
	count(&sh.log.Gaps, name+": the tier's contractions are on binary16 operands; the composer has no matrix-unit form")
	count(&sh.log.GapSlices, name)
	if failed {
		count(&sh.log.FailedSlices, name)
	}
}

// doubled is a copy of the n floats of b at twice their values.
func doubled(g *devTier, s backend.Session, b backend.Buf, n int) (backend.Buf, error) {
	f, err := readF32(s, b, n, nil)
	if err != nil {
		return nil, err
	}
	for i := range f {
		f[i] *= 2
	}
	d, err := g.dev.Alloc(len(f) * 4)
	if err != nil {
		return nil, err
	}
	if err := s.Write(d, f32b(f)); err != nil {
		d.Free()
		return nil, err
	}
	return d, nil
}

func readF32(s backend.Session, b backend.Buf, n int, into []float32) ([]float32, error) {
	into = append(into[:0], make([]float32, n)...)
	if err := s.Read(b, f32b(into)); err != nil {
		return nil, err
	}
	return into, nil
}

// nmseOver is |a-b|^2 / |ref|^2, +Inf where any is not finite.
func nmseOver(a, b, ref []float32) float64 {
	var num, den float64
	for i := range ref {
		x, y, r := float64(a[i]), float64(b[i]), float64(ref[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) || math.IsNaN(r) || math.IsInf(r, 0) {
			return math.Inf(1)
		}
		num += (x - y) * (x - y)
		den += r * r
	}
	if den == 0 {
		return num
	}
	return num / den
}

// nmse is |got-want|^2 / |want|^2, +Inf where either is not finite.
func nmse(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			return math.Inf(1)
		}
		num += (g - w) * (g - w)
		den += w * w
	}
	if den == 0 {
		return num
	}
	return num / den
}

// form is sl's kernels on g's device, with the int8 form or without it,
// compiled once for the composer's Key.
func (sh *shadowTier) form(g *devTier, sl *shadowSlice, float bool) (*shadowForm, error) {
	if g.composer == nil {
		g.composer = jit.Default()
	}
	r := g.composer
	if float {
		if sh.float == nil {
			sh.float = g.composer.WithoutInt8(jit.DeviceGPU)
		}
		r = sh.float
	}
	gr := sl.graph
	key, codes, err := r.Form(jit.Request{
		Graph:   gr,
		Formats: sl.formats,
		Step:    jit.Step{Rows: sh.rows, Page: max(sl.page, 64), Keys: max(sl.keys, 1), ActWindow: g.actWin},
		Device:  jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: g.pickLanes()},
		History: sl.history,
	})
	if err != nil {
		return nil, err
	}
	if f, ok := sh.forms[key]; ok {
		if f == nil {
			return nil, fmt.Errorf("the slice of key %#x did not compile earlier", key)
		}
		return f, nil
	}
	sh.forms[key] = nil
	f := &shadowForm{}
	vals := map[graph.ValueID]int{}
	own := map[jit.Operand]int{}
	alloc := func(n int) (int, error) {
		b, err := g.dev.Alloc(max(n, 4))
		if err != nil {
			return 0, err
		}
		f.bufs = append(f.bufs, b)
		return len(f.bufs) - 1, nil
	}
	value := func(v graph.ValueID) (int, error) {
		if i, ok := vals[v]; ok {
			return i, nil
		}
		i, err := alloc(4 * sh.rows * gr.Values[v].Width)
		vals[v] = i
		return i, err
	}
	for _, v := range gr.In {
		i, err := value(v)
		if err != nil {
			f.close()
			return nil, err
		}
		f.in = append(f.in, i)
	}
	for _, c := range codes {
		if c.GPU == nil {
			f.close()
			return nil, fmt.Errorf("%s has no GPU code", c.Name)
		}
		kern, err := g.dev.Compile(c.GPU)
		if err != nil {
			f.close()
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		f.codes = append(f.codes, formedCode{k: kern, groups: c.Groups, width: c.GPU.Group[0]})
		var args []shadowArg
		for _, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue:
				i, err := value(o.Value)
				if err != nil {
					f.close()
					return nil, err
				}
				args = append(args, shadowArg{buf: i})
			case o.Kind == jit.OperandWeight && o.Region >= 0 && o.Region < 3:
				args = append(args, shadowArg{buf: shadowOwn, role: o.Role, region: o.Region})
			case o.Kind == jit.OperandVector:
				args = append(args, shadowArg{buf: shadowOwn, role: o.Role, vector: true})
			case o.Kind == jit.OperandTable && sl.tables[o.Name] != nil:
				args = append(args, shadowArg{buf: shadowOwn, table: o.Name})
			case bd.Words != nil || o.Kind == jit.OperandScratch:
				i, ok := own[o]
				if !ok {
					if i, err = alloc(int(max(bd.Bytes, 4*int64(len(bd.Words))))); err != nil {
						f.close()
						return nil, err
					}
					if bd.Words != nil {
						if err := f.bufs[i].Write(u32b(bd.Words)); err != nil {
							f.close()
							return nil, err
						}
					}
					own[o] = i
				}
				args = append(args, shadowArg{buf: i})
			default:
				f.close()
				return nil, fmt.Errorf("%s binds %+v, which the slice's call site does not provide", c.Name, o)
			}
		}
		f.sargs = append(f.sargs, args)
		f.codes[len(f.codes)-1].bufs = make([]backend.Buf, len(args))
	}
	for _, v := range gr.Out {
		i, err := value(v)
		if err != nil {
			f.close()
			return nil, err
		}
		f.out = append(f.out, i)
		f.width = append(f.width, sh.rows*gr.Values[v].Width)
	}
	sh.forms[key] = f
	return f, nil
}

// run uploads the slice's inputs, launches its kernels on the block's
// weights and reads its outputs back.
func (sh *shadowTier) run(lc *launcher, s backend.Session, f *shadowForm, sl *shadowSlice, ins [][]float32) ([][]float32, error) {
	for i, v := range ins {
		if err := s.Write(f.bufs[f.in[i]], f32b(v)); err != nil {
			return nil, err
		}
	}
	weights := sl.weights
	for i := range f.codes {
		fc := &f.codes[i]
		for j, a := range f.sargs[i] {
			fc.bufs[j] = nil
			switch {
			case a.buf >= 0:
				fc.bufs[j] = f.bufs[a.buf]
			case a.vector:
				fc.bufs[j] = sl.vectors[a.role]
			case a.table != "":
				fc.bufs[j] = sl.tables[a.table]
			default:
				if r := weights[a.role]; r != nil {
					fc.bufs[j] = [3]backend.Buf{r.qs, r.d, r.sc}[a.region]
				}
			}
			if fc.bufs[j] == nil {
				return nil, fmt.Errorf("the composer's kernel %d reads %v, which the block does not hold", i, a.role)
			}
		}
		if err := lc.launch(fc.k, fc.groups, fc.width, fc.bufs...); err != nil {
			return nil, err
		}
	}
	outs := make([][]float32, len(f.out))
	for i, b := range f.out {
		var err error
		if outs[i], err = readF32(s, f.bufs[b], f.width[i], nil); err != nil {
			return nil, err
		}
	}
	return outs, nil
}
