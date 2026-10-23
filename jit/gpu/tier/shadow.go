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

// The shadow (ShadowForTest, jitllmtest builds only) asks the composer for
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
	// Swap binds these two roles' weights, each with its own format, the
	// wrong way round in the composer's slice: a slice that forms and runs
	// and computes the wrong thing, a violation the compare must see.
	Swap [2]jlm.Role
	// Compared counts compared slices by name.
	Compared map[string]int
	// Worst is the largest NMSE between a composer's slice and the tier's;
	// WorstTier and WorstComposer each one's largest from the float slice;
	// WorstRatio the largest composer-to-tier distance as a fraction of its
	// slice's band.
	Worst, WorstTier, WorstComposer, WorstRatio float64
	// Refused counts slices not shadowed, by reason; Unformed slices the
	// composer could not form, by its error; Failed compares over the bound.
	Refused  map[string]int
	Unformed map[string]int
	Failed   []string
}

func count(m *map[string]int, k string) {
	if *m == nil {
		*m = map[string]int{}
	}
	(*m)[k]++
}

// shadowDefault is the log a tier opened in a jitllmtest build shadows into
// (ShadowForTest); nil, and no tier shadows, elsewhere.
var shadowDefault *ShadowLog

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
}

type sliceKind uint8

const (
	sliceAttnIn sliceKind = iota + 1
	slicePostAttn
)

var sliceNames = [...]string{sliceAttnIn: "attention in", slicePostAttn: "post-attention"}

// shadowSlice is one call site's slice of the block that just ran: its
// generic ops, and the block's weights and vectors under their roles, its
// weights' roles in order.
type shadowSlice struct {
	kind    sliceKind
	graph   *graph.Graph
	formats map[jlm.Role]jlm.Type
	weights map[jlm.Role]*resident
	vectors map[jlm.Role]backend.Buf
	roles   []jlm.Role
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

// shadowWhy names what about the step, or this place in it, keeps a slice
// from being shadowed, "" for nothing.
func shadowWhy(R int, ragged, persist, skips bool, here string) string {
	switch {
	case R != 1 || ragged:
		return "not a one-row step"
	case persist:
		return "the persistent block holds its launches"
	case skips:
		return "a bisection probe skips phases"
	}
	return here
}

var shadowActs = map[kernels.ActKind]graph.Act{
	kernels.ActSiLU:      graph.ActSiLU,
	kernels.ActGELU:      graph.ActGELUTanh,
	kernels.ActGELUErf:   graph.ActGELUErf,
	kernels.ActQuickGELU: graph.ActQuickGELU,
	kernels.ActReLU:      graph.ActReLU,
}

// jlmType is a resident's format as the container names it.
func jlmType(q kernels.Quant) (jlm.Type, bool) {
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
	in, err := readF32(s, x, p.NEmbd, nil)
	if err != nil {
		count(&sh.log.Refused, "capture: "+err.Error())
		return
	}
	sh.check(g, lc, s, li, sl, [][]float32{in}, []backend.Buf{q, k, v})
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
	if sh.x, err = readF32(s, x, p.NEmbd, sh.x); err == nil {
		sh.a, err = readF32(s, a, qdim, sh.a)
	}
	if err != nil {
		count(&sh.log.Refused, "capture: "+err.Error())
		return
	}
	sh.li = li
}

// postAttention is the post-attention half: x2 = x + Wo a, then
// x2 + down(act(gate(n)) * up(n)) where n = RMSNorm(x2), held to want, the
// row the tier's own kernels wrote.
func (sh *shadowTier) postAttention(g *devTier, lc *launcher, s backend.Session, l *layer, li int, p *nn.LayerPlan, qdim int, want backend.Buf) {
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
	b.Out(b.Add(x2, b.MatVec(b.ActMul(act, gt, up), jlm.RoleFFNDown, p.NEmbd, biasOf(bias[3], jlm.RoleFFNDownBias)...)))
	sl.graph = b.G
	sh.check(g, lc, s, li, sl, [][]float32{sh.x, sh.a}, []backend.Buf{want})
}

// check forms sl with and without the int8 form, runs both on ins, and holds
// each output to the tier's buffer in wants.
func (sh *shadowTier) check(g *devTier, lc *launcher, s backend.Session, li int, sl *shadowSlice, ins [][]float32, wants []backend.Buf) {
	name := sliceNames[sl.kind]
	arms := [2]*shadowSlice{sl, sl}
	if a, b := sh.log.Swap[0], sh.log.Swap[1]; sl.weights[a] != nil && sl.weights[b] != nil {
		v := *sl
		v.weights = maps.Clone(sl.weights)
		v.weights[a], v.weights[b] = sl.weights[b], sl.weights[a]
		if err := v.setFormats(sl.roles...); err != nil {
			count(&sh.log.Unformed, name+": "+err.Error())
			return
		}
		arms[0] = &v
	}
	var outs [2][][]float32
	for i, float := range [2]bool{false, true} {
		f, err := sh.form(g, arms[i], float)
		if err != nil {
			count(&sh.log.Unformed, name+": "+err.Error())
			return
		}
		if outs[i], err = sh.run(lc, s, f, arms[i], ins); err != nil {
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, %s: %v", li, name, err))
			return
		}
	}
	for j, want := range wants {
		tier, err := readF32(s, want, len(outs[0][j]), nil)
		if err != nil {
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, %s: %v", li, name, err))
			return
		}
		got, ref := outs[0][j], outs[1][j]
		e, et, ec := nmse(got, tier), nmse(tier, ref), nmse(got, ref)
		sh.log.Worst = max(sh.log.Worst, e)
		sh.log.WorstTier, sh.log.WorstComposer = max(sh.log.WorstTier, et), max(sh.log.WorstComposer, ec)
		if et > 0 {
			sh.log.WorstRatio = max(sh.log.WorstRatio, e/et)
		}
		if !(e <= sh.log.Ratio*et+sh.log.Floor) {
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d, %s, output %d: the composer's slice is %.3g from the tier's, whose int8 band here is %.3g (the composer's own %.3g)",
				li, name, j, e, et, ec))
		}
	}
	count(&sh.log.Compared, name)
}

// withoutInt8 is r with no int8 form on the GPU: a matvec decodes its
// weights and keeps its activation float.
func withoutInt8(r *jit.Registry) *jit.Registry {
	fr := *r
	fr.Profiles = maps.Clone(fr.Profiles)
	gp := fr.Profiles[jit.DeviceGPU]
	gp.Int8 = nil
	fr.Profiles[jit.DeviceGPU] = gp
	return &fr
}

func readF32(s backend.Session, b backend.Buf, n int, into []float32) ([]float32, error) {
	into = append(into[:0], make([]float32, n)...)
	if err := s.Read(b, f32b(into)); err != nil {
		return nil, err
	}
	return into, nil
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
			sh.float = withoutInt8(g.composer)
		}
		r = sh.float
	}
	gr := sl.graph
	key, codes, err := r.Form(jit.Request{
		Graph:   gr,
		Formats: sl.formats,
		Step:    jit.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: g.actWin},
		Device:  jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: g.pickLanes()},
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
		i, err := alloc(4 * gr.Values[v].Width)
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
		f.width = append(f.width, gr.Values[v].Width)
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
