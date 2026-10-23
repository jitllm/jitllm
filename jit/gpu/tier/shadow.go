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

// The shadow (ShadowForTest, jitllmtest builds only) asks the composer for a
// slice of each block the tier has just run, as the call site will once the
// composer serves it: it runs the composer's kernels on the same inputs into
// its own buffers and compares them with what the tier's own kernels wrote.
// What the composer cannot form, and what does not match, is the work list
// before the switch. The composer and its kernels may split or fuse the
// slice their own way; only its float edges are compared.

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
	// SwapGateUp binds the gate's weight as up's and up's as the gate's, a
	// violation the compare must see.
	SwapGateUp bool
	Compared int
	// Worst is the largest NMSE between the composer's slice and the
	// tier's; WorstTier and WorstComposer each one's largest from the float
	// slice.
	Worst, WorstTier, WorstComposer float64
	// WorstRatio is the largest composer-to-tier distance as a fraction of
	// its slice's int8 band.
	WorstRatio float64
	// Refused counts blocks not shadowed, by reason; Unformed slices the
	// composer could not form, by its error; Failed compares over Margin.
	Refused  map[string]int
	Unformed map[string]int
	Failed   []string
}

func (s *ShadowLog) refuse(why string) {
	if s.Refused == nil {
		s.Refused = map[string]int{}
	}
	s.Refused[why]++
}

// shadowDefault is the log a tier opened in a jitllmtest build shadows into
// (ShadowForTest); nil, and no tier shadows, elsewhere.
var shadowDefault *ShadowLog

// shadowTier is one tier's shadow: the slices the composer formed, by shape,
// and the inputs captured from the block running now.
type shadowTier struct {
	log   *ShadowLog
	forms map[shadowKey]*shadowForm
	// float is the composer with no int8 form: the float slice's.
	float *jit.Registry
	// li is the block whose inputs x and a hold, -1 for none.
	li   int
	x, a []float32
}

// shadowKey is a post-attention slice's shape: what the composer's kernels
// for it depend on.
type shadowKey struct {
	embd, qdim, nffn, win int
	act                   graph.Act
	eps                   uint64
	o, g, u, d            jlm.Type
	bias                  [4]bool
	float                 bool
}

// shadowForm is a slice's kernels compiled on the device with the buffers
// the composer's own operands bind: the inputs uploaded, the output, the
// values between its ops, its scratch and constants.
type shadowForm struct {
	codes  []formedCode
	sargs  [][]shadowArg
	bufs   []backend.Buf
	in     []int // the slice's inputs' buffers, in graph.In order
	out    int
	widths []int
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
	return &shadowTier{log: log, forms: map[shadowKey]*shadowForm{}, li: -1}
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

// shadowPlain is "" for a block whose post-attention half is the plain
// slice: o, the residual add, an RMSNorm, a gated MLP and its add. Anything
// else names itself, not shadowed yet.
func shadowPlain(l *layer, p *nn.LayerPlan) string {
	switch {
	case l.linear || l.ds4 != nil || l.k3 != nil || l.q4e != nil || l.altRouter != nil || l.mla:
		return "not an attention block's plain residual"
	case l.noFFN:
		return "no FFN"
	case l.router != nil || l.gate == nil || l.up == nil || l.down == nil || l.wo == nil:
		return "not a gated dense FFN"
	case l.ungated || p.Parallel:
		return "ungated or parallel FFN"
	case l.nPostAttn != nil || l.nPostFFN != nil || l.nSubAttn != nil || l.nSubFFN != nil || l.nResid != nil:
		return "a norm beside the residual"
	case l.nFFNB != nil || p.LayerNorm:
		return "LayerNorm"
	case l.clampB != nil:
		return "clamped activations"
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

// jlmType is a resident's format as the container names it.
func jlmType(q kernels.Quant) (jlm.Type, bool) {
	for _, t := range quant.PackedTypes {
		if k, ok := kernels.QuantOf(t); ok && k == q {
			return jlm.TypeOf(t)
		}
	}
	return 0, false
}

// capture reads the block's input row and its attention output, the inputs
// of the post-attention slice, before the tier's kernels overwrite them.
func (sh *shadowTier) capture(g *devTier, s backend.Session, l *layer, li int, p *nn.LayerPlan, x, a backend.Buf, qdim int, why string) {
	sh.li = -1
	if why == "" {
		why = shadowPlain(l, p)
	}
	if why != "" {
		sh.log.refuse(why)
		return
	}
	var err error
	if sh.x, err = readF32(s, x, p.NEmbd, sh.x); err == nil {
		sh.a, err = readF32(s, a, qdim, sh.a)
	}
	if err != nil {
		sh.log.refuse("capture: " + err.Error())
		return
	}
	sh.li = li
}

func readF32(s backend.Session, b backend.Buf, n int, into []float32) ([]float32, error) {
	into = append(into[:0], make([]float32, n)...)
	if err := s.Read(b, f32b(into)); err != nil {
		return nil, err
	}
	return into, nil
}

// compare runs the composer's slice on the captured inputs and holds its
// output to want, the row the tier's own kernels wrote.
func (sh *shadowTier) compare(g *devTier, lc *launcher, s backend.Session, l *layer, li int, p *nn.LayerPlan, qdim int, want backend.Buf) {
	if sh.li != li {
		return
	}
	sh.li = -1
	var outs [2][]float32
	for i, float := range [2]bool{false, true} {
		f, err := sh.form(g, l, p, qdim, float)
		if err != nil {
			if sh.log.Unformed == nil {
				sh.log.Unformed = map[string]int{}
			}
			sh.log.Unformed[err.Error()]++
			return
		}
		if outs[i], err = sh.run(g, lc, s, f, l, !float && sh.log.SwapGateUp); err != nil {
			sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d: %v", li, err))
			return
		}
	}
	tier, err := readF32(s, want, p.NEmbd, nil)
	if err != nil {
		sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d: %v", li, err))
		return
	}
	got, ref := outs[0], outs[1]
	e, et, ec := nmse(got, tier), nmse(tier, ref), nmse(got, ref)
	sh.log.Compared++
	sh.log.Worst = max(sh.log.Worst, e)
	sh.log.WorstTier, sh.log.WorstComposer = max(sh.log.WorstTier, et), max(sh.log.WorstComposer, ec)
	if et > 0 {
		sh.log.WorstRatio = max(sh.log.WorstRatio, e/et)
	}
	if !(e <= sh.log.Ratio*et+sh.log.Floor) {
		sh.log.Failed = append(sh.log.Failed, fmt.Sprintf("block %d: the composer's slice is %.3g from the tier's, whose int8 band here is %.3g (the composer's own %.3g)", li, e, et, ec))
	}
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

// postAttention is the post-attention half as generic ops: x2 = x + Wo a, then
// x2 + down(act(gate(n)) * up(n)) where n = RMSNorm(x2).
func postAttention(p *nn.LayerPlan, qdim, nffn int, act graph.Act, bias [4]bool) *graph.Graph {
	b := graph.NewSlice()
	x := b.In("x", p.NEmbd)
	a := b.In("attn", qdim)
	withBias := func(on bool, r jlm.Role) []jlm.Role {
		if on {
			return []jlm.Role{r}
		}
		return nil
	}
	x2 := b.Add(x, b.MatVec(a, jlm.RoleAttnOut, p.NEmbd, withBias(bias[0], jlm.RoleAttnOutBias)...))
	n := b.Norm(x2, jlm.RoleFFNNorm, p.RMSEps)
	gt := b.MatVec(n, jlm.RoleFFNGate, nffn)
	up := b.MatVec(n, jlm.RoleFFNUp, nffn, withBias(bias[2], jlm.RoleFFNUpBias)...)
	b.Out(b.Add(x2, b.MatVec(b.ActMul(act, gt, up), jlm.RoleFFNDown, p.NEmbd, withBias(bias[3], jlm.RoleFFNDownBias)...)))
	return b.G
}

func (sh *shadowTier) form(g *devTier, l *layer, p *nn.LayerPlan, qdim int, float bool) (*shadowForm, error) {
	act, ok := shadowActs[l.act]
	if !ok {
		return nil, fmt.Errorf("activation %d has no generic op", l.act)
	}
	nffn := l.gate.nrows
	key := shadowKey{embd: p.NEmbd, qdim: qdim, nffn: nffn, win: g.actWin, act: act, eps: math.Float64bits(p.RMSEps),
		bias: [4]bool{l.mvo.bias != nil, l.mvg.bias != nil, l.mvu.bias != nil, l.mvd.bias != nil}, float: float}
	if key.bias[1] {
		return nil, fmt.Errorf("a gate bias has no role")
	}
	ts := [4]*jlm.Type{&key.o, &key.g, &key.u, &key.d}
	for i, r := range [4]*resident{l.wo, l.gate, l.up, l.down} {
		t, ok := jlmType(r.t)
		if !ok {
			return nil, fmt.Errorf("format %d has no container type", r.t)
		}
		*ts[i] = t
	}
	if f, ok := sh.forms[key]; ok {
		if f == nil {
			return nil, fmt.Errorf("the slice at %+v did not form earlier", key)
		}
		return f, nil
	}
	sh.forms[key] = nil
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
	gr := postAttention(p, qdim, nffn, act, key.bias)
	codes, err := r.Form(jit.Request{
		Graph: gr,
		Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnOut: key.o, jlm.RoleFFNGate: key.g, jlm.RoleFFNUp: key.u,
			jlm.RoleFFNDown: key.d},
		Step:   jit.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: g.actWin},
		Device: jit.Device{Kind: jit.DeviceGPU, Slots: g.dev.Slots(), Lanes: g.pickLanes()},
	})
	if err != nil {
		return nil, err
	}
	f := &shadowForm{out: -1}
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
	for _, v := range gr.In {
		i, err := alloc(4 * gr.Values[v].Width)
		if err != nil {
			f.close()
			return nil, err
		}
		vals[v] = i
		f.in = append(f.in, i)
		f.widths = append(f.widths, gr.Values[v].Width)
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
				i, ok := vals[o.Value]
				if !ok {
					if i, err = alloc(4 * gr.Values[o.Value].Width); err != nil {
						f.close()
						return nil, err
					}
					vals[o.Value] = i
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
	f.out = vals[gr.Out[0]]
	sh.forms[key] = f
	return f, nil
}

// run uploads the captured inputs, launches the slice's kernels on this
// block's weights and reads the output back.
func (sh *shadowTier) run(g *devTier, lc *launcher, s backend.Session, f *shadowForm, l *layer, swap bool) ([]float32, error) {
	for i, v := range [][]float32{sh.x, sh.a} {
		if err := s.Write(f.bufs[f.in[i]], f32b(v)); err != nil {
			return nil, err
		}
	}
	weights := map[jlm.Role]*resident{jlm.RoleAttnOut: l.wo, jlm.RoleFFNGate: l.gate, jlm.RoleFFNUp: l.up, jlm.RoleFFNDown: l.down}
	if swap {
		weights[jlm.RoleFFNGate], weights[jlm.RoleFFNUp] = l.up, l.gate
	}
	vectors := map[jlm.Role]backend.Buf{jlm.RoleFFNNorm: l.nFFN, jlm.RoleAttnOutBias: l.mvo.bias,
		jlm.RoleFFNUpBias: l.mvu.bias, jlm.RoleFFNDownBias: l.mvd.bias}
	for i := range f.codes {
		fc := &f.codes[i]
		for j, a := range f.sargs[i] {
			switch {
			case a.buf >= 0:
				fc.bufs[j] = f.bufs[a.buf]
			case a.vector:
				fc.bufs[j] = vectors[a.role]
			default:
				r := weights[a.role]
				fc.bufs[j] = [3]backend.Buf{r.qs, r.d, r.sc}[a.region]
			}
			if fc.bufs[j] == nil {
				return nil, fmt.Errorf("the composer's kernel %d reads %v, which the block does not hold", i, a.role)
			}
		}
		if err := lc.launch(fc.k, fc.groups, fc.width, fc.bufs...); err != nil {
			return nil, err
		}
	}
	return readF32(s, f.bufs[f.out], len(sh.x), nil)
}

// shadowWhy names what about the step keeps its blocks from being shadowed,
// "" for nothing.
func shadowWhy(R int, ragged, persist, skips, outGate bool) string {
	switch {
	case R != 1 || ragged:
		return "not a one-row step"
	case persist:
		return "the persistent block holds its launches"
	case skips:
		return "a bisection probe skips phases"
	case outGate:
		return "an output gate"
	}
	return ""
}
