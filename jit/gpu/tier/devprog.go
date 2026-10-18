package tier

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The device program (docs/design/composer.md, "The device program"): a
// block's graph lowered to a list of device ops, each one kernel launch over
// typed values, that rewrites fuse and the tier launches. It is the level the
// block encoder (layersSession's emit) skips; a block whose graph the program
// covers runs through it, every other block through the encoder, and the
// manifest says which (BlockForms "block").

// devElem is a value's element type on the device.
type devElem uint8

const (
	elemF32 devElem = iota
	// elemQ8 is the int8 activation with its per-32 scales (bs.a, bs.ax),
	// what an integer matvec reads.
	elemQ8
)

// devLayout is a value's memory layout. A consumer reads one layout; a
// rewrite that hands it another does not build.
type devLayout uint8

const (
	layoutRow devLayout = iota // row-major, one row a token
	layoutKV                   // the paged KV history
)

// devType is a value's type: element, layout and width a row.
type devType struct {
	Elem   devElem
	Layout devLayout
	Width  int
}

func (t devType) String() string {
	e := [...]string{"f32", "q8"}[t.Elem]
	l := [...]string{"row", "kv"}[t.Layout]
	return fmt.Sprintf("%s.%s[%d]", e, l, t.Width)
}

// devKind is a device op's kind.
type devKind uint8

const (
	dkNorm    devKind = iota // RMSNorm of a row by a weight
	dkQuant                  // a float row to the int8 operand a matvec reads
	dkMatVec                 // an integer matvec over a weight
	dkRopeQ                  // q's rotary
	dkKVWrite                // k's rotary into the history, and v's copy
	dkAttend                 // the paged decode attention
	dkAdd                    // a residual add
	dkActMul                 // act(gate) * up
)

var devKindName = [...]string{"norm", "quant", "matvec", "rope_q", "kv_write", "attend", "add", "act_mul"}

func (k devKind) String() string { return devKindName[k] }

// devVal names a value of the program; its buffer is the scratch slot the
// lowering assigns (devSlot).
type devVal int

// devSlot is the block scratch buffer a value lives in.
type devSlot uint8

const (
	slotX   devSlot = iota // bs.x: the block's input and output
	slotX2                 // bs.x2: the residual after attention
	slotH                  // bs.h: a normed row
	slotQ8                 // bs.a / bs.ax: the int8 operand
	slotQ                  // bs.q
	slotK                  // bs.k
	slotV                  // bs.v
	slotQR                 // bs.qr: q rotated
	slotKV                 // the history
	slotXB                 // bs.xb: the attention's output
	slotOut                // bs.mvOut: a projection's output
	slotG                  // bs.g: the gate
	slotU                  // bs.u: up
	slotAct                // bs.act: act(gate)*up
	numSlots
)

var slotName = [...]string{"x", "x2", "h", "q8", "q", "k", "v", "qr", "kv", "xb", "out", "g", "u", "act"}

// devOp is one launch of the program.
type devOp struct {
	Kind devKind
	// Src is the graph ops it lowers (indices into the graph's Ops): one,
	// or several where a lowering takes them together.
	Src  []int
	In   []devVal
	Out  []devVal
	Role jlm.Role // the weight a norm or matvec reads
}

// devProgram is a block's device program.
type devProgram struct {
	vals  []devType
	slots []devSlot
	names []string
	ops   []devOp
}

func (p *devProgram) val(name string, t devType, s devSlot) devVal {
	p.vals = append(p.vals, t)
	p.slots = append(p.slots, s)
	p.names = append(p.names, name)
	return devVal(len(p.vals) - 1)
}

// String is the program, one op a line, each value with its type and slot.
func (p *devProgram) String() string {
	var b strings.Builder
	v := func(x devVal) string { return fmt.Sprintf("%%%s:%s@%s", p.names[x], p.vals[x], slotName[p.slots[x]]) }
	for _, o := range p.ops {
		var outs, ins []string
		for _, x := range o.Out {
			outs = append(outs, v(x))
		}
		for _, x := range o.In {
			ins = append(ins, v(x))
		}
		fmt.Fprintf(&b, "  %s = %s(%s", strings.Join(outs, ", "), o.Kind, strings.Join(ins, ", "))
		if o.Role != 0 {
			fmt.Fprintf(&b, ", %s", o.Role)
		}
		fmt.Fprintf(&b, ")  ; graph ops %v\n", o.Src)
	}
	return b.String()
}

// wants is the type each kind reads on each input, and what it writes; the
// check holds every edge to it.
func (o devOp) wants(p *devProgram, i int) devType {
	in := p.vals[o.In[i]]
	switch o.Kind {
	case dkMatVec:
		return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
	case dkQuant, dkNorm, dkRopeQ, dkAdd, dkActMul, dkKVWrite, dkAttend:
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	}
	return in
}

// check holds every op's inputs to the types the op reads: a value in a
// layout or element type its consumer does not read refuses the program.
func (p *devProgram) check() error {
	for i, o := range p.ops {
		for j := range o.In {
			got, want := p.vals[o.In[j]], o.wants(p, j)
			if got != want {
				return fmt.Errorf("device program: op %d (%s) reads %%%s as %s, which is %s",
					i, o.Kind, p.names[o.In[j]], want, got)
			}
		}
	}
	return nil
}

// d1Kinds is the graph ops the first slice of the program lowers.
var d1Kinds = map[graph.OpKind]bool{
	graph.OpNorm: true, graph.OpMatVec: true, graph.OpRope: true, graph.OpKVAppend: true,
	graph.OpAttend: true, graph.OpAdd: true, graph.OpActMul: true,
}

// buildDevProgram lowers p's graph to a device program, or says why it does
// not: an op or an attribute the program does not lower yet. The reason is
// what BlockProgram reports for a block left on the encoder.
func buildDevProgram(p *nn.LayerPlan, l *layer) (*devProgram, string) {
	g := p.Graph
	switch {
	case g == nil:
		return nil, "no graph"
	case p.NonCausal:
		return nil, "non-causal attention"
	case p.AttnSinks:
		return nil, "attention sinks"
	case p.NoPEGlobal || p.QKL2Norm || p.QKNorm || p.QKNormPost || p.RopeSplit:
		return nil, "a q/k feature (norm, NoPE, XD-RoPE)"
	case partialRotary(p):
		return nil, "a partial rotary"
	case l.vFromK || l.kvSrc != g.Block:
		return nil, "a shared or derived history"
	case l.nAttnB != nil || l.nFFNB != nil:
		return nil, "a biased norm"
	case l.mvq.bias != nil || l.mvk.bias != nil || l.mvv.bias != nil || l.mvo.bias != nil ||
		l.mvg.bias != nil || l.mvu.bias != nil || l.mvd.bias != nil:
		return nil, "a projection bias"
	case p.Act != nn.ActSiLU:
		return nil, "an activation other than SiLU"
	}
	for _, o := range g.Ops {
		if !d1Kinds[o.Kind] {
			return nil, "op " + o.Kind.String()
		}
		if o.Kind == graph.OpAttend && (o.Attr.Chunked || o.Attr.Bidir) {
			return nil, "an attention attribute"
		}
	}
	prog := &devProgram{}
	ne, qd, kv, nf := p.NEmbd, p.NHead*p.HeadDim, p.KVRow(), p.NFFN
	f32 := func(w int) devType { return devType{Elem: elemF32, Layout: layoutRow, Width: w} }
	vals := map[graph.ValueID]devVal{g.In[0]: prog.val("x", f32(ne), slotX)}
	// The int8 operand of a matvec: the float row it reads, quantized once
	// per source (q, k and v read one quantize of the normed row).
	quantOf := map[devVal]devVal{}
	operand := func(src devVal, op int) devVal {
		if q, ok := quantOf[src]; ok {
			return q
		}
		q := prog.val(prog.names[src]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[src].Width}, slotQ8)
		prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{op}, In: []devVal{src}, Out: []devVal{q}})
		quantOf = map[devVal]devVal{src: q} // one int8 operand buffer: the last quantize wins
		return q
	}
	matvecSlot := map[jlm.Role]devSlot{
		jlm.RoleAttnQ: slotQ, jlm.RoleAttnK: slotK, jlm.RoleAttnV: slotV, jlm.RoleAttnOut: slotOut,
		jlm.RoleFFNGate: slotG, jlm.RoleFFNUp: slotU, jlm.RoleFFNDown: slotOut,
	}
	adds, norms := 0, 0
	var kRot graph.ValueID = -1
	for i, o := range g.Ops {
		in := func(j int) devVal { return vals[o.In[j]] }
		name := func(j int) string { return g.Values[o.Out[j]].Name }
		switch o.Kind {
		case graph.OpNorm:
			if len(o.Weight) != 1 {
				return nil, "a norm without one weight"
			}
			out := prog.val(name(0), f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: dkNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
			vals[o.Out[0]] = out
			norms++
		case graph.OpMatVec:
			role := o.Weight[0].Role
			s, ok := matvecSlot[role]
			if !ok {
				return nil, "a matvec over " + role.String()
			}
			w := map[devSlot]int{slotQ: qd, slotK: kv, slotV: kv, slotOut: ne, slotG: nf, slotU: nf}[s]
			a := operand(in(0), i)
			out := prog.val(name(0), f32(w), s)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{a}, Out: []devVal{out}, Role: role})
			vals[o.Out[0]] = out
		case graph.OpRope:
			if prog.slots[in(0)] == slotQ {
				out := prog.val(name(0), f32(qd), slotQR)
				prog.ops = append(prog.ops, devOp{Kind: dkRopeQ, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
			} else {
				// k's rotary writes the history itself (the device has no
				// rotated-k row): it lowers with the kv_append that reads it.
				kRot = o.Out[0]
				vals[o.Out[0]] = in(0)
			}
		case graph.OpKVAppend:
			if o.In[0] != kRot {
				return nil, "a history append of an unrotated key"
			}
			hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: kv}, slotKV)
			prog.ops = append(prog.ops, devOp{Kind: dkKVWrite, Src: []int{i - 1, i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}})
		case graph.OpAttend:
			out := prog.val(name(0), f32(qd), slotXB)
			prog.ops = append(prog.ops, devOp{Kind: dkAttend, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpAdd:
			s := slotX2
			if adds > 0 {
				s = slotX
			}
			out := prog.val(name(0), f32(ne), s)
			prog.ops = append(prog.ops, devOp{Kind: dkAdd, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
			adds++
		case graph.OpActMul:
			out := prog.val(name(0), f32(nf), slotAct)
			prog.ops = append(prog.ops, devOp{Kind: dkActMul, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		}
	}
	if adds != 2 || norms != 2 || len(g.Out) != 1 || prog.slots[vals[g.Out[0]]] != slotX {
		return nil, "not a pre-norm block of two residual adds"
	}
	if err := prog.check(); err != nil {
		return nil, err.Error()
	}
	return prog, ""
}

// BlockProgram is block li's device program as text and, where its last step
// did not run it (or the program does not lower the block), why; both empty
// where the device does not hold the block.
func (g *GPU) BlockProgram(li int) (prog, why string) {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return "", ""
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	l := d.layers[li]
	if l == nil {
		return "", ""
	}
	if l.prog == nil {
		return "", l.progWhy
	}
	return l.prog.String(), l.progRun
}
