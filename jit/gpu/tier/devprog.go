package tier

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/backend"
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
	// elemF16 is the binary16 operand a GEMM reads, in its layout.
	elemF16
)

// devLayout is a value's memory layout. A consumer reads one layout; a
// rewrite that hands it another does not build.
type devLayout uint8

const (
	layoutRow    devLayout = iota // row-major, one row a token
	layoutKV                      // the paged KV history
	layoutTokF16                  // token-major binary16 (kernels.ActF16T)
	layoutFrag                    // the MMA's m16n8 fragments (kernels.ActF16)
)

// devType is a value's type: element, layout and width a row.
type devType struct {
	Elem   devElem
	Layout devLayout
	Width  int
}

func (t devType) String() string {
	e := [...]string{"f32", "q8", "f16"}[t.Elem]
	l := [...]string{"row", "kv", "tok", "frag"}[t.Layout]
	return fmt.Sprintf("%s.%s[%d]", e, l, t.Width)
}

// devKind is a device op's kind.
type devKind uint8

const (
	dkNorm         devKind = iota // RMSNorm of a row by a weight
	dkQuant                       // a float row to the int8 operand a matvec reads
	dkMatVec                      // an integer matvec over a weight
	dkRopeQ                       // q's rotary
	dkKVWrite                     // k's rotary into the history, and v's copy
	dkAttend                      // the paged decode attention
	dkAdd                         // a residual add
	dkActMul                      // act(gate) * up
	dkHeadNorm                    // each head's RMSNorm (q's or k's)
	dkPostNorm                    // a sublayer's output normed before its residual add (post-norm)
	dkMixture                     // a routed mixture and its shared expert, from route to its sum (mixFFN)
	dkRecurrent                   // a recurrent mixer with its pre-norm, from the block's row (emitLinear)
	dkCopy                        // a row moved to another slot (the sum home to the block's row)
	dkKVCopy                      // an unrotated key and its value copied into the history (NoPE)
	dkAct                         // an ungated FFN's activation of up's row
	dkAttendLatent                // latent attention, from the normed row to its output projection (emitMLA)
	dkQTemp                       // the query's temperature, a factor of each row's position
	dkSplitQG                     // a double-width query split into its heads and its output gate
	dkGateOut                     // the attention's output times the sigmoid of the gate
	dkPerLayer                    // the per-layer embedding added to the block's row (perLayerEmbed)
	dkScale                       // the block's row times the layer's output scale
	dkClamp                       // q, k or v clamped to the plan's bound
	dkSubNorm                     // an inner norm before wo or down, with the quantize it feeds
	numBaseKinds                  // the fused kinds (devrewrite.go) count on from here
)

// devVal names a value of the program; its buffer is the scratch slot the
// lowering assigns (devSlot).
type devVal int

// devSlot is the block scratch buffer a value lives in.
type devSlot uint8

const (
	slotX      devSlot = iota // bs.x: the block's input and output
	slotX2                    // bs.x2: the residual after attention
	slotH                     // bs.h: a normed row
	slotQ8                    // bs.a / bs.ax: the int8 operand
	slotQ                     // bs.q
	slotK                     // bs.k
	slotV                     // bs.v
	slotQR                    // bs.qr: q rotated
	slotKV                    // the history
	slotXB                    // bs.xb: the attention's output
	slotOut                   // bs.mvOut: a projection's output
	slotG                     // bs.g: the gate
	slotU                     // bs.u: up
	slotAct                   // bs.act: act(gate)*up
	slotQN                    // bs.qn: q's heads normed
	slotKN                    // bs.kn: k's heads normed
	slotF16                   // g.f16Buf: a GEMM's binary16 operand
	slotMix                   // the mixture's output: whichever buffer mixFFN returns
	slotQT                    // bs.qt: a rotated query scaled by its temperature
	slotQG                    // bs.qg: the query's heads, split from a gated query
	slotOGate                 // bs.ogate: the output gate
	slotOGated                // bs.ogated: the attention's output, gated
	slotVN                    // bs.vn: v's heads normed
	slotQC                    // bs.qClamp
	slotKC                    // bs.kClamp
	slotVC                    // bs.vClamp
	slotSub                   // bs.sub: an inner norm's row
	slotSide                  // bs.side: a recurrent mixer's output beside the attention
	numSlots
)

func (s devSlot) String() string {
	switch s {
	case slotX:
		return "x"
	case slotX2:
		return "x2"
	case slotH:
		return "h"
	case slotQ8:
		return "q8"
	case slotQ:
		return "q"
	case slotK:
		return "k"
	case slotV:
		return "v"
	case slotQR:
		return "qr"
	case slotKV:
		return "kv"
	case slotXB:
		return "xb"
	case slotOut:
		return "out"
	case slotG:
		return "g"
	case slotU:
		return "u"
	case slotAct:
		return "act"
	case slotQN:
		return "qn"
	case slotKN:
		return "kn"
	case slotF16:
		return "f16"
	case slotMix:
		return "mix"
	case slotQT:
		return "qt"
	case slotQG:
		return "qg"
	case slotOGate:
		return "ogate"
	case slotOGated:
		return "ogated"
	case slotVN:
		return "vn"
	case slotQC:
		return "qc"
	case slotKC:
		return "kc"
	case slotVC:
		return "vc"
	case slotSub:
		return "sub"
	case slotSide:
		return "side"
	}
	return ""
}

// devOp is one launch of the program.
type devOp struct {
	Kind devKind
	// Src is the graph ops it lowers (indices into the graph's Ops): one,
	// or several where a lowering takes them together.
	Src  []int
	In   []devVal
	Out  []devVal
	Role jlm.Role // the weight a norm or matvec reads
	// Need overrides what the op reads on an input: a GEMM's binary16
	// operand is the layout its batched twin reads.
	Need []devType
	// K and Conv are a binary16 writer's kernel and the conversion it
	// stands in for (rmsF16For, actF16For), chosen when it was rewritten.
	K, Conv backend.Kernel
	// Staged says the op is several launches that pass through scratch, so
	// reading and writing one slot is safe: no single launch does both.
	Staged bool
	N      int     // K's thread count, where it is not the rows'
	QS     *qkvSeg // the q/k/v grid's segments (qkv-gemm)
}

// devProgram is a block's device program.
type devProgram struct {
	// f16 says the program's matvecs are binary16 GEMMs reading the float
	// rows (a prompt chunk on a card whose twins are all binary16): no
	// int8 operand, each GEMM converting its row or finding it written.
	f16 bool
	// attends says the program runs an attention: it reads the paged history;
	// ropes that it rotates q and k.
	attends, ropes bool
	vals           []devType
	slots          []devSlot
	names          []string
	ops            []devOp
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
	v := func(x devVal) string { return fmt.Sprintf("%%%s:%s@%s", p.names[x], p.vals[x], p.slots[x]) }
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
	if i < len(o.Need) && o.Need[i] != (devType{}) {
		return o.Need[i]
	}
	if p.f16 && (o.Kind == dkMatVec || o.Kind == dkQKV || o.Kind == dkQKVGemm) {
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	}
	switch o.Kind {
	case dkRecurrent, dkCopy, dkKVCopy, dkAct, dkQTemp, dkSplitQG, dkGateOut, dkPerLayer, dkScale, dkClamp, dkSubNorm:
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkMixture, dkAttendLatent:
		if i == 1 {
			return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkMatVec, dkQKV, dkGate2:
		return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
	case dkMatVecRes, dkMatVecGate:
		if i == 0 {
			return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkQuant, dkNorm, dkRopeQ, dkAdd, dkActMul, dkKVWrite, dkAttend, dkHeadNorm, dkPostNorm,
		dkNormQuant, dkAddNorm, dkAddNormQuant, dkActQuant, dkRopeQKV, dkGate2In,
		dkQKVIn, dkQKVSep, dkResIn, dkFFNIn, dkDownAct, dkNormF16, dkAddNormF16, dkActF16:
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	}
	return in
}

// check holds every op's inputs to the types the op reads: a value in a
// layout or element type its consumer does not read refuses the program.
func (p *devProgram) check() error {
	// Every value an op reads is written by an earlier op or is the block's
	// input: a rewrite that moves a launch above what it reads is refused.
	written := make([]int, len(p.vals))
	for i := range written {
		written[i] = -1
	}
	for i, o := range p.ops {
		for _, y := range o.Out {
			if written[y] < 0 {
				written[y] = i
			}
		}
	}
	for i, o := range p.ops {
		for _, x := range o.In {
			if w := written[x]; w >= i {
				return fmt.Errorf("device program: op %d (%s) reads %%%s, which op %d writes after it", i, o.Kind, p.names[x], w)
			}
		}
	}
	for i, o := range p.ops {
		// RULE 13: no launch reads a buffer it writes (a staged op's
		// launches go through scratch).
		for _, x := range o.In {
			if o.Staged {
				break
			}
			for _, y := range o.Out {
				if p.slots[x] == p.slots[y] && p.slots[x] != slotKV {
					return fmt.Errorf("device program: op %d (%s) reads and writes %s", i, o.Kind, p.slots[x])
				}
			}
		}
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

// lowersOp reports the program lowers graph ops of kind k.
func lowersOp(k graph.OpKind) bool {
	switch k {
	case graph.OpNorm, graph.OpMatVec, graph.OpRope, graph.OpKVAppend, graph.OpAttend, graph.OpAdd,
		graph.OpActMul, graph.OpAct, graph.OpHeadNorm, graph.OpLayerNorm, graph.OpScale, graph.OpClamp,
		// The mixture's ops, lowered whole as one op (dkMixture) for now.
		graph.OpRoute, graph.OpGather, graph.OpExpertMatVec, graph.OpCombine, graph.OpGateScalar,
		// A recurrent mixer, lowered whole with its pre-norm (emitLinear).
		graph.OpRecurrent,
		// Latent attention's own ops, lowered whole (emitMLA).
		graph.OpSlice, graph.OpLatentAppend, graph.OpAttendLatent:
		return true
	}
	return false
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
	case p.RopeSplit:
		return nil, "XD-RoPE"
	case !actHasKernel(p):
		return nil, "an activation with no kernel"
	case bindsRole(g, jlm.RoleFFNRoutedDown):
		// Read off the graph, not the architecture: experts run in a latent
		// (the row taken down before them).
		return nil, "latent experts"
	}
	for _, o := range g.Ops {
		if !lowersOp(o.Kind) {
			return nil, "op " + o.Kind.String()
		}
		// A window or an aligned chunk is the plan's per-row key start (the
		// descriptors); a bidirectional sequence is an encoder's, not a
		// paged history's.
		if o.Kind == graph.OpAttend && o.Attr.Bidir {
			return nil, "a bidirectional attention"
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
	adds := 0
	var kRot graph.ValueID = -1
	// Latent attention lowers whole: the region from the first op that reads
	// the attention norm's row to the value the attention's residual add
	// reads.
	mla, attnRow := false, graph.ValueID(-1)
	for _, o := range g.Ops {
		switch {
		case o.Kind == graph.OpAttendLatent:
			mla = true
		case (o.Kind == graph.OpNorm || o.Kind == graph.OpLayerNorm) && len(o.Weight) > 0 && o.Weight[0].Role == jlm.RoleAttnNorm:
			attnRow = o.Out[0]
		}
	}
	if mla && bindsRole(g, jlm.RoleAttnGate) {
		return nil, "an output gate on latent attention"
	}
	skipTo := 0
	for i, o := range g.Ops {
		if i < skipTo {
			continue // inside a region lowered whole (the mixture, latent attention)
		}
		in := func(j int) devVal { return vals[o.In[j]] }
		if mla && skipTo <= i && len(o.In) > 0 && o.In[0] == attnRow && o.Kind != graph.OpNorm && o.Kind != graph.OpLayerNorm {
			a := -1
			for k := i; k < len(g.Ops) && a < 0; k++ {
				if g.Ops[k].Kind == graph.OpAdd && (g.Ops[k].In[0] == g.In[0] || g.Ops[k].In[1] == g.In[0]) {
					a = k
				}
			}
			if a < 0 {
				return nil, "latent attention with no residual add"
			}
			res := g.Ops[a].In[1]
			if res == g.In[0] {
				res = g.Ops[a].In[0]
			}
			src := make([]int, 0, a-i)
			for k := i; k < a; k++ {
				src = append(src, k)
			}
			q := operand(in(0), i)
			out := prog.val(g.Values[res].Name, f32(ne), slotOut)
			prog.ops = append(prog.ops, devOp{Kind: dkAttendLatent, Src: src, In: []devVal{in(0), q}, Out: []devVal{out}})
			vals[res] = out
			skipTo = a
			mla = false
			continue
		}
		name := func(j int) string { return g.Values[o.Out[j]].Name }
		switch o.Kind {
		case graph.OpNorm, graph.OpLayerNorm:
			// A LayerNorm binds its bias second; the block's norm (norm) is
			// the form its plan was built with, and reads the bias.
			if len(o.Weight) == 0 || o.Kind == graph.OpNorm && len(o.Weight) != 1 {
				return nil, "a norm without its weight"
			}
			k := dkNorm
			switch o.Weight[0].Role {
			case jlm.RoleAttnSubNorm, jlm.RoleFFNSubNorm:
				// BitNet's inner norm, fused with the quantize its
				// projection reads: that projection takes the int8 row
				// written here.
				sub := prog.val(name(0), prog.vals[in(0)], slotSub)
				q := prog.val(name(0)+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[in(0)].Width}, slotQ8)
				prog.ops = append(prog.ops, devOp{Kind: dkSubNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{sub, q}, Role: o.Weight[0].Role})
				quantOf = map[devVal]devVal{sub: q}
				vals[o.Out[0]] = sub
				continue
			case jlm.RoleAttnNorm, jlm.RoleFFNNorm:
			case jlm.RolePostAttnNorm, jlm.RolePostFFNNorm:
				k = dkPostNorm
			case jlm.RoleAttnQNorm, jlm.RoleAttnKNorm:
				// A whole-projection q/k norm (OLMo 2's, OLMoE's): the head
				// norm's lowering at the plan's wide geometry.
				if !p.QKNormWide {
					return nil, "a row norm of q or k on a plan that is not wide"
				}
				s := slotQN
				if prog.slots[in(0)] == slotK {
					s = slotKN
				}
				out := prog.val(name(0), prog.vals[in(0)], s)
				prog.ops = append(prog.ops, devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
				vals[o.Out[0]] = out
				continue
			default:
				return nil, "a norm over " + o.Weight[0].Role.String()
			}
			out := prog.val(name(0), f32(ne), slotH)
			prog.ops = append(prog.ops, devOp{Kind: k, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
			vals[o.Out[0]] = out
		case graph.OpMatVec:
			role := o.Weight[0].Role
			if role == jlm.RoleShExpGate && p.DenseMoE {
				// A dense MLP beside the mixture: from its gate to the add of
				// the two branches, lowered whole with the mixture (mixFFN
				// runs both, from the normed row and the attention's sum).
				a := -1
				for k := i + 1; k < len(g.Ops) && a < 0; k++ {
					if g.Ops[k].Kind == graph.OpAdd && readsRoleOut(g, g.Ops[k], jlm.RolePostFFNNorm1) {
						a = k
					}
				}
				if a < 0 {
					return nil, "a dense MLP beside the mixture with no sum"
				}
				src := make([]int, 0, a+1-i)
				for k := i; k <= a; k++ {
					src = append(src, k)
				}
				q := operand(in(0), i)
				out := prog.val(g.Values[g.Ops[a].Out[0]].Name, f32(ne), slotMix)
				prog.ops = append(prog.ops, devOp{Kind: dkMixture, Src: src, In: []devVal{in(0), q}, Out: []devVal{out}})
				vals[g.Ops[a].Out[0]] = out
				skipTo = a + 1
				continue
			}
			if role == jlm.RolePLEGate {
				// The per-layer embedding: its gate to the add that brings
				// it into the row, lowered whole (perLayerEmbed).
				a := -1
				for k := i + 1; k < len(g.Ops) && a < 0; k++ {
					if g.Ops[k].Kind == graph.OpAdd && (g.Ops[k].In[0] == o.In[0] || g.Ops[k].In[1] == o.In[0]) {
						a = k
					}
				}
				if a < 0 {
					return nil, "a per-layer embedding with no add"
				}
				src := make([]int, 0, a+1-i)
				for k := i; k <= a; k++ {
					src = append(src, k)
				}
				out := prog.val(g.Values[g.Ops[a].Out[0]].Name, f32(ne), slotX)
				prog.ops = append(prog.ops, devOp{Kind: dkPerLayer, Src: src, In: []devVal{in(0)}, Out: []devVal{out}, Staged: true})
				vals[g.Ops[a].Out[0]] = out
				skipTo = a + 1
				continue
			}
			s, ok := matvecSlot[role]
			if !ok {
				return nil, "a matvec over " + role.String()
			}
			w := o.Weight[0].Rows
			if w == 0 {
				return nil, "a matvec with no stated width"
			}
			a := operand(in(0), i)
			out := prog.val(name(0), f32(w), s)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{a}, Out: []devVal{out}, Role: role})
			vals[o.Out[0]] = out
		case graph.OpRecurrent:
			// emitLinear runs the block's pre-norm and quantize itself, from
			// the block's row, and writes the mixer's output: the norm the
			// graph states before it goes into this op.
			src := []int{i}
			x, normed := in(0), devVal(-1)
			if n := len(prog.ops); n > 0 && prog.ops[n-1].Kind == dkNorm && prog.ops[n-1].Out[0] == x {
				src = append(prog.ops[n-1].Src, i)
				normed, x = x, prog.ops[n-1].In[0]
				prog.ops = prog.ops[:n-1]
			}
			if prog.slots[x] != slotX {
				return nil, "a recurrent mixer not over the block's row"
			}
			// Beside an attention (the graph has both) the mixer leaves its
			// output in bs.side for the sum, and the normed row it wrote is
			// the attention's.
			s := slotOut
			if hasOp(g, graph.OpAttend) {
				s = slotSide
			}
			out := prog.val(name(0), f32(ne), s)
			outs := []devVal{out}
			if normed >= 0 {
				outs = append(outs, normed)
			}
			prog.ops = append(prog.ops, devOp{Kind: dkRecurrent, Src: src, In: []devVal{x}, Out: outs})
			vals[o.Out[0]] = out
		case graph.OpRoute:
			// The mixture: from the router to the value the block's last
			// residual add reads, lowered whole (mixFFN) over the normed row
			// and its quantize.
			last := len(g.Ops) - 1
			fin := g.Ops[last]
			if fin.Kind != graph.OpAdd || len(fin.Out) != 1 || fin.Out[0] != g.Out[0] {
				return nil, "a mixture not ended by the block's residual add"
			}
			mixOut := fin.In[1]
			if _, ok := vals[fin.In[0]]; !ok {
				mixOut = fin.In[0]
			}
			src := make([]int, 0, last-i)
			for k := i; k < last; k++ {
				src = append(src, k)
			}
			q := operand(in(0), i)
			out := prog.val(g.Values[mixOut].Name, f32(ne), slotMix)
			prog.ops = append(prog.ops, devOp{Kind: dkMixture, Src: src, In: []devVal{in(0), q}, Out: []devVal{out}})
			vals[mixOut] = out
			skipTo = last
		case graph.OpClamp:
			var cs devSlot
			switch prog.slots[in(0)] {
			case slotQ:
				cs = slotQC
			case slotK:
				cs = slotKC
			case slotV:
				cs = slotVC
			default:
				return nil, "a clamp of neither q, k nor v"
			}
			out := prog.val(name(0), prog.vals[in(0)], cs)
			prog.ops = append(prog.ops, devOp{Kind: dkClamp, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpScale:
			// The layer's output scale over the block's row.
			if len(o.Weight) != 1 || o.Weight[0].Role != jlm.RoleLayerOutScale {
				return nil, "a scale the program does not lower"
			}
			out := prog.val(name(0), f32(ne), slotX)
			prog.ops = append(prog.ops, devOp{Kind: dkScale, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Staged: true})
			vals[o.Out[0]] = out
		case graph.OpSlice:
			// A gated query's split: two slices of q's projection, its heads
			// and its gate, lowered as one launch at the first.
			if prog.slots[in(0)] != slotQ || o.Attr.HeadDim != 2*p.HeadDim || o.Attr.Width != p.HeadDim {
				return nil, "a slice the program does not lower"
			}
			if o.Attr.Offset != 0 {
				continue // the second half, written with the first
			}
			gate := graph.ValueID(-1)
			for k := i + 1; k < len(g.Ops); k++ {
				if n := g.Ops[k]; n.Kind == graph.OpSlice && n.In[0] == o.In[0] && n.Attr.Offset == p.HeadDim {
					gate = n.Out[0]
				}
			}
			if gate < 0 {
				return nil, "a gated query without its gate"
			}
			qh := prog.val(name(0), f32(qd), slotQG)
			gv := prog.val(g.Values[gate].Name, f32(qd), slotOGate)
			prog.ops = append(prog.ops, devOp{Kind: dkSplitQG, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{qh, gv}})
			vals[o.Out[0]], vals[gate] = qh, gv
		case graph.OpHeadNorm:
			var role jlm.Role
			if len(o.Weight) > 0 {
				role = o.Weight[0].Role
			}
			src := in(0)
			switch {
			case role == 0 && o.In[0] == kRot:
				// A weightless (L2) norm of k after its rotary: a rotation
				// keeps a head's norm, so it norms k before, and the rotary
				// still writes the history (the norm then the rotation).
				out := prog.val(name(0), prog.vals[src], slotKN)
				prog.ops = append(prog.ops, devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{src}, Out: []devVal{out}})
				kRot = o.Out[0]
				vals[o.Out[0]] = out
				continue
			case role == 0 && prog.slots[src] == slotQR:
				// The same for q: its rotary's input is normed first.
				r := len(prog.ops) - 1
				for r >= 0 && prog.ops[r].Kind != dkRopeQ {
					r--
				}
				if r < 0 || prog.ops[r].Out[0] != src {
					return nil, "an L2 norm of q not after its rotary"
				}
				qn := prog.val(name(0)+".pre", prog.vals[prog.ops[r].In[0]], slotQN)
				norm := devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{prog.ops[r].In[0]}, Out: []devVal{qn}}
				prog.ops = append(prog.ops[:r], append([]devOp{norm}, prog.ops[r:]...)...)
				prog.ops[r+1].In[0] = qn
				vals[o.Out[0]] = src
				continue
			}
			s := slotQN
			switch prog.slots[src] {
			case slotK, slotV, slotKC, slotVC:
				// k's norm; or v's (weightless, on k's heads), which reads
				// the projection k's norm does not overwrite where v is k's.
				s = slotKN
				if role == 0 && o.Out[0] != kRot && isValue(g, o.Out[0]) {
					s = slotVN
				}
			case slotQ, slotQG, slotQR, slotQC:
				// q's norm, before its rotary or after it (a weighted norm
				// after the rotary does not commute with it).
			default:
				return nil, "a head norm of neither q nor k"
			}
			out := prog.val(name(0), prog.vals[src], s)
			prog.ops = append(prog.ops, devOp{Kind: dkHeadNorm, Src: []int{i}, In: []devVal{src}, Out: []devVal{out}, Role: role})
			vals[o.Out[0]] = out
		case graph.OpRope:
			if s := prog.slots[in(0)]; s == slotQ || s == slotQG || s == slotQN || s == slotQC {
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
				// A layer without a rotary (NoPE): k and v copied in.
				hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: kv}, slotKV)
				prog.ops = append(prog.ops, devOp{Kind: dkKVCopy, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}})
				continue
			}
			hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: kv}, slotKV)
			prog.ops = append(prog.ops, devOp{Kind: dkKVWrite, Src: []int{i - 1, i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}})
		case graph.OpAttend:
			q := in(0)
			if o.Attr.TempScale != 0 {
				// The query's temperature, a scalar a row: it commutes with
				// the rotary, so it scales whichever q the attention reads.
				ts := slotQR
				if prog.slots[q] == slotQR {
					ts = slotQT
				}
				qt := prog.val(prog.names[q]+".temp", prog.vals[q], ts)
				prog.ops = append(prog.ops, devOp{Kind: dkQTemp, Src: []int{i}, In: []devVal{q}, Out: []devVal{qt}})
				q = qt
			}
			out := prog.val(name(0), f32(qd), slotXB)
			prog.ops = append(prog.ops, devOp{Kind: dkAttend, Src: []int{i}, In: []devVal{q}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpAdd:
			// The add that is the block's output writes its row; another
			// (the attention's residual, a mixture's sum) the residual's
			// second slot.
			home := len(g.Out) == 1 && o.Out[0] == g.Out[0]
			reads := func(sl devSlot) bool { return prog.slots[in(0)] == sl || prog.slots[in(1)] == sl }
			if !reads(slotX) && !reads(slotX2) {
				// A sum of two branches (an attention's and a mixer's beside
				// it), not a residual add: into bs.h, as the encoder sums.
				out := prog.val(name(0), f32(ne), slotH)
				prog.ops = append(prog.ops, devOp{Kind: dkAdd, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			s := slotX2
			switch {
			case home && !reads(slotX):
				s = slotX
			case !home && reads(slotX2):
				// An add a later op ends the block after (a per-layer
				// embedding, a scale) writes the slot it does not read.
				s = slotX
			}
			out := prog.val(name(0), f32(ne), s)
			prog.ops = append(prog.ops, devOp{Kind: dkAdd, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
			if home && s != slotX {
				// The block's only add reads its row: the sum goes to x2 and
				// is moved home, as no launch reads what it writes (RULE 13).
				row := prog.val(name(0)+".home", f32(ne), slotX)
				prog.ops = append(prog.ops, devOp{Kind: dkCopy, Src: []int{i}, In: []devVal{out}, Out: []devVal{row}})
				out = row
			}
			vals[o.Out[0]] = out
			adds++
		case graph.OpAct:
			// An ungated FFN: the activation of up's row alone.
			out := prog.val(name(0), f32(prog.vals[in(0)].Width), slotAct)
			prog.ops = append(prog.ops, devOp{Kind: dkAct, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpActMul:
			if o.Attr.Act == graph.ActSigmoid && prog.slots[in(0)] == slotOGate {
				// The attention's output gate: sigmoid(gate) * attention.
				out := prog.val(name(0), f32(qd), slotOGated)
				prog.ops = append(prog.ops, devOp{Kind: dkGateOut, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			out := prog.val(name(0), f32(nf), slotAct)
			prog.ops = append(prog.ops, devOp{Kind: dkActMul, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		}
	}
	// Pre-norms, post-norms or both (OLMo 2 has no pre-norm: its sublayers
	// read the residual), one residual add a sublayer (a recurrent mixer
	// alone has one), ending in the block's row.
	if adds == 0 || adds > 2 || len(g.Out) != 1 || prog.slots[vals[g.Out[0]]] != slotX {
		return nil, "not a block of one or two residual adds"
	}
	for _, o := range prog.ops {
		switch o.Kind {
		case dkAttend:
			prog.attends = true
		case dkRopeQ, dkKVWrite:
			prog.ropes = true
		}
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
	var out []string
	for _, f := range [3]progFused{l.progOne, l.progRows, l.progRag} {
		if f.prog != nil {
			head := "rewrites: " + f.took.names(d.rewrites)
			if f.err != nil {
				head += " (refused: " + f.err.Error() + ")"
			}
			out = append(out, head+"\n"+f.prog.String())
		}
	}
	if l.progRun != progRanProgram {
		why = l.progRun.String()
	}
	if l.progRunRows != progRanProgram {
		why += "; rows: " + l.progRunRows.String()
	}
	if out != nil {
		return strings.Join(out, ""), why
	}
	return l.prog.String(), why
}

// progReason is why a step ran a block through the encoder rather than its
// program: a byte the step writes, named only when reported.
type progReason uint8

const (
	progRanProgram     progReason = iota
	progNotLowered                // the program does not lower the block (progWhy)
	progNoProgram                 // Config.NoProgram
	progRowsNotPrefill            // rows that are not a prefill chunk
	progFault                     // a fault skips a kernel
	progNotPaged                  // not a paged decode or prefill
	progRotary                    // a rotary the program does not lower
	progNormResid                 // a norm or residual the program does not lower
	progTableSplit                // TableSplit
	numProgReasons
)

func (r progReason) String() string {
	switch r {
	case progNotLowered:
		return "not lowered"
	case progNoProgram:
		return "NoProgram"
	case progRowsNotPrefill:
		return "rows that are not a prefill chunk"
	case progFault:
		return "a fault skips a kernel"
	case progNotPaged:
		return "not a paged decode or prefill"
	case progRotary:
		return "a rotary the program does not lower"
	case progNormResid:
		return "a norm or residual the program does not lower"
	case progTableSplit:
		return "TableSplit"
	}
	return ""
}

// bindsRole reports that some op of g reads a weight of role r.
func bindsRole(g *graph.Graph, r jlm.Role) bool {
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if w.Role == r {
				return true
			}
		}
	}
	return false
}

// isValue reports that v is the value a history append stores (its second
// input): a norm writing it is v's.
func isValue(g *graph.Graph, v graph.ValueID) bool {
	for _, o := range g.Ops {
		if o.Kind == graph.OpKVAppend && len(o.In) > 1 && o.In[1] == v {
			return true
		}
	}
	return false
}

// readsRoleOut reports that op o reads the output of an op binding role r.
func readsRoleOut(g *graph.Graph, o graph.Op, r jlm.Role) bool {
	for _, p := range g.Ops {
		if len(p.Weight) == 0 || p.Weight[0].Role != r {
			continue
		}
		for _, x := range o.In {
			for _, y := range p.Out {
				if x == y {
					return true
				}
			}
		}
	}
	return false
}

// hasOp reports that g has an op of kind k.
func hasOp(g *graph.Graph, k graph.OpKind) bool {
	for _, o := range g.Ops {
		if o.Kind == k {
			return true
		}
	}
	return false
}
