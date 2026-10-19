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
	// elemU32 is an index: a router's selection of experts.
	elemU32
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
	e := [...]string{"f32", "q8", "f16", "u32"}[t.Elem]
	l := [...]string{"row", "kv", "tok", "frag"}[t.Layout]
	return fmt.Sprintf("%s.%s[%d]", e, l, t.Width)
}

// devKind is a device op's kind.
type devKind uint8

const (
	dkNorm          devKind = iota // RMSNorm of a row by a weight
	dkQuant                        // a float row to the int8 operand a matvec reads
	dkMatVec                       // an integer matvec over a weight
	dkRopeQ                        // q's rotary
	dkKVWrite                      // k's rotary into the history, and v's copy
	dkAttend                       // the paged decode attention
	dkAdd                          // a residual add
	dkActMul                       // act(gate) * up
	dkHeadNorm                     // each head's RMSNorm (q's or k's)
	dkPostNorm                     // a sublayer's output normed before its residual add (post-norm)
	dkRecurrent                    // a recurrent mixer with its pre-norm, from the block's row (emitLinear)
	dkCopy                         // a row moved to another slot (the sum home to the block's row)
	dkKVCopy                       // an unrotated key and its value copied into the history (NoPE)
	dkAct                          // an ungated FFN's activation of up's row
	dkQTemp                        // the query's temperature, a factor of each row's position
	dkSplitQG                      // a double-width query split into its heads and its output gate
	dkGateOut                      // the attention's output times the sigmoid of the gate
	dkPLESlice                     // this block's rows of the per-layer input table
	dkScale                        // the block's row times the layer's output scale
	dkClamp                        // q, k or v clamped to the plan's bound
	dkSubNorm                      // an inner norm before wo or down, with the quantize it feeds
	dkRouterLogits                 // a router's logits over the normed row (the router read as floats)
	dkRouterBias                   // a bias added to the router's logits before the choice
	dkGroupMask                    // the experts of the best-scoring groups kept, the rest masked
	dkTopK                         // the top k of the logits (or a masked or biased plane): the selection
	dkRouteWeights                 // the selection's weights: softmax, sigmoid or renormalised, per the plan
	dkExpertMatVec                 // a matvec per selected expert over a bank, indexed by the selection
	dkExpertBias                   // each selected expert's bias, indexed by its id
	dkScaleWeights                 // each routed weight times its expert's own factor
	dkCombine                      // the selection's weighted sum of the experts' outputs
	dkGateLogit                    // a shared expert's gate: one logit over the normed row
	dkGatedAdd                     // a branch added in, scaled by the sigmoid of its gate's logit
	dkExpertFetch                  // a paged bank's selected sheets brought to the card, and the index its matvecs read
	dkGroupByExpert                // over rows, the selection sorted by expert: the order the grouped matvecs read
	dkSlice                        // a strided copy: each head's slice, or a row's tail
	dkView                         // a row's prefix read in place (no launch)
	dkRopeTail                     // the rotary over each head's tail, compacted
	dkLatentWrite                  // a latent row and its rotated key into this position's paged row
	dkConcatHeads                  // per-head parts assembled into one query row
	dkLatentAttend                 // attention over latent rows: the key the row, the value its prefix
	dkExpertMatVecE                // an expert matvec with its epilogue: its bias, or act(gate)*up (the bank's fused kernel)
	dkConv1d                       // a causal convolution over the session's convolution state, and its shift
	dkScan                         // a recurrence's state update over the session's state, by the plan's rule
	dkGatedNorm                    // a recurrence's output: each head normed, gated by z
	dkMul                          // an elementwise product of two rows
	dkConvBias                     // the convolved row plus its bias, activated (Mamba-1's)
	dkSSMNorm                      // a selective scan's dt, B or C row normed (Mamba-1's)
	numBaseKinds                   // the fused kinds (devrewrite.go) count on from here
)

// devVal names a value of the program; its buffer is the scratch slot the
// lowering assigns (devSlot).
type devVal int

// devSlot is the block scratch buffer a value lives in.
type devSlot uint8

const (
	slotX        devSlot = iota // bs.x: the block's input and output
	slotX2                      // bs.x2: the residual after attention
	slotH                       // bs.h: a normed row
	slotQ8                      // bs.a / bs.ax: the int8 operand
	slotQ                       // bs.q
	slotK                       // bs.k
	slotV                       // bs.v
	slotQR                      // bs.qr: q rotated
	slotKV                      // the history
	slotXB                      // bs.xb: the attention's output
	slotOut                     // bs.mvOut: a projection's output
	slotG                       // bs.g: the gate
	slotU                       // bs.u: up
	slotAct                     // bs.act: act(gate)*up
	slotQN                      // bs.qn: q's heads normed
	slotKN                      // bs.kn: k's heads normed
	slotF16                     // g.f16Buf: a GEMM's binary16 operand
	slotQT                      // bs.qt: a rotated query scaled by its temperature
	slotQG                      // bs.qg: the query's heads, split from a gated query
	slotOGate                   // bs.ogate: the output gate
	slotOGated                  // bs.ogated: the attention's output, gated
	slotVN                      // bs.vn: v's heads normed
	slotQC                      // bs.qClamp
	slotKC                      // bs.kClamp
	slotVC                      // bs.vClamp
	slotSub                     // bs.sub: an inner norm's row
	slotSide                    // bs.side: a recurrent mixer's output beside the attention
	slotRLogits                 // bs.rlogits: a router's logits
	slotRLogitsB                // bs.rlogitsB: the logits biased
	slotRBM                     // bs.rbm: the group-masked selection plane
	slotSel                     // bs.rsel: the selection
	slotTop                     // bs.rtop: the selection's logits
	slotRW                      // bs.rw: the routed weights
	slotRWS                     // bs.rwS: the routed weights times the experts' factors
	slotEG                      // bs.eg: the experts' gate rows
	slotEU                      // bs.eu: the experts' up rows
	slotEGB                     // bs.egB: the gate rows biased
	slotEUB                     // bs.euB: the up rows biased
	slotEAct                    // bs.eact: the experts' activations
	slotEDown                   // bs.edown: the experts' down rows
	slotEDownB                  // bs.edownB: the down rows biased
	slotShG                     // bs.shg: a shared expert's gate row
	slotShU                     // bs.shu: its up row
	slotShAct                   // bs.shact: its activation
	slotShOut                   // bs.shout: its output
	slotShLogit                 // bs.shlogit: its gate's logit
	slotShSum                   // bs.shsum: the routed and shared sum
	slotSelW                    // the index the expert matvecs read: the selection, its cache slots, or the identity
	slotOnes                    // bs.wOnes: unit weights (a combine of experts weighted at their input)
	slotMQA                     // bs.mlaQA: the query latent
	slotMQAn                    // bs.mlaQAn: it normed
	slotMKV                     // bs.mlaKV: the kv latent and its rotary key
	slotMKVn                    // bs.mlaKVn: the latent normed
	slotMNope                   // bs.mlaNope: the query's unrotated parts, compacted
	slotMPE                     // bs.mlaPE: its rotary parts, compacted
	slotMPErot                  // bs.mlaPErot: them rotated
	slotMKPE                    // bs.mlaKPE: the shared rotary key
	slotMAbsRaw                 // bs.mlaAbsRaw: the absorbed query, head-major
	slotMAbs                    // bs.mlaAbs: the query row the latent attention reads
	slotMAcc                    // bs.mlaAcc: the attention over the latent
	slotMOut                    // bs.mlaOut: it un-absorbed
	slotMIdent                  // bs.mlaIdent: the identity selection 0..NHead-1
	slotMKVLat                  // the kv latent's prefix: bs.mlaKV read in place, copied out over rows
	slotHR                      // bs.hr: a router's own norm of the row
	slotDMixed                  // bs.dMixed: a recurrence's in-projection
	slotDZ                      // bs.dZ: its output gate
	slotDBA                     // bs.dBA: its b and a streams (the delta rule's)
	slotDBRaw                   // bs.dBRaw: its dt stream (Mamba-2's)
	slotDConv                   // bs.dConv: the in-projection convolved
	slotDOut                    // bs.dOut: the recurrence's output
	slotDAct                    // bs.dAct: it normed and gated
	slotScB                     // bs.scB: a short convolution's input gate
	slotScC                     // bs.scC: its output gate
	slotScBx                    // bs.scBx: the gated input
	slotScY                     // bs.scY: the gated output
	slotLRQ8                    // bs.dLRa, bs.dLRax: the low-rank projections' own int8 operand
	slotDLow                    // bs.dLowRank: a low-rank projection's rank-wide middle
	slotDAlpha                  // bs.dAlpha: the per-channel decay's logits (KDA's)
	slotM1B                     // bs.m1B: a selective scan's B
	slotM1C                     // bs.m1C: its C
	slotM1LowN                  // bs.m1LowN: its dt row normed
	slotM1BN                    // bs.m1BN: B normed
	slotM1CN                    // bs.m1CN: C normed
	slotPLEG                    // bs.pleG: the per-layer embedding's gate
	slotPLES                    // bs.pleS: this block's per-layer input
	slotPLEA                    // bs.pleA: the gate activated, times the input
	slotPLEO                    // bs.pleO: projected back to the row
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
	case slotRLogits:
		return "rlogits"
	case slotRLogitsB:
		return "rlogits_b"
	case slotRBM:
		return "rbm"
	case slotSel:
		return "sel"
	case slotTop:
		return "top"
	case slotRW:
		return "rw"
	case slotRWS:
		return "rw_s"
	case slotEG:
		return "eg"
	case slotEU:
		return "eu"
	case slotEGB:
		return "eg_b"
	case slotEUB:
		return "eu_b"
	case slotEAct:
		return "eact"
	case slotEDown:
		return "edown"
	case slotEDownB:
		return "edown_b"
	case slotShG:
		return "shg"
	case slotShU:
		return "shu"
	case slotShAct:
		return "shact"
	case slotShOut:
		return "shout"
	case slotShLogit:
		return "shlogit"
	case slotShSum:
		return "shsum"
	case slotSelW:
		return "sel_w"
	case slotOnes:
		return "ones"
	case slotMQA:
		return "mla_qa"
	case slotMQAn:
		return "mla_qa_n"
	case slotMKV:
		return "mla_kv"
	case slotMKVn:
		return "mla_kv_n"
	case slotMNope:
		return "mla_nope"
	case slotMPE:
		return "mla_pe"
	case slotMPErot:
		return "mla_pe_rot"
	case slotMKPE:
		return "mla_kpe"
	case slotMAbsRaw:
		return "mla_abs_raw"
	case slotMAbs:
		return "mla_abs"
	case slotMAcc:
		return "mla_acc"
	case slotMOut:
		return "mla_out"
	case slotMIdent:
		return "mla_ident"
	case slotMKVLat:
		return "mla_kv_lat"
	case slotHR:
		return "hr"
	case slotDMixed:
		return "d_mixed"
	case slotDZ:
		return "d_z"
	case slotDBA:
		return "d_ba"
	case slotDBRaw:
		return "d_braw"
	case slotDConv:
		return "d_conv"
	case slotDOut:
		return "d_out"
	case slotDAct:
		return "d_act"
	case slotScB:
		return "sc_b"
	case slotScC:
		return "sc_c"
	case slotScBx:
		return "sc_bx"
	case slotScY:
		return "sc_y"
	case slotLRQ8:
		return "lr_q8"
	case slotDLow:
		return "d_low"
	case slotDAlpha:
		return "d_alpha"
	case slotM1B:
		return "m1_b"
	case slotM1C:
		return "m1_c"
	case slotM1LowN:
		return "m1_low_n"
	case slotM1BN:
		return "m1_b_n"
	case slotM1CN:
		return "m1_c_n"
	case slotPLEG:
		return "ple_g"
	case slotPLES:
		return "ple_s"
	case slotPLEA:
		return "ple_a"
	case slotPLEO:
		return "ple_o"
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
	// W is a matvec's weight as the block holds it: its index in the
	// block's matrix list (layer.rs, layer.ms), resolved from Role once
	// when the program is built.
	W int
	// V and VB are a norm's weight and bias on the device, bound by role
	// when the program is built (layer.vecOf); a fused op carries the norm
	// it absorbed. nil for a weightless norm.
	V, VB backend.Buf
	// OneRow says the op runs for one row only: over rows, what reads it
	// produces it itself (an expert matvec's quantize).
	OneRow bool
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
	// experts says the program runs routed experts, which it lowers for one
	// row: rows take the encoder until the grouped kernels are lowered.
	experts bool
	// streamed says an expert bank is paged in per token (dkExpertFetch).
	streamed bool
	// scans says the program steps a recurrence: its state's parity flips
	// once the block's launches are queued (devTier.stepFlip).
	scans bool
	// latent says the program attends over latent rows, which it lowers for
	// one row: rows take the encoder until latent prefill is lowered.
	latent bool
	vals   []devType
	slots  []devSlot
	names  []string
	ops    []devOp
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
	case dkRecurrent, dkCopy, dkKVCopy, dkAct, dkQTemp, dkSplitQG, dkGateOut, dkPLESlice, dkScale, dkClamp, dkSubNorm:
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkExpertMatVec:
		// The activation, int8; the selection, indices; the float rows the
		// grouped form sorts and quantizes itself.
		switch i {
		case 1:
			return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
		case 2:
			return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
	case dkExpertMatVecE:
		switch i {
		case 0:
			return devType{Elem: elemQ8, Layout: layoutRow, Width: in.Width}
		case 1:
			return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkExpertBias, dkScaleWeights:
		if i == 1 {
			return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
		}
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkSlice, dkView, dkRopeTail, dkLatentWrite, dkConcatHeads, dkLatentAttend, dkConv1d, dkScan, dkGatedNorm, dkMul, dkConvBias, dkSSMNorm:
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkExpertFetch, dkGroupByExpert:
		return devType{Elem: elemU32, Layout: layoutRow, Width: in.Width}
	case dkRouterLogits, dkRouterBias, dkGroupMask, dkTopK, dkRouteWeights, dkCombine, dkGateLogit, dkGatedAdd:
		return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width}
	case dkMatVec, dkQKV, dkGate2:
		if i > 0 {
			return devType{Elem: elemF32, Layout: layoutRow, Width: in.Width} // a low-rank operand's float source
		}
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
	// Every value an op reads is written by an earlier op, is the block's
	// input or is a constant (constSlot): a rewrite that moves a launch above
	// what it reads is refused.
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
			if constSlot(p.slots[x]) {
				continue
			}
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
		// The mixture's ops.
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
	ne, qd, kv := p.NEmbd, p.NHead*p.HeadDim, p.KVRow()
	f32 := func(w int) devType { return devType{Elem: elemF32, Layout: layoutRow, Width: w} }
	vals := map[graph.ValueID]devVal{g.In[0]: prog.val("x", f32(ne), slotX)}
	if len(g.In) > 1 && p.PLEDim != 0 {
		// The per-layer input: this block's rows of the model's table.
		pl := prog.val(g.Values[g.In[1]].Name, f32(p.PLEDim), slotPLES)
		prog.ops = append(prog.ops, devOp{Kind: dkPLESlice, Out: []devVal{pl}})
		vals[g.In[1]] = pl
	}
	// The int8 operand of a matvec: the float row it reads, quantized once
	// per source (q, k and v read one quantize of the normed row).
	//
	// One int8 operand buffer: the last quantize wins, so the cache is the
	// last source, its quantize and that op. A quantize only expert matvecs
	// read is one row's (OneRow): over rows they quantize their own sorted
	// operand.
	lastSrc, lastQ, lastOp := devVal(-1), devVal(-1), -1
	operand := func(src devVal, op int, expert bool) devVal {
		if src == lastSrc {
			if !expert && lastOp >= 0 {
				prog.ops[lastOp].OneRow = false
			}
			return lastQ
		}
		q := prog.val(prog.names[src]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[src].Width}, slotQ8)
		prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{op}, In: []devVal{src}, Out: []devVal{q}, OneRow: expert})
		lastSrc, lastQ, lastOp = src, q, len(prog.ops)-1
		return q
	}
	matvecSlot := map[jlm.Role]devSlot{
		jlm.RoleAttnQ: slotQ, jlm.RoleAttnK: slotK, jlm.RoleAttnV: slotV, jlm.RoleAttnOut: slotOut,
		jlm.RoleFFNGate: slotG, jlm.RoleFFNUp: slotU, jlm.RoleFFNDown: slotOut,
		jlm.RoleShExpGate: slotShG, jlm.RoleShExpUp: slotShU, jlm.RoleShExpDown: slotShOut,
		jlm.RoleAttnQA: slotMQA, jlm.RoleAttnQB: slotQ, jlm.RoleAttnKVA: slotMKV,
		jlm.RolePLEGate: slotPLEG, jlm.RolePLEProj: slotPLEO,
	}
	adds := 0
	// routeW is each selection's routed weights; gated is a shared expert's
	// output (by graph value) and its gate's logit.
	routeW := map[devVal]devVal{}
	// weightSel is a paged bank's weight-side index for a selection, and
	// weightIn marks a selection whose weights scale the experts' input.
	weightSel := map[devVal]devVal{}
	weightIn := map[devVal]bool{}
	gated := map[graph.ValueID]devVal{}
	var kRot graph.ValueID = -1
	mla := false
	for _, o := range g.Ops {
		if o.Kind == graph.OpAttendLatent {
			mla = true
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
				lastSrc, lastQ, lastOp = sub, q, -1
				vals[o.Out[0]] = sub
				continue
			case jlm.RoleAttnNorm, jlm.RoleFFNNorm:
			case jlm.RolePostAttnNorm, jlm.RolePostFFNNorm:
				k = dkPostNorm
			case jlm.RolePostFFNNorm1, jlm.RolePostFFNNorm2, jlm.RoleRouterNorm, jlm.RoleFFNNorm2, jlm.RolePLEPostNorm:
				// A mixture beside a dense MLP: each branch's norms, out of
				// place, into the buffers the branches read (the router's
				// own row, the experts', and each branch's output).
				ns := slotH
				switch o.Weight[0].Role {
				case jlm.RolePostFFNNorm1:
					ns = slotShSum
				case jlm.RoleRouterNorm:
					ns = slotHR
				}
				out := prog.val(name(0), prog.vals[in(0)], ns)
				prog.ops = append(prog.ops, devOp{Kind: dkPostNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
				vals[o.Out[0]] = out
				continue
			case jlm.RoleAttnQANorm, jlm.RoleAttnKVANorm:
				// A latent's norm, at the latent's width, into its own buffer.
				ls := slotMQAn
				if o.Weight[0].Role == jlm.RoleAttnKVANorm {
					ls = slotMKVn
				}
				out := prog.val(name(0), prog.vals[in(0)], ls)
				prog.ops = append(prog.ops, devOp{Kind: dkNorm, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Role: o.Weight[0].Role})
				vals[o.Out[0]] = out
				continue
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
			s, ok := matvecSlot[role]
			if !ok {
				return nil, "a matvec over " + role.String()
			}
			w := o.Weight[0].Rows
			if w == 0 {
				return nil, "a matvec with no stated width"
			}
			wi := l.matIndex(role)
			if wi < 0 {
				return nil, "a matvec over " + role.String() + ", which the block's device copy does not hold"
			}
			a := operand(in(0), i, false)
			out := prog.val(name(0), f32(w), s)
			prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{a}, Out: []devVal{out}, Role: role, W: wi})
			vals[o.Out[0]] = out
		case graph.OpRecurrent:
			if rr := p.Recurrent; prog.slots[in(0)] == slotH && rr.ShortConv && !l.centred && l.k3 == nil && l.q4e == nil {
				// A short convolution (LFM2's): x, B and C projected, B*x
				// convolved over the session's state, gated by C, projected
				// out.
				for _, w := range [...]int{posSSMInProj, posSSMGate, posSSMBA, posSSMOut} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a short convolution whose projections the block's device copy does not hold"
					}
				}
				ch := f32(rr.Chans)
				q := operand(in(0), i, false)
				mv := func(nm string, w int, s devSlot) devVal {
					v := prog.val(name(0)+nm, ch, s)
					prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{v}, W: w})
					return v
				}
				x, b, c := mv(".x", posSSMInProj, slotDMixed), mv(".b", posSSMGate, slotScB), mv(".c", posSSMBA, slotScC)
				bx := prog.val(name(0)+".bx", ch, slotScBx)
				prog.ops = append(prog.ops, devOp{Kind: dkMul, Src: []int{i}, In: []devVal{b, x}, Out: []devVal{bx}})
				conv := prog.val(name(0)+".conv", ch, slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{bx}, Out: []devVal{conv}})
				y := prog.val(name(0)+".y", ch, slotScY)
				prog.ops = append(prog.ops, devOp{Kind: dkMul, Src: []int{i}, In: []devVal{c, conv}, Out: []devVal{y}})
				qy := operand(y, i, false)
				out := prog.val(name(0), f32(ne), slotOut)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qy}, Out: []devVal{out}, W: posSSMOut})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			if rr := p.Recurrent; prog.slots[in(0)] == slotH && rr.Mamba1 && !l.centred && l.k3 == nil && l.q4e == nil {
				// Mamba-1's selective scan: x and z projected, x convolved
				// over the session's state and biased, dt, B and C off x (dt
				// through its own low-rank operand), each optionally normed,
				// the scan, y*silu(z), projected out.
				for _, w := range [...]int{posSSMInProj, posSSMGate, posSSMOut, posLowA, posLowB, posGateA, posGateB} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a selective scan whose projections the block's device copy does not hold"
					}
				}
				inner := rr.VHeads * rr.VDim
				mv := func(nm string, q devVal, w int, s devSlot, width int, extra ...devVal) devVal {
					v := prog.val(name(0)+nm, f32(width), s)
					prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: append([]devVal{q}, extra...), Out: []devVal{v}, W: w})
					return v
				}
				q := operand(in(0), i, false)
				x, z := mv(".x", q, posSSMInProj, slotDMixed, rr.Chans), mv(".z", q, posSSMGate, slotDZ, inner)
				conv := prog.val(name(0)+".conv", f32(rr.Chans), slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{x}, Out: []devVal{conv}})
				xa := prog.val(name(0)+".xa", f32(rr.Chans), slotDMixed)
				prog.ops = append(prog.ops, devOp{Kind: dkConvBias, Src: []int{i}, In: []devVal{conv}, Out: []devVal{xa}})
				qx := operand(xa, i, false)
				low := mv(".dt_low", qx, posLowA, slotDLow, l.ms[posLowA].rows)
				b, c := mv(".b", qx, posGateA, slotM1B, l.ms[posGateA].rows), mv(".c", qx, posGateB, slotM1C, l.ms[posGateB].rows)
				if l.m1DtN != nil {
					norm := func(v devVal, s devSlot) devVal {
						n := prog.val(prog.names[v]+".n", prog.vals[v], s)
						prog.ops = append(prog.ops, devOp{Kind: dkSSMNorm, Src: []int{i}, In: []devVal{v}, Out: []devVal{n}})
						return n
					}
					low, b, c = norm(low, slotM1LowN), norm(b, slotM1BN), norm(c, slotM1CN)
				}
				lq := prog.val(prog.names[low]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[low].Width}, slotLRQ8)
				prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{low}, Out: []devVal{lq}})
				dt := mv(".dt", lq, posLowB, slotDBRaw, l.ms[posLowB].rows, low)
				st := prog.val(name(0)+".out", f32(inner), slotDOut)
				prog.ops = append(prog.ops, devOp{Kind: dkScan, Src: []int{i}, In: []devVal{xa, b, c, dt}, Out: []devVal{st}})
				act := prog.val(name(0)+".act", f32(inner), slotDAct)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedNorm, Src: []int{i}, In: []devVal{st, z}, Out: []devVal{act}, Staged: true})
				qa := operand(act, i, false)
				out := prog.val(name(0), f32(ne), slotOut)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qa}, Out: []devVal{out}, W: posSSMOut})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			if rr := p.Recurrent; prog.slots[in(0)] == slotH && rr.ChanDecay && !l.centred && l.k3 == nil && l.q4e == nil {
				// Kimi Delta Attention: the fused in-projection, the output
				// and decay gates each through a low-rank pair (the second
				// matvec reading its own int8 operand), beta's logits, the
				// convolution and the per-channel scan over the session's
				// state, the sigmoid-gated norm, projected out.
				for _, w := range [...]int{posSSMInProj, posSSMBA, posSSMOut, posLowA, posLowB, posGateA, posGateB} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a delta rule whose projections the block's device copy does not hold"
					}
				}
				inner := rr.VHeads * rr.VDim
				q := operand(in(0), i, false)
				mv := func(nm string, q devVal, w int, s devSlot, width int, extra ...devVal) devVal {
					v := prog.val(name(0)+nm, f32(width), s)
					prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: append([]devVal{q}, extra...), Out: []devVal{v}, W: w})
					return v
				}
				lowRank := func(nm string, wa, wb int, s devSlot, width int) devVal {
					low := mv(nm+"_low", q, wa, slotDLow, l.ms[wa].rows)
					lq := prog.val(prog.names[low]+".q8", devType{Elem: elemQ8, Layout: layoutRow, Width: prog.vals[low].Width}, slotLRQ8)
					prog.ops = append(prog.ops, devOp{Kind: dkQuant, Src: []int{i}, In: []devVal{low}, Out: []devVal{lq}})
					return mv(nm, lq, wb, s, width, low)
				}
				mixed := mv(".mixed", q, posSSMInProj, slotDMixed, rr.Chans)
				z := lowRank(".z", posGateA, posGateB, slotDZ, inner)
				alpha := lowRank(".alpha", posLowA, posLowB, slotDAlpha, l.ms[posLowB].rows)
				beta := mv(".beta", q, posSSMBA, slotDBRaw, l.ms[posSSMBA].rows)
				conv := prog.val(name(0)+".conv", f32(rr.Chans), slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{mixed}, Out: []devVal{conv}})
				st := prog.val(name(0)+".out", f32(inner), slotDOut)
				prog.ops = append(prog.ops, devOp{Kind: dkScan, Src: []int{i}, In: []devVal{conv, alpha, beta}, Out: []devVal{st}})
				act := prog.val(name(0)+".act", f32(inner), slotDAct)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedNorm, Src: []int{i}, In: []devVal{st, z}, Out: []devVal{act}, Staged: true})
				qa := operand(act, i, false)
				out := prog.val(name(0), f32(ne), slotOut)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qa}, Out: []devVal{out}, W: posSSMOut})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
			if rr := p.Recurrent; prog.slots[in(0)] == slotH && !l.centred && l.k3 == nil && l.q4e == nil &&
				!rr.ShortConv && !rr.Mamba1 && !rr.ChanDecay {
				// The delta rule or Mamba-2's SSD as primitives: the
				// projections by role (a mixer beside an attention has its own
				// in-projection), the convolution and the scan over the
				// session's state, the gated norm, the out projection.
				// The tier's own slots for a recurrence's four projections
				// (layer.tensors' order), until the positional list goes.
				inW, zW, baW, oW, outSlot := posSSMInProj, posSSMGate, posSSMBA, posSSMOut, slotOut
				if rr.WithAttn {
					inW, outSlot = posSSMIn, slotSide
				}
				for _, w := range [...]int{inW, zW, baW, oW} {
					if w >= len(l.rs) || *l.rs[w] == nil {
						return nil, "a recurrence whose projections the block's device copy does not hold"
					}
				}
				inner := rr.VHeads * rr.VDim
				q := operand(in(0), i, false)
				mixed := prog.val(name(0)+".mixed", f32(rr.Chans), slotDMixed)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{mixed}, W: inW})
				z := prog.val(name(0)+".z", f32(inner), slotDZ)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{z}, W: zW})
				gs := slotDBA
				if rr.SSD {
					gs = slotDBRaw
				}
				gates := prog.val(name(0)+".gates", f32(l.ms[baW].rows), gs)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{q}, Out: []devVal{gates}, W: baW})
				conv := prog.val(name(0)+".conv", f32(rr.Chans), slotDConv)
				prog.ops = append(prog.ops, devOp{Kind: dkConv1d, Src: []int{i}, In: []devVal{mixed}, Out: []devVal{conv}})
				st := prog.val(name(0)+".out", f32(inner), slotDOut)
				prog.ops = append(prog.ops, devOp{Kind: dkScan, Src: []int{i}, In: []devVal{conv, gates}, Out: []devVal{st}})
				act := prog.val(name(0)+".act", f32(inner), slotDAct)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedNorm, Src: []int{i}, In: []devVal{st, z}, Out: []devVal{act}, Staged: true})
				qa := operand(act, i, false)
				out := prog.val(name(0), f32(ne), outSlot)
				prog.ops = append(prog.ops, devOp{Kind: dkMatVec, Src: []int{i}, In: []devVal{qa}, Out: []devVal{out}, W: oW})
				vals[o.Out[0]] = out
				prog.scans = true
				continue
			}
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
			// The router as primitives: logits, a bias, a group mask, the
			// top k and their weights, each from the op's attributes and
			// bindings.
			switch {
			case l.stream != nil && l.stream.hybrid:
				return nil, "experts run on the host beside the card"
			case l.router == nil:
				return nil, "a router the block's device copy does not hold"
			}
			var selBias, hash jlm.Role
			for _, w := range o.Weight[1:] {
				switch w.Role {
				case jlm.RoleExpProbsB:
					selBias = w.Role
				case jlm.RoleHashExperts:
					hash = w.Role
				}
			}
			if hash != 0 {
				return nil, "hash routing"
			}
			ex, k := o.Attr.Experts, o.Attr.TopK
			logits := prog.val(name(0)+".logits", f32(ex), slotRLogits)
			prog.ops = append(prog.ops, devOp{Kind: dkRouterLogits, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{logits}})
			for _, w := range o.Weight[1:] {
				if w.Role != jlm.RoleRouterBias {
					continue
				}
				lb := prog.val(name(0)+".logits_b", f32(ex), slotRLogitsB)
				prog.ops = append(prog.ops, devOp{Kind: dkRouterBias, Src: []int{i}, In: []devVal{logits}, Out: []devVal{lb}, Role: w.Role})
				logits = lb
			}
			rank := []devVal{logits}
			if o.Attr.Groups > 1 {
				plane := prog.val(name(0)+".plane", f32(ex), slotRBM)
				prog.ops = append(prog.ops, devOp{Kind: dkGroupMask, Src: []int{i}, In: []devVal{logits}, Out: []devVal{plane}, Role: selBias})
				rank = append(rank, plane)
			}
			sel := prog.val(name(0), devType{Elem: elemU32, Layout: layoutRow, Width: k}, slotSel)
			top := prog.val(name(0)+".top", f32(k), slotTop)
			prog.ops = append(prog.ops, devOp{Kind: dkTopK, Src: []int{i}, In: rank, Out: []devVal{sel, top}, Role: selBias})
			rw := prog.val(name(0)+".weights", f32(k), slotRW)
			prog.ops = append(prog.ops, devOp{Kind: dkRouteWeights, Src: []int{i}, In: []devVal{top, logits}, Out: []devVal{rw}})
			vals[o.Out[0]] = sel
			routeW[sel] = rw
			prog.ops = append(prog.ops, devOp{Kind: dkGroupByExpert, Src: []int{i}, In: []devVal{sel}})
			if l.stream != nil {
				// A paged bank: the selected sheets come to the card here,
				// and the matvecs index what they landed in.
				sw := prog.val(name(0)+".slots", devType{Elem: elemU32, Layout: layoutRow, Width: k}, slotSelW)
				prog.ops = append(prog.ops, devOp{Kind: dkExpertFetch, Src: []int{i}, In: []devVal{sel}, Out: []devVal{sw}, Staged: true})
				weightSel[sel] = sw
				prog.streamed = true
			}
		case graph.OpGather:
			// One row: the selection indexes the weight side, so the
			// experts read the row itself.
			// Weighted at their input (Llama 4's): the weight scales the
			// activation, and the combine sums at one.
			if o.Attr.WeightIn {
				weightIn[in(1)] = true
			}
			vals[o.Out[0]] = in(0)
		case graph.OpExpertMatVec:
			role := o.Weight[0].Role
			wi := l.matIndex(role)
			if wi < 0 {
				return nil, "an expert bank " + role.String() + ", which the block's device copy does not hold"
			}
			var s, sb devSlot
			switch role {
			case jlm.RoleExpGateBank:
				s, sb = slotEG, slotEGB
			case jlm.RoleExpUpBank:
				s, sb = slotEU, slotEUB
			case jlm.RoleExpDownBank:
				s, sb = slotEDown, slotEDownB
			default:
				return nil, "an expert matvec over " + role.String()
			}
			width := g.Values[o.Out[0]].Width
			a := operand(in(0), i, true)
			out := prog.val(name(0), f32(width), s)
			prog.experts = true
			ix := in(1)
			if w, ok := weightSel[ix]; ok {
				ix = w
			}
			prog.ops = append(prog.ops, devOp{Kind: dkExpertMatVec, Src: []int{i}, In: []devVal{a, ix, in(0)}, Out: []devVal{out}, Role: role, W: wi})
			if len(o.Weight) > 1 {
				b := prog.val(name(0)+".biased", f32(width), sb)
				prog.ops = append(prog.ops, devOp{Kind: dkExpertBias, Src: []int{i}, In: []devVal{out, in(1)}, Out: []devVal{b}, Role: o.Weight[1].Role})
				out = b
			}
			vals[o.Out[0]] = out
		case graph.OpCombine:
			rw, ok := routeW[in(1)]
			if !ok {
				return nil, "a combine of no selection"
			}
			if weightIn[in(1)] {
				rw = prog.val("ones", prog.vals[rw], slotOnes)
			}
			if len(o.Weight) > 0 {
				ws := prog.val(name(0)+".weights", prog.vals[rw], slotRWS)
				prog.ops = append(prog.ops, devOp{Kind: dkScaleWeights, Src: []int{i}, In: []devVal{rw, in(1)}, Out: []devVal{ws}, Role: o.Weight[0].Role})
				rw = ws
			}
			out := prog.val(name(0), f32(g.Values[o.Out[0]].Width), slotOut)
			prog.ops = append(prog.ops, devOp{Kind: dkCombine, Src: []int{i}, In: []devVal{in(0), rw}, Out: []devVal{out}})
			vals[o.Out[0]] = out
		case graph.OpGateScalar:
			// The gate's logit now; the scale rides the add that brings the
			// branch in (dkGatedAdd).
			lg := prog.val(name(0)+".logit", f32(1), slotShLogit)
			prog.ops = append(prog.ops, devOp{Kind: dkGateLogit, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{lg}, Role: o.Weight[0].Role})
			vals[o.Out[0]] = in(1)
			gated[o.Out[0]] = lg
		case graph.OpLatentAppend:
			hist := prog.val("kv", devType{Elem: elemF32, Layout: layoutKV, Width: prog.vals[in(0)].Width + prog.vals[in(1)].Width}, slotKV)
			k := devOp{Kind: dkLatentWrite, Src: []int{i}, In: []devVal{in(0), in(1)}, Out: []devVal{hist}}
			if o.In[1] == kRot {
				k.N = 1 // rotate the key as it is stored
			}
			prog.ops = append(prog.ops, k)
		case graph.OpAttendLatent:
			if len(o.In) > 2 {
				return nil, "an attention indexer"
			}
			kb, vb := l.matIndex(o.Weight[0].Role), l.matIndex(o.Weight[1].Role)
			if kb < 0 || vb < 0 {
				return nil, "a latent attention whose banks the block's device copy does not hold"
			}
			nh := o.Attr.NHead
			lat := o.Weight[0].Rows
			ident := prog.val("ident", devType{Elem: elemU32, Layout: layoutRow, Width: nh}, slotMIdent)
			// Absorb: each head's unrotated query against its sheet of the k
			// up-projection, one selected matvec over the heads.
			qa := operand(in(0), i, false)
			raw := prog.val(name(0)+".abs_raw", f32(nh*lat), slotMAbsRaw)
			prog.ops = append(prog.ops, devOp{Kind: dkExpertMatVec, Src: []int{i}, In: []devVal{qa, ident, in(0)}, Out: []devVal{raw}, Role: o.Weight[0].Role, W: kb})
			abs := prog.val(name(0)+".query", f32(nh*(lat+o.Attr.NRot)), slotMAbs)
			prog.ops = append(prog.ops, devOp{Kind: dkConcatHeads, Src: []int{i}, In: []devVal{raw, in(1)}, Out: []devVal{abs}})
			acc := prog.val(name(0)+".acc", f32(nh*lat), slotMAcc)
			prog.ops = append(prog.ops, devOp{Kind: dkLatentAttend, Src: []int{i}, In: []devVal{abs}, Out: []devVal{acc}})
			// Un-absorb: each head's attention through its sheet of the v
			// up-projection.
			qc := operand(acc, i, false)
			out := prog.val(name(0), f32(g.Values[o.Out[0]].Width), slotMOut)
			prog.ops = append(prog.ops, devOp{Kind: dkExpertMatVec, Src: []int{i}, In: []devVal{qc, ident, acc}, Out: []devVal{out}, Role: o.Weight[1].Role, W: vb})
			vals[o.Out[0]] = out
			prog.latent = true
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
			if mla {
				// Latent attention's slices: each head's unrotated and
				// rotary parts of q, compacted; the kv latent's prefix, read
				// in place; its rotary key.
				src, w := prog.slots[in(0)], g.Values[o.Out[0]].Width
				var k devKind
				var ss devSlot
				switch {
				case src == slotQ && o.Attr.HeadDim == p.HeadDim && o.Attr.Offset == 0:
					k, ss = dkSlice, slotMNope
				case src == slotQ && o.Attr.HeadDim == p.HeadDim:
					k, ss = dkSlice, slotMPE
				case src == slotMKV && o.Attr.HeadDim == 0 && o.Attr.Offset == 0:
					k, ss = dkView, slotMKVLat
				case src == slotMKV && o.Attr.HeadDim == 0:
					k, ss = dkSlice, slotMKPE
				default:
					return nil, "a latent slice the program does not lower"
				}
				out := prog.val(name(0), f32(w), ss)
				prog.ops = append(prog.ops, devOp{Kind: k, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}, Staged: k == dkView})
				vals[o.Out[0]] = out
				continue
			}
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
			switch prog.slots[in(0)] {
			case slotMPE:
				// The query's rotary parts, compacted.
				out := prog.val(name(0), prog.vals[in(0)], slotMPErot)
				prog.ops = append(prog.ops, devOp{Kind: dkRopeTail, Src: []int{i}, In: []devVal{in(0)}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			case slotMKPE:
				// The shared rotary key: rotated as the latent write stores it.
				kRot = o.Out[0]
				vals[o.Out[0]] = in(0)
				continue
			}
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
			if lg, ok := gated[o.In[1]]; ok {
				// The routed sum and a gated shared expert: one launch.
				out := prog.val(name(0), f32(ne), slotShSum)
				prog.ops = append(prog.ops, devOp{Kind: dkGatedAdd, Src: []int{i}, In: []devVal{in(0), in(1), lg}, Out: []devVal{out}})
				vals[o.Out[0]] = out
				continue
			}
			// The add that is the block's output writes its row; another
			// (the attention's residual, a mixture's sum) the residual's
			// second slot.
			home := len(g.Out) == 1 && o.Out[0] == g.Out[0]
			reads := func(sl devSlot) bool { return prog.slots[in(0)] == sl || prog.slots[in(1)] == sl }
			if !reads(slotX) && !reads(slotX2) {
				// A sum of two branches (an attention's and a mixer's beside
				// it, a dense MLP's and a mixture's), not a residual add: into
				// the first of bs.h, bs.mvOut and bs.shsum it does not read.
				bsum := slotH
				for _, c := range [...]devSlot{slotH, slotOut, slotShSum} {
					if !reads(c) {
						bsum = c
						break
					}
				}
				out := prog.val(name(0), f32(ne), bsum)
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
			out := prog.val(name(0), f32(prog.vals[in(0)].Width), actSlot(prog.slots[in(0)]))
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
			out := prog.val(name(0), f32(prog.vals[in(0)].Width), actSlot(prog.slots[in(0)]))
			ins := []devVal{in(0), in(1)}
			if s := actSlot(prog.slots[in(0)]); s == slotEAct {
				for sel := range weightIn {
					ins = append(ins, routeW[sel])
				}
			}
			prog.ops = append(prog.ops, devOp{Kind: dkActMul, Src: []int{i}, In: ins, Out: []devVal{out}})
			vals[o.Out[0]] = out
		}
	}
	// Pre-norms, post-norms or both (OLMo 2 has no pre-norm: its sublayers
	// read the residual), one residual add a sublayer (a recurrent mixer
	// alone has one; a per-layer embedding is a third), ending in the
	// block's row.
	// Every norm binds its weight and bias by role, from the graph op it
	// lowers: the device copy prep uploaded (layer.vecs).
	for i := range prog.ops {
		o := &prog.ops[i]
		switch o.Kind {
		case dkNorm, dkPostNorm, dkHeadNorm, dkSubNorm:
		case dkRouterBias, dkGroupMask, dkTopK, dkExpertBias, dkScaleWeights, dkGateLogit:
			// A vector operand of the mixture's, by its role.
			if o.Role != jlm.RoleNone {
				if o.V = l.vecOf(o.Role); o.V == nil {
					return nil, "a vector " + o.Role.String() + ", which the block's device copy does not hold"
				}
			}
			continue
		default:
			continue
		}
		if o.Role == jlm.RoleNone {
			continue // a weightless norm
		}
		ws := g.Ops[o.Src[0]].Weight
		if o.V = l.vecOf(ws[0].Role); o.V == nil {
			return nil, "a norm over " + ws[0].Role.String() + ", which the block's device copy does not hold"
		}
		if len(ws) > 1 {
			if o.VB = l.vecOf(ws[1].Role); o.VB == nil {
				return nil, "a norm bias " + ws[1].Role.String() + ", which the block's device copy does not hold"
			}
		}
		if g.Ops[o.Src[0]].Kind == graph.OpLayerNorm && o.VB == nil {
			// A LayerNorm kernel always reads a bias, and a file without
			// one gives the block zeros (the model's lnBias): until vectors
			// the model synthesises carry roles, the block's own.
			switch ws[0].Role {
			case jlm.RoleAttnNorm:
				o.VB = l.nAttnB
			case jlm.RoleFFNNorm:
				o.VB = l.nFFNB
			}
			if o.VB == nil {
				return nil, "a layer norm over " + ws[0].Role.String() + " with no bias on the device"
			}
		}
	}
	if adds == 0 || adds > 3 || len(g.Out) != 1 || prog.slots[vals[g.Out[0]]] != slotX {
		return nil, "not a block of one to three residual adds"
	}
	for _, o := range prog.ops {
		switch o.Kind {
		case dkAttend, dkLatentAttend:
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

// compositeKind reports that kind k runs one of the block encoder's
// sequences whole rather than primitives the program compiles: it cannot be
// fused, reordered or re-tiled, so a block holding one does not count as
// compiled (GPU.BlockComposite). The set only shrinks
// (TestCompositeKindsOnlyShrink), to empty at the encoder's deletion.
func compositeKind(k devKind) bool {
	switch k {
	case dkRecurrent:
		return true
	}
	return false
}

// BlockComposite is the first composite op (compositeKind) in block li's
// program, by name, or "" where every op is a primitive or the device holds
// no program for it.
func (g *GPU) BlockComposite(li int) string {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return ""
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	l := d.layers[li]
	if l == nil || l.prog == nil {
		return ""
	}
	for _, o := range l.prog.ops {
		if compositeKind(o.Kind) {
			return o.Kind.String()
		}
	}
	return ""
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
	progRowsExperts               // experts over several rows (grouped expert kernels not yet lowered)
	progStreamTuning              // a paged bank under a probe, prefetch or fixed selection
	progRowsLatent                // latent attention over several rows (latent prefill not yet lowered)
	numProgReasons
)

func (r progReason) String() string {
	switch r {
	case progNotLowered:
		return "not lowered"
	case progNoProgram:
		return "NoProgram"
	case progRowsExperts:
		return "experts over several rows"
	case progRowsLatent:
		return "latent attention over several rows"
	case progStreamTuning:
		return "a paged bank under a probe, prefetch or fixed selection"
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

// matIndex is the index of block l's matrix of role r in its list (layer.rs,
// layer.ms), or -1 where the block holds none.
func (l *layer) matIndex(r jlm.Role) int {
	for _, a := range l.roles {
		if a.role == r {
			return a.i
		}
	}
	return -1
}

// actSlot is where an activation of a row in slot s goes: the experts' rows'
// to theirs, a shared expert's to its own, a dense FFN's to bs.act.
func actSlot(s devSlot) devSlot {
	switch s {
	case slotEG, slotEU, slotEGB, slotEUB:
		return slotEAct
	case slotShG, slotShU:
		return slotShAct
	case slotPLEG:
		return slotPLEA
	}
	return slotAct
}

// constSlot reports that slot s holds a constant the block's scratch is
// built with: read without a writer.
func constSlot(s devSlot) bool { return s == slotOnes || s == slotMIdent }

// A recurrence's projections in the tier's positional list (layer.tensors'
// order): the in-projection (the fused q|k|v, or x|B|C), the output gate,
// the gate streams, the out projection, and a mixer's own in-projection
// beside an attention.
const (
	posSSMInProj = 0
	posSSMGate   = 10
	posSSMBA     = 11
	posSSMOut    = 12
	posSSMIn     = 29
	// The low-rank pairs (KDA's decay and output gates; Mamba-1's dt, B
	// and C off x ride the same four).
	posLowA  = 18
	posLowB  = 19
	posGateA = 20
	posGateB = 21
)
