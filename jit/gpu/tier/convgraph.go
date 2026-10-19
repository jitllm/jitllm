package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A convolution-path block (one ConvLayers runs: the call's rows on the card
// from block to block, a program built at admission) lowered from its graph
// rather than from a list written for its kind. The walk is the device
// program's: each graph op becomes typed program ops, every value takes the
// first free buffer of a pool coloured by liveness (RULE 13 the colouring's
// constraint: an op's output never takes a buffer it reads), and each op
// records its launches through the path's own builders (kern, launch, mm), so
// a block of any composition of these primitives builds -- the FastConformer
// is one.

// numPool is the liveness pool's size.
const numPool = 12

// poolBufs is the scratch each pool slot names.
var poolBufs = [numPool]cbuf{cT1, cT2, cE1, cE2, cQ, cK, cV, cAO, cX1, cX2, cX3, cCol}

// vecAt is a block vector by role: the scratch name its upload is reached by.
type vecAt struct {
	role jlm.Role
	b    cbuf
}

// graphConvProgram records block cl's launches from graph gr over r rows
// (plan HIn): the block's input in cX, its output into cY, its second input
// (the positions' table, 2r-1 rows) in cCPE. ws is the block's weights in
// weightList order, each carrying its role.
func (g *devTier) graphConvProgram(cl *convLayer, gr *graph.Graph, ws []nn.Weight,
	kern func(convKey, func() (*ir.Kernel, error)) backend.Kernel,
	launch func(backend.Kernel, int, ...cbuf), mm func(int, cbuf, cbuf, int, int), want func(cbuf, int),
	need []int) (*devProgram, string) {
	r, eps := cl.p.HIn, float32(cl.p.Eps)
	prog := &devProgram{}
	need[cTN] = max(need[cTN], 4)
	mat := func(role jlm.Role) int {
		for i, w := range ws {
			if w.Role == role && (len(w.Data) != 0 || w.Packed != nil) {
				return i
			}
		}
		return -1
	}
	vec := func(role jlm.Role) (cbuf, bool) {
		for _, v := range cl.vecRole {
			if v.role == role {
				return v.b, true
			}
		}
		return 0, false
	}
	// Liveness: each graph value's last reader, and the block's output.
	last := make([]int, len(gr.Values))
	for i := range last {
		last[i] = -1
	}
	for i, o := range gr.Ops {
		for _, x := range o.In {
			last[x] = i
		}
	}
	out := gr.Out[0]
	// Each graph value's program value, buffer and row count.
	pv := make([]devVal, len(gr.Values))
	buf := make([]cbuf, len(gr.Values))
	rows := make([]int, len(gr.Values))
	var busy [numPool]bool
	slotOf := func(b cbuf) devSlot {
		switch b {
		case cX:
			return slotConvIn
		case cY:
			return slotConvOut
		case cCPE:
			return slotConvPos
		}
		for i, p := range poolBufs {
			if p == b {
				return slotPool0 + devSlot(i)
			}
		}
		return slotConvIn
	}
	take := func(v graph.ValueID) (cbuf, bool) {
		if v == out {
			return cY, true
		}
		for i := range busy {
			if !busy[i] {
				busy[i] = true
				return poolBufs[i], true
			}
		}
		return 0, false
	}
	free := func(b cbuf) {
		for i, p := range poolBufs {
			if p == b {
				busy[i] = false
			}
		}
	}
	def := func(v graph.ValueID, b cbuf, n int) devVal {
		buf[v], rows[v] = b, n
		pv[v] = prog.val(gr.Values[v].Name, devType{Elem: elemF32, Layout: layoutRow, Width: gr.Values[v].Width}, slotOf(b))
		if b < nConvScratch {
			want(b, n*gr.Values[v].Width) // the scratch's; a vector is the block's own
		}
		return pv[v]
	}
	def(gr.In[0], cX, r)
	def(gr.In[1], cCPE, 2*r-1)
	f32 := func(v graph.ValueID) devType {
		return devType{Elem: elemF32, Layout: layoutRow, Width: gr.Values[v].Width}
	}
	for i := 0; i < len(gr.Ops); i++ {
		o := gr.Ops[i]
		src := []int{i}
		ins := make([]devVal, len(o.In))
		need := make([]devType, len(o.In))
		for j, x := range o.In {
			ins[j], need[j] = pv[x], f32(x)
		}
		y := o.Out[0]
		w := gr.Values[y].Width
		n := r
		if len(o.In) > 0 {
			n = rows[o.In[0]]
		}
		missing := false
		bind := func(role jlm.Role) cbuf {
			b, ok := vec(role)
			missing = missing || !ok
			return b
		}
		in0 := func() cbuf { return buf[o.In[0]] }
		// A matvec's bias with the activation of its row's only reader
		// fused into one launch (BiasAct): the program op spans both.
		act := kernels.ActIdentity
		if o.Kind == graph.OpMatVec && len(o.Weight) == 2 && i+1 < len(gr.Ops) {
			if nx := gr.Ops[i+1]; nx.Kind == graph.OpAct && nx.In[0] == y && last[y] == i+1 && nx.Attr.Act == graph.ActSiLU {
				act, y, src = kernels.ActSiLU, nx.Out[0], []int{i, i + 1}
			}
		}
		// The output's buffer, taken before the inputs free: an op never
		// writes a buffer it reads.
		dst, ok := take(y)
		if !ok {
			return nil, fmt.Sprintf("op %d (%s): more live values than the pool's %d buffers", i, o.Kind, numPool)
		}
		var kind devKind
		switch o.Kind {
		case graph.OpLayerNorm:
			kind = dkNorm
			parts := partsFor(w)
			want(cLP, parts*n)
			want(cLV, parts*n)
			wb, bb := bind(o.Weight[0].Role), bind(o.Weight[1].Role)
			launch(kern(convKey{id: kidLnpart, a: [8]int{w, parts, n}}, func() (*ir.Kernel, error) {
				return kernels.LayerNormPartRows(w, parts, n)
			}), parts*n, in0(), cLP)
			launch(kern(convKey{id: kidLnvar, a: [8]int{w, parts, n}}, func() (*ir.Kernel, error) {
				return kernels.LayerNormVarRows(w, parts, n)
			}), parts*n, in0(), cLP, cLV)
			launch(kern(convKey{id: kidLnapply, a: [8]int{w, parts, n}, eps: eps}, func() (*ir.Kernel, error) {
				return kernels.LayerNormApplyRows(w, parts, eps, true, n)
			}), w*n, in0(), wb, cLP, cLV, bb, dst)
		case graph.OpMatVec:
			kind = dkMatVec
			slot := mat(o.Weight[0].Role)
			if slot < 0 {
				return nil, fmt.Sprintf("op %d: %v, which the block's device copy does not hold", i, o.Weight[0].Role)
			}
			k := gr.Values[o.In[0]].Width
			if len(o.Weight) == 1 {
				mm(slot, in0(), dst, n, k)
				break
			}
			tmp, ok := take(-1)
			if !ok {
				return nil, fmt.Sprintf("op %d: more live values than the pool's %d buffers", i, numPool)
			}
			mm(slot, in0(), tmp, n, k)
			bb := bind(o.Weight[1].Role)
			launch(kern(convKey{id: kidBiasAct, act: act, a: [8]int{w, n}}, func() (*ir.Kernel, error) {
				return kernels.BiasAct(w, n, act)
			}), w*n, tmp, bb, dst)
			free(tmp)
		case graph.OpAct:
			kind = dkAct
			if o.Attr.Act != graph.ActSiLU {
				return nil, fmt.Sprintf("op %d: an activation %d on the convolution path", i, o.Attr.Act)
			}
			launch(kern(convKey{id: kidAct, act: kernels.ActSiLU, a: [8]int{w * n}}, func() (*ir.Kernel, error) {
				return kernels.Act(w*n, kernels.ActSiLU)
			}), w*n, in0(), dst)
		case graph.OpAdd:
			kind = dkAdd
			id, mk := kidAdd, kernels.Add
			if o.Attr.Alpha == graph.ConformerHalf {
				id, mk = kidAddhalf, kernels.AddHalf
			} else if o.Attr.Alpha != 0 {
				return nil, fmt.Sprintf("op %d: a residual scale %g on the convolution path", i, o.Attr.Alpha)
			}
			launch(kern(convKey{id: id, a: [8]int{w * n}}, func() (*ir.Kernel, error) { return mk(w * n) }),
				w*n, in0(), buf[o.In[1]], dst)
		case graph.OpActMul:
			kind = dkActMul
			if o.Attr.Act != graph.ActSigmoid {
				return nil, fmt.Sprintf("op %d: a gate %d on the convolution path", i, o.Attr.Act)
			}
			launch(kern(convKey{id: kidSigmoidmul, a: [8]int{w * n}}, func() (*ir.Kernel, error) {
				return kernels.SigmoidMul(w * n)
			}), w*n, in0(), buf[o.In[1]], dst)
		case graph.OpConv:
			kind = dkDWConvRows
			taps := o.Attr.Taps
			launch(kern(convKey{id: kidDwconv1drows, a: [8]int{taps, w, n}}, func() (*ir.Kernel, error) {
				return kernels.DWConv1dRows(taps, w, n)
			}), w*n, in0(), bind(o.Weight[0].Role), cTN, dst)
		case graph.OpAffine:
			kind = dkAffineRows
			launch(kern(convKey{id: kidAffinerows, a: [8]int{w, n}}, func() (*ir.Kernel, error) {
				return kernels.AffineRows(w, n)
			}), w*n, in0(), bind(o.Weight[0].Role), bind(o.Weight[1].Role), dst)
		case graph.OpRelAttend:
			kind = dkRelAttend
			heads, kd := o.Attr.NHead, o.Attr.HeadDim
			want(cSc, n*heads*n)
			want(cSc2, n*heads*n)
			launch(kern(convKey{id: kidRelposscores, a: [8]int{n, heads, kd}}, func() (*ir.Kernel, error) {
				return kernels.RelPosScores(n, heads, kd)
			}), n*heads*n, buf[o.In[0]], buf[o.In[1]], buf[o.In[3]], bind(o.Weight[0].Role), bind(o.Weight[1].Role), cTN, cSc)
			launch(kern(convKey{id: kidRowsoftmax, a: [8]int{n * heads, n}}, func() (*ir.Kernel, error) {
				return kernels.RowSoftmax(n*heads, n)
			}), n*heads, cSc, cSc2)
			launch(kern(convKey{id: kidAttnaccrows, a: [8]int{n, heads, kd}}, func() (*ir.Kernel, error) {
				return kernels.AttnAccRows(n, heads, kd)
			}), n*w, cSc2, buf[o.In[2]], cTN, dst)
		default:
			return nil, fmt.Sprintf("op %d: %s on the convolution path", i, o.Kind)
		}
		if missing {
			return nil, fmt.Sprintf("op %d (%s): a vector the block's device copy does not hold", i, o.Kind)
		}
		prog.ops = append(prog.ops, devOp{Kind: kind, Src: src, In: ins, Out: []devVal{def(y, dst, n)}, Need: need})
		// The inputs whose last reader this was free their buffers.
		at := src[len(src)-1]
		for _, x := range o.In {
			if last[x] <= at {
				free(buf[x])
			}
		}
		i = at
	}
	return prog, ""
}
