package amdgpu

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// Lower translates a kernel into LLVM IR text for a target.
//
// It is one forward walk, as ptx.Lower is. Every IR value is one SSA value
// named %v<id>. A loop is a while loop -- the header tests the count before
// the body, as SPIR-V's and MSL's do -- so a loop-carried value read after the
// loop is the header's phi, which is the final value, and a zero trip count
// leaves it at its init with no guard of its own.
func Lower(k *ir.Kernel, t Target) (string, error) {
	if err := k.Validate(); err != nil {
		return "", err
	}
	for i, o := range k.Ops {
		if ir.IsLdMatrix(o) {
			// ldmatrix's lane contract is NVIDIA's warp of 32 with a fixed
			// row-to-lane map; there is no instruction here that keeps it.
			return "", fmt.Errorf("amdgpu: op %d: no lowering for an ldmatrix load", i+1)
		}
	}
	if t.Wave != 32 && t.Wave != 64 {
		return "", fmt.Errorf("amdgpu: a wave of %d lanes", t.Wave)
	}
	for i, o := range k.Ops {
		switch {
		case o.Kind == ir.OpMMA || o.Kind == ir.OpMMAGet:
			// A deliberate refusal: OpMMA is NVIDIA's fragment contract, a
			// 32-lane warp with a per-lane layout neither MFMA nor WMMA has.
			return "", fmt.Errorf("amdgpu: op %d: no lowering for %s: it is the 32-lane mma.sync "+
				"fragment layout, which neither MFMA nor WMMA has", i, o.Kind)
		case ir.IsTileOp(o.Kind) || (o.Kind == ir.OpPhi && o.Tile != nil):
			return "", fmt.Errorf("amdgpu: op %d: no lowering for %s yet: the matrix units "+
				"(WMMA, MFMA) are reached through the tile ops and are not built", i, o.Kind)
		}
	}
	l := &lowerer{k: k, t: t, space: make([]int, len(k.Ops)+1), decls: map[string]bool{}}
	return l.lower()
}

type lowerer struct {
	k *ir.Kernel
	t Target
	b strings.Builder
	// globals is the module-scope text: the shared arrays.
	globals strings.Builder
	decls   map[string]bool
	// space is the address space a buffer value lives in: 1 for a parameter,
	// 3 for a shared array, 0 for a value that is not a buffer.
	space []int
	tmp   int
	// cur is the label of the block being emitted, which a phi names as the
	// edge it came from.
	cur  string
	lane string // the lane index within the wave, computed once at entry
}

func (l *lowerer) emit(format string, args ...any) {
	fmt.Fprintf(&l.b, "  "+format+"\n", args...)
}

func (l *lowerer) label(name string) {
	fmt.Fprintf(&l.b, "%s:\n", name)
	l.cur = name
}

func (l *lowerer) fresh() string {
	l.tmp++
	return fmt.Sprintf("%%t%d", l.tmp)
}

func (l *lowerer) declare(s string) { l.decls[s] = true }

func val(v ir.Value) string { return fmt.Sprintf("%%v%d", v) }

// ty is the LLVM type of an IR value type. U32 and I32 are both i32: LLVM's
// integers carry no sign, and each op says how it reads them.
func ty(t ir.Type) string {
	switch t {
	case ir.F32:
		return "float"
	case ir.Pred:
		return "i1"
	case ir.Ptr:
		return "ptr addrspace(1)"
	}
	return "i32"
}

// f32lit is an f32 in LLVM's text form: the double with the same value, in
// hex, which is exact for every f32 including NaN payloads and infinities.
func f32lit(bits uint32) string {
	return fmt.Sprintf("0x%016X", math.Float64bits(float64(math.Float32frombits(bits))))
}

func (l *lowerer) typeOf(v ir.Value) ir.Type { return l.k.Ops[v-1].Type }

func (l *lowerer) lower() (string, error) {
	k := l.k
	for _, o := range k.Ops {
		switch o.Kind {
		case ir.OpShuffleXor, ir.OpShuffle, ir.OpSubgroupBroadcast, ir.OpSubgroupReduce, ir.OpLaneID:
			l.lane = "%lane"
		}
	}
	l.label("entry")
	if l.lane != "" {
		l.declare("declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)")
		if l.t.Wave == 64 {
			l.declare("declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)")
			l.emit("%%lanelo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)")
			l.emit("%%lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %%lanelo)")
		} else {
			l.emit("%%lane = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)")
		}
	}
	if err := l.block(0, len(k.Ops)); err != nil {
		return "", err
	}
	l.emit("ret void")

	var s strings.Builder
	s.WriteString("; jitllm " + k.Name + " for " + l.t.Arch + "\n")
	s.WriteString("target triple = \"amdgcn-amd-amdhsa\"\n\n")
	s.WriteString(l.globals.String())
	ds := make([]string, 0, len(l.decls))
	for d := range l.decls {
		ds = append(ds, d)
	}
	sort.Strings(ds)
	for _, d := range ds {
		s.WriteString(d + "\n")
	}
	params := make([]string, len(k.Params))
	for i := range k.Params {
		params[i] = fmt.Sprintf("ptr addrspace(1) %%p%d", i)
	}
	fmt.Fprintf(&s, "\ndefine amdgpu_kernel void @%s(%s) #0 {\n", k.Name, strings.Join(params, ", "))
	s.WriteString(l.b.String())
	s.WriteString("}\n\n")
	g := k.Group[0] * max(k.Group[1], 1) * max(k.Group[2], 1)
	fmt.Fprintf(&s, "attributes #0 = { \"amdgpu-flat-work-group-size\"=\"%d,%d\" "+
		"\"uniform-work-group-size\"=\"true\" \"target-features\"=\"+wavefrontsize%d\" }\n", g, g, l.t.Wave)
	s.WriteString("\n!0 = !{}\n")
	return s.String(), nil
}

// invariant is the metadata a load of buffer value buf at index idx carries.
//
// A parameter is never written by a kernel that reads it (ir.Validate refuses
// a kernel that reads a buffer it writes, RULE 13), so every load from one is
// invariant for the launch. Only a load at a lane-independent index is marked:
// that is a value the whole wave shares, which LLVM may then read once, with a
// scalar load, and keep out of the loop it sits in. Without the mark LICM
// hoisted the ADDRESS of each unrolled load of flash attention's query row --
// one 64-bit pointer per element, 128 VGPRs for a 64-wide head -- and spilled
// it on gfx1030. Marking every parameter load hoists lane-varying loads too,
// which spilled the gated delta rule on every target: the mark is a register
// decision as much as a memory one, and shared memory, written and read by one
// kernel, carries none.
func (l *lowerer) invariant(buf, idx ir.Value) string {
	if l.space[buf] != 1 || l.k.LaneDependent(idx) {
		return ""
	}
	return ", !invariant.load !0"
}

// matchEnd is the EndLoop closing the Loop at op i.
func (l *lowerer) matchEnd(i int) int {
	d := 0
	for j := i; j < len(l.k.Ops); j++ {
		switch l.k.Ops[j].Kind {
		case ir.OpLoop:
			d++
		case ir.OpEndLoop:
			d--
			if d == 0 {
				return j
			}
		}
	}
	return -1
}

// block lowers ops[lo:hi), recursing on loops.
func (l *lowerer) block(lo, hi int) error {
	for i := lo; i < hi; i++ {
		o := l.k.Ops[i]
		v := ir.Value(i + 1)
		if o.Kind == ir.OpLoop {
			end := l.matchEnd(i)
			if end < 0 || end >= hi {
				return fmt.Errorf("amdgpu: unmatched loop at op %d", i)
			}
			if err := l.loop(i, end); err != nil {
				return err
			}
			i = end
			continue
		}
		if err := l.op(i, o, v); err != nil {
			return err
		}
	}
	return nil
}

// loop lowers the loop opened at op lo and closed at op hi:
//
//	pre:    br header
//	header: phi counter, phi per OpPhi; br (counter < n) body, exit
//	body:   ...; counter+1; br header
//	exit:
func (l *lowerer) loop(lo, hi int) error {
	o := l.k.Ops[lo]
	id := lo + 1
	header, body, exit := fmt.Sprintf("L%dh", id), fmt.Sprintf("L%db", id), fmt.Sprintf("L%dx", id)
	count := fmt.Sprintf("%d", o.Imm)
	if o.Imm < 0 {
		count = val(o.Args[0])
	}
	// The phis directly in this loop, not in a nested one.
	var phis []ir.Value
	d := 0
	for j := lo + 1; j < hi; j++ {
		switch l.k.Ops[j].Kind {
		case ir.OpLoop:
			d++
		case ir.OpEndLoop:
			d--
		case ir.OpPhi:
			if d == 0 {
				phis = append(phis, ir.Value(j+1))
			}
		}
	}
	pre := l.cur
	l.emit("br label %%%s", header)
	l.label(header)
	// The back-edge operands are not known until the body has been lowered:
	// the header is written with a placeholder per phi, patched below.
	ctr := fmt.Sprintf("%%c%d", id)
	ctrNext := fmt.Sprintf("%%c%dn", id)
	latchMark := fmt.Sprintf("@@latch%d@@", id)
	l.emit("%s = phi i32 [ 0, %%%s ], [ %s, %%%s ]", ctr, pre, ctrNext, latchMark)
	for _, p := range phis {
		po := l.k.Ops[p-1]
		l.emit("%s = phi %s [ %s, %%%s ], [ @@phi%d@@, %%%s ]", val(p), ty(po.Type),
			val(po.Args[0]), pre, p, latchMark)
	}
	cond := l.fresh()
	l.emit("%s = icmp ult i32 %s, %s", cond, ctr, count)
	l.emit("br i1 %s, label %%%s, label %%%s", cond, body, exit)
	l.label(body)
	if err := l.block(lo+1, hi); err != nil {
		return err
	}
	l.emit("%s = add i32 %s, 1", ctrNext, ctr)
	latch := l.cur
	l.emit("br label %%%s", header)
	l.label(exit)
	text := strings.ReplaceAll(l.b.String(), latchMark, latch)
	for _, p := range phis {
		text = strings.ReplaceAll(text, fmt.Sprintf("@@phi%d@@", p), val(l.k.Ops[p-1].Args[1]))
	}
	l.b.Reset()
	l.b.WriteString(text)
	return nil
}

// gep is the address of element idx+off of a buffer value.
func (l *lowerer) gep(buf, idx ir.Value, off int64, elem string) (string, string) {
	sp := l.space[buf]
	i64 := l.fresh()
	l.emit("%s = zext i32 %s to i64", i64, val(idx))
	if off != 0 {
		o := l.fresh()
		l.emit("%s = add i64 %s, %d", o, i64, off)
		i64 = o
	}
	p := l.fresh()
	l.emit("%s = getelementptr inbounds %s, ptr addrspace(%d) %s, i64 %s", p, elem, sp, l.base(buf), i64)
	return p, fmt.Sprintf("ptr addrspace(%d)", sp)
}

// base is the pointer a buffer value names.
func (l *lowerer) base(buf ir.Value) string {
	if l.space[buf] == 3 {
		return fmt.Sprintf("@sh%d", buf)
	}
	return val(buf)
}

func (l *lowerer) op(i int, o ir.Op, v ir.Value) error {
	a0, a1, a2 := o.Args[0], o.Args[1], o.Args[2]
	r := val(v)
	switch o.Kind {
	case ir.OpParam:
		l.space[v] = 1
		l.emit("%s = getelementptr i8, ptr addrspace(1) %%p%d, i64 0", r, o.Imm)
	case ir.OpShared:
		l.space[v] = 3
		align := 4
		for _, x := range l.k.Ops {
			if x.Kind == ir.OpLoadV || x.Kind == ir.OpStoreV {
				align = 16
			}
		}
		fmt.Fprintf(&l.globals, "@sh%d = internal addrspace(3) global [%d x %s] poison, align %d\n",
			v, o.Imm, ty(o.Type), align)
	case ir.OpBarrier:
		// __syncthreads: the fences are what order LDS and global accesses
		// across the barrier under the AMDGPU memory model.
		l.declare("declare void @llvm.amdgcn.s.barrier()")
		l.emit("fence syncscope(\"workgroup\") release")
		l.emit("call void @llvm.amdgcn.s.barrier()")
		l.emit("fence syncscope(\"workgroup\") acquire")
	case ir.OpGridBarrier:
		l.gridBarrier(i+1, val(a0))
	case ir.OpDepTask, ir.OpDepEpoch, ir.OpDepWait, ir.OpDepSignal, ir.OpDepExit:
		return fmt.Errorf("amdgpu: op %d: %s: dependency counters are not lowered for AMDGPU yet", i, o.Kind)
	case ir.OpConst:
		switch o.Type {
		case ir.F32:
			l.emit("%s = fadd float %s, 0.0", r, f32lit(uint32(o.Imm)))
		case ir.Pred:
			l.emit("%s = or i1 %t, false", r, o.Imm&1 != 0)
		default:
			l.emit("%s = add i32 %d, 0", r, int32(uint32(o.Imm)))
		}
	case ir.OpTID:
		l.declare("declare i32 @llvm.amdgcn.workitem.id.x()")
		l.emit("%s = call i32 @llvm.amdgcn.workitem.id.x()", r)
	case ir.OpCTAID:
		l.declare("declare i32 @llvm.amdgcn.workgroup.id.x()")
		l.emit("%s = call i32 @llvm.amdgcn.workgroup.id.x()", r)
	case ir.OpNTID:
		// The launch width is the declared group (backend.checkWidth), so it
		// is a constant here, as MSL lowers it.
		l.emit("%s = add i32 %d, 0", r, l.k.Group[0])
	case ir.OpAdd, ir.OpSub, ir.OpMul, ir.OpDiv, ir.OpRem:
		f := o.Type == ir.F32
		s := o.Type == ir.I32
		name := map[ir.Kind][3]string{
			ir.OpAdd: {"fadd", "add", "add"},
			ir.OpSub: {"fsub", "sub", "sub"},
			ir.OpMul: {"fmul", "mul", "mul"},
			ir.OpDiv: {"fdiv", "sdiv", "udiv"},
			ir.OpRem: {"frem", "srem", "urem"},
		}[o.Kind]
		op := name[2]
		if f {
			op = name[0]
		} else if s {
			op = name[1]
		}
		l.emit("%s = %s %s %s, %s", r, op, ty(o.Type), val(a0), val(a1))
	case ir.OpMin, ir.OpMax:
		// minnum/maxnum return the non-NaN operand, as PTX's min.f32 does.
		fn := map[ir.Kind][3]string{
			ir.OpMin: {"llvm.minnum.f32", "llvm.smin.i32", "llvm.umin.i32"},
			ir.OpMax: {"llvm.maxnum.f32", "llvm.smax.i32", "llvm.umax.i32"},
		}[o.Kind]
		n := fn[2]
		if o.Type == ir.F32 {
			n = fn[0]
		} else if o.Type == ir.I32 {
			n = fn[1]
		}
		t := ty(o.Type)
		l.declare(fmt.Sprintf("declare %s @%s(%s, %s)", t, n, t, t))
		l.emit("%s = call %s @%s(%s %s, %s %s)", r, t, n, t, val(a0), t, val(a1))
	case ir.OpShl, ir.OpShr:
		// PTX's shl.b32 and shr.u32 give 0 for a shift of 32 or more, where
		// LLVM's shl and lshr give poison. The select keeps PTX's answer, and
		// folds away when the shift is a constant below 32.
		op := "shl"
		if o.Kind == ir.OpShr {
			op = "lshr"
		}
		sh, in := l.fresh(), l.fresh()
		l.emit("%s = %s i32 %s, %s", sh, op, val(a0), val(a1))
		l.emit("%s = icmp ult i32 %s, 32", in, val(a1))
		l.emit("%s = select i1 %s, i32 %s, i32 0", r, in, sh)
	case ir.OpAnd:
		l.emit("%s = and i32 %s, %s", r, val(a0), val(a1))
	case ir.OpXor:
		l.emit("%s = xor i32 %s, %s", r, val(a0), val(a1))
	case ir.OpFunnel:
		// (hi:lo) >> n for n in [0,31]: fshr's operands are (hi, lo, n).
		l.declare("declare i32 @llvm.fshr.i32(i32, i32, i32)")
		l.emit("%s = call i32 @llvm.fshr.i32(i32 %s, i32 %s, i32 %s)", r, val(a1), val(a0), val(a2))
	case ir.OpLt:
		at := l.typeOf(a0)
		switch at {
		case ir.F32:
			l.emit("%s = fcmp olt float %s, %s", r, val(a0), val(a1))
		case ir.I32:
			l.emit("%s = icmp slt i32 %s, %s", r, val(a0), val(a1))
		default:
			l.emit("%s = icmp ult i32 %s, %s", r, val(a0), val(a1))
		}
	case ir.OpSelect:
		t := ty(o.Type)
		l.emit("%s = select i1 %s, %s %s, %s %s", r, val(a0), t, val(a1), t, val(a2))
	case ir.OpCvtF32:
		l.emit("%s = sitofp i32 %s to float", r, val(a0))
	case ir.OpCvtI32:
		// Saturating and NaN to 0, as PTX's cvt.rzi.s32.f32; a bare fptosi is
		// poison out of range.
		l.declare("declare i32 @llvm.fptosi.sat.i32.f32(float)")
		l.emit("%s = call i32 @llvm.fptosi.sat.i32.f32(float %s)", r, val(a0))
	case ir.OpExp:
		// exp(x) = 2^(x*log2 e), the form PTX's ex2.approx takes.
		l.declare("declare float @llvm.exp2.f32(float)")
		t := l.fresh()
		l.emit("%s = fmul float %s, %s", t, val(a0), f32lit(0x3FB8AA3B))
		l.emit("%s = call float @llvm.exp2.f32(float %s)", r, t)
	case ir.OpSqrt:
		l.declare("declare float @llvm.sqrt.f32(float)")
		l.emit("%s = call float @llvm.sqrt.f32(float %s)", r, val(a0))
	case ir.OpBitcast:
		from, to := l.typeOf(a0), o.Type
		switch {
		case from == ir.F32 && to == ir.F32:
			return fmt.Errorf("amdgpu: op %d: bitcast f32 to f32", i)
		case (from == ir.F32) == (to == ir.F32):
			// u32 <-> i32 is the same i32.
			l.emit("%s = or i32 %s, 0", r, val(a0))
		default:
			l.emit("%s = bitcast %s %s to %s", r, ty(from), val(a0), ty(to))
		}
	case ir.OpCvtF16H:
		h, hf := l.fresh(), l.fresh()
		l.emit("%s = trunc i32 %s to i16", h, val(a0))
		l.emit("%s = bitcast i16 %s to half", hf, h)
		l.emit("%s = fpext half %s to float", r, hf)
	case ir.OpPackF16:
		// fptrunc rounds to nearest even, which is what the other backends'
		// pack does, so all four agree bit for bit.
		var w [2]string
		for j, x := range [2]ir.Value{a0, a1} {
			hf, h, z := l.fresh(), l.fresh(), l.fresh()
			l.emit("%s = fptrunc float %s to half", hf, val(x))
			l.emit("%s = bitcast half %s to i16", h, hf)
			l.emit("%s = zext i16 %s to i32", z, h)
			w[j] = z
		}
		hi := l.fresh()
		l.emit("%s = shl i32 %s, 16", hi, w[1])
		l.emit("%s = or i32 %s, %s", r, w[0], hi)
	case ir.OpSubF16x2, ir.OpFmaF16x2:
		var hs []string
		args := []ir.Value{a0, a1}
		if o.Kind == ir.OpFmaF16x2 {
			args = append(args, a2)
		}
		for _, x := range args {
			h := l.fresh()
			l.emit("%s = bitcast i32 %s to <2 x half>", h, val(x))
			hs = append(hs, h)
		}
		res := l.fresh()
		if o.Kind == ir.OpSubF16x2 {
			l.emit("%s = fsub <2 x half> %s, %s", res, hs[0], hs[1])
		} else {
			l.declare("declare <2 x half> @llvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>)")
			l.emit("%s = call <2 x half> @llvm.fma.v2f16(<2 x half> %s, <2 x half> %s, <2 x half> %s)",
				res, hs[0], hs[1], hs[2])
		}
		l.emit("%s = bitcast <2 x half> %s to i32", r, res)
	case ir.OpFma:
		l.declare("declare float @llvm.fma.f32(float, float, float)")
		l.emit("%s = call float @llvm.fma.f32(float %s, float %s, float %s)", r, val(a0), val(a1), val(a2))
	case ir.OpDot4:
		l.dot4(r, a0, a1, a2)
	case ir.OpLoad:
		p, _ := l.gep(a0, a1, o.Imm, ty(o.Type))
		l.emit("%s = load %s, ptr addrspace(%d) %s, align 4%s", r, ty(o.Type), l.space[a0], p, l.invariant(a0, a1))
	case ir.OpStore:
		t := ty(l.typeOf(a2))
		p, _ := l.gep(a0, a1, o.Imm, t)
		l.emit("store %s %s, ptr addrspace(%d) %s, align 4", t, val(a2), l.space[a0], p)
	case ir.OpLoadV:
		off, n := ir.VecOf(o)
		t := ty(o.Type)
		p, _ := l.gep(a0, a1, off, t)
		vec := fmt.Sprintf("%%vec%d", v)
		l.emit("%s = load <%d x %s>, ptr addrspace(%d) %s, align %d%s", vec, n, t, l.space[a0], p, 4*n, l.invariant(a0, a1))
		l.emit("%s = extractelement <%d x %s> %s, i32 0", r, n, t, vec)
	case ir.OpLoadVGet:
		_, n := ir.VecOf(l.k.Ops[a0-1])
		l.emit("%s = extractelement <%d x %s> %%vec%d, i32 %d", r, n, ty(o.Type), a0, o.Imm)
	case ir.OpStoreV:
		off, n := ir.VecOf(o)
		t := ty(l.typeOf(o.Vargs[0]))
		vec := "poison"
		for j := 0; j < n; j++ {
			nv := l.fresh()
			l.emit("%s = insertelement <%d x %s> %s, %s %s, i32 %d", nv, n, t, vec, t, val(o.Vargs[j]), j)
			vec = nv
		}
		p, _ := l.gep(a0, a1, off, t)
		l.emit("store <%d x %s> %s, ptr addrspace(%d) %s, align %d", n, t, vec, l.space[a0], p, 4*n)
	case ir.OpShuffleXor:
		// ds_bpermute reads the value of the lane whose index is addr/4. A
		// mask below 32 keeps lane^mask inside the lane's own aligned 32, so
		// on a 64-wide wave the two halves each run the IR's 32-lane exchange.
		l.declare("declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)")
		src := val(a0)
		if o.Type == ir.F32 {
			b := l.fresh()
			l.emit("%s = bitcast float %s to i32", b, src)
			src = b
		}
		x, addr, got := l.fresh(), l.fresh(), l.fresh()
		l.emit("%s = xor i32 %s, %d", x, l.lane, o.Imm)
		l.emit("%s = shl i32 %s, 2", addr, x)
		l.emit("%s = call i32 @llvm.amdgcn.ds.bpermute(i32 %s, i32 %s)", got, addr, src)
		if o.Type == ir.F32 {
			l.emit("%s = bitcast i32 %s to float", r, got)
		} else {
			l.emit("%s = or i32 %s, 0", r, got)
		}
	case ir.OpSubgroupReduce:
		// The butterfly OpShuffleXor spells, each step a ds_bpermute inside
		// the lane's aligned 32, so a 64-wide wave runs it as two subgroups.
		cur := val(a0)
		for m := ir.SubgroupLanes / 2; m >= 1; m /= 2 {
			got := l.permute(cur, o.Type, func(addr string) {
				x := l.fresh()
				l.emit("%s = xor i32 %s, %d", x, l.lane, m)
				l.emit("%s = shl i32 %s, 2", addr, x)
			})
			nx := l.fresh()
			l.combine(nx, ir.ReduceOf(o), o.Type, cur, got)
			cur = nx
		}
		l.copy(r, o.Type, cur)
	case ir.OpSubgroupBroadcast, ir.OpShuffle:
		// Lane (src & 31) of the lane's own aligned 32: the high bits of
		// the lane index pick which half of a 64-wide wave.
		got := l.permute(val(a0), o.Type, func(addr string) {
			hi, lo, idx := l.fresh(), l.fresh(), l.fresh()
			l.emit("%s = and i32 %s, -32", hi, l.lane)
			l.emit("%s = and i32 %s, 31", lo, val(a1))
			l.emit("%s = or i32 %s, %s", idx, hi, lo)
			l.emit("%s = shl i32 %s, 2", addr, idx)
		})
		l.copy(r, o.Type, got)
	case ir.OpLaneID:
		l.emit("%s = and i32 %s, 31", r, l.lane)
	case ir.OpSubgroupSize:
		// The IR's subgroup is the aligned 32 its collectives act on when the
		// kernel declares Lanes; otherwise it is the wave the device runs.
		n := l.t.Wave
		if l.k.Lanes != 0 {
			n = ir.SubgroupLanes
		}
		l.emit("%s = add i32 %d, 0", r, n)
	case ir.OpPhi:
		// Written at the loop header by loop().
	case ir.OpPrefetch:
		// A hint; this lowering emits none.
	case ir.OpEndLoop:
		return fmt.Errorf("amdgpu: op %d: EndLoop outside a loop", i)
	default:
		return fmt.Errorf("amdgpu: no lowering for %s", o.Kind)
	}
	return nil
}

// permute reads x from the lane whose byte address addr computes, through
// ds_bpermute, bitcasting a float through i32. It returns the value read.
func (l *lowerer) permute(x string, t ir.Type, addr func(addr string)) string {
	l.declare("declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)")
	src := x
	if t == ir.F32 {
		b := l.fresh()
		l.emit("%s = bitcast float %s to i32", b, x)
		src = b
	}
	ad, got := l.fresh(), l.fresh()
	addr(ad)
	l.emit("%s = call i32 @llvm.amdgcn.ds.bpermute(i32 %s, i32 %s)", got, ad, src)
	if t != ir.F32 {
		return got
	}
	f := l.fresh()
	l.emit("%s = bitcast i32 %s to float", f, got)
	return f
}

// combine is r = op(a, b) for an OpSubgroupReduce step.
func (l *lowerer) combine(r string, op ir.ReduceOp, t ir.Type, a, b string) {
	if op == ir.ReduceAdd {
		if t == ir.F32 {
			l.emit("%s = fadd float %s, %s", r, a, b)
		} else {
			l.emit("%s = add i32 %s, %s", r, a, b)
		}
		return
	}
	fn := map[ir.ReduceOp][3]string{
		ir.ReduceMin: {"llvm.minnum.f32", "llvm.smin.i32", "llvm.umin.i32"},
		ir.ReduceMax: {"llvm.maxnum.f32", "llvm.smax.i32", "llvm.umax.i32"},
	}[op]
	n := fn[2]
	if t == ir.F32 {
		n = fn[0]
	} else if t == ir.I32 {
		n = fn[1]
	}
	tt := ty(t)
	l.declare(fmt.Sprintf("declare %s @%s(%s, %s)", tt, n, tt, tt))
	l.emit("%s = call %s @%s(%s %s, %s %s)", r, tt, n, tt, a, tt, b)
}

// copy names x as r: LLVM has no move, and an fadd of zero would turn -0
// into +0, so a select that cannot pick anything else does it.
func (l *lowerer) copy(r string, t ir.Type, x string) {
	l.emit("%s = select i1 true, %s %s, %s %s", r, ty(t), x, ty(t), x)
}

// dot4 is acc + the dot product of two words of four signed bytes.
func (l *lowerer) dot4(r string, a, b, acc ir.Value) {
	switch l.t.Dot4 {
	case Dot4Signed:
		l.declare("declare i32 @llvm.amdgcn.sdot4(i32, i32, i32, i1)")
		l.emit("%s = call i32 @llvm.amdgcn.sdot4(i32 %s, i32 %s, i32 %s, i1 false)", r, val(a), val(b), val(acc))
	case Dot4Mixed:
		l.declare("declare i32 @llvm.amdgcn.sudot4(i1, i32, i1, i32, i32, i1)")
		l.emit("%s = call i32 @llvm.amdgcn.sudot4(i1 true, i32 %s, i1 true, i32 %s, i32 %s, i1 false)",
			r, val(a), val(b), val(acc))
	default:
		va, vb := l.fresh(), l.fresh()
		l.emit("%s = bitcast i32 %s to <4 x i8>", va, val(a))
		l.emit("%s = bitcast i32 %s to <4 x i8>", vb, val(b))
		wa, wb, m := l.fresh(), l.fresh(), l.fresh()
		l.emit("%s = sext <4 x i8> %s to <4 x i32>", wa, va)
		l.emit("%s = sext <4 x i8> %s to <4 x i32>", wb, vb)
		l.emit("%s = mul <4 x i32> %s, %s", m, wa, wb)
		l.declare("declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)")
		s := l.fresh()
		l.emit("%s = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %s)", s, m)
		l.emit("%s = add i32 %s, %s", r, s, val(acc))
	}
}
