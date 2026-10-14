// Package msl lowers the IR to Metal Shading Language.
//
// MSL has no packed int8 dot product (PTX has dp4a, SPIR-V OpSDot, arm64
// SDOT), so Dot4 unpacks both operands and issues a four-wide dot. The same
// kernel therefore costs more arithmetic here, and a tile tuned on an NVIDIA
// card does not transfer: the tuner re-sweeps per device.
//
// Metal compiles source at runtime the way the CUDA driver JITs PTX.
package msl

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// preamble is prepended to every kernel.
//
// Its int8 dots are done in float. Apple's ALU is fp32-centric: a 32-bit
// integer multiply is several instructions there while a float FMA is one.
// Each product is int8 x int8, at most 16129 in magnitude, and four of them
// sum to at most 64516, all exact in a float32 mantissa, so the integer
// accumulation stays bit-identical.
//
// unpack_snorm4x8_to_float widens four bytes in one instruction, where
// float4(as_type<char4>(w)) is four byte extracts and four converts per
// operand. It yields v/127, so the dot comes out scaled by 1/127^2; a multiply
// by 16129 and rint recover the exact integer (the true value is an integer of
// at most 64516 and the accumulated relative error is ~1e-7). snorm clamps to
// [-1, 1], so it needs no operand byte to be -128: every signed byte the
// kernels see comes from a quantizer that scales by 127/amax, and every other
// format masks its quants to a small unsigned range before the dot.
// TestPackNeverEmitsInt8Min holds the packers to it. The measurements behind
// both choices are in docs/engineering-history/gpu-kernels.md.
const preamble = `#include <metal_stdlib>
using namespace metal;

// Four-way packed signed int8 dot product with accumulate, done in float:
// every product and partial sum is exact in a float32 mantissa.
inline int jitllm_dot4(uint a, uint b, int c) {
    float4 x = float4(as_type<char4>(a)), y = float4(as_type<char4>(b));
    return c + int(dot(x, y));
}

// The same dots for operands within [-127,127] (ir.Dot4Bounded):
// unpack_snorm4x8_to_float widens in one instruction, and rint recovers the
// exact integer from the 1/127^2 scale.
inline int jitllm_dot4b(uint a, uint b, int c) {
    float4 x = unpack_snorm4x8_to_float(a), y = unpack_snorm4x8_to_float(b);
    return c + int(rint(dot(x, y) * 16129.0f));
}

inline float jitllm_dot4bf(uint a, uint b, float c) {
    float4 x = unpack_snorm4x8_to_float(a), y = unpack_snorm4x8_to_float(b);
    return c + rint(dot(x, y) * 16129.0f);
}

// The same dot with the accumulator left in float, for a chain whose only
// exit is a conversion to float anyway (floatCarried).
inline float jitllm_dot4f(uint a, uint b, float c) {
    float4 x = float4(as_type<char4>(a)), y = float4(as_type<char4>(b));
    return c + dot(x, y);
}
`

func tname(t ir.Type) string {
	switch t {
	case ir.I32:
		return "int"
	case ir.U32:
		return "uint"
	case ir.F32:
		return "float"
	case ir.Pred:
		return "bool"
	}
	return "void"
}

type lowerer struct {
	b    strings.Builder
	k    *ir.Kernel
	name []string // ir.Value -> MSL expression or variable
	fc   []bool   // ir.Value -> carried as float; see floatCarried
	ind  int
}

// tn is tname with the float-carrying override applied.
func (l *lowerer) tn(v ir.Value, t ir.Type) string {
	if int(v) < len(l.fc) && l.fc[v] {
		return "float"
	}
	return tname(t)
}

// floatCarried marks the integer values that MSL should hold in a float
// register instead.
//
// Apple has no integer dot product, so jitllm_dot4 computes in float, and a
// Dot4 chain whose only exit is CvtF32 would otherwise convert float to int
// and back every four elements for no consumer. Carrying it in float is exact:
// a float32 mantissa holds every integer below 2^24, and the largest
// accumulation any format reaches (a Q6_K super-block, 256 x 127 x 127) is
// four times inside that.
//
// A value qualifies when it is produced by Dot4 or a Phi and every consumer is
// another Dot4's accumulator, a CvtF32 or a Phi. Any other consumer wants the
// integer and disqualifies the chain.
func floatCarried(k *ir.Kernel) []bool {
	fc := make([]bool, len(k.Ops)+1)
	for i, o := range k.Ops {
		if o.Kind == ir.OpDot4 || o.Kind == ir.OpPhi {
			fc[i+1] = true
		}
	}
	// Consumers first, then the chain conditions, to a fixed point: dropping one
	// value can disqualify the Phi that carries it and the Dot4 that feeds it.
	for changed := true; changed; {
		changed = false
		drop := func(v ir.Value) {
			if v != 0 && int(v) < len(fc) && fc[v] {
				fc[v] = false
				changed = true
			}
		}
		for i, o := range k.Ops {
			for pos, a := range o.Args {
				if a == 0 || int(a) >= len(fc) || !fc[a] {
					continue
				}
				ok := (o.Kind == ir.OpDot4 && pos == 2) ||
					(o.Kind == ir.OpCvtF32 && pos == 0) ||
					(o.Kind == ir.OpPhi && pos < 2)
				if !ok {
					drop(a)
				}
			}
			if !fc[i+1] {
				continue
			}
			// A Phi only carries a float if what it is updated with is one.
			if o.Kind == ir.OpPhi && !fc[o.Args[1]] {
				drop(ir.Value(i + 1))
			}
		}
	}
	return fc
}

// Emit returns MSL source for the kernel.
func Emit(k *ir.Kernel) (string, error) {
	if err := k.Validate(); err != nil {
		return "", err
	}
	for i, o := range k.Ops {
		if ir.IsLdMatrix(o) {
			// ldmatrix's lane contract is NVIDIA's warp of 32 with a fixed
			// row-to-lane map; there is no instruction here that keeps it.
			return "", fmt.Errorf("msl: op %d: no lowering for an ldmatrix load", i+1)
		}
	}
	l := &lowerer{k: k, name: make([]string, len(k.Ops)+1), fc: floatCarried(k), ind: 1}
	l.b.WriteString(preamble)
	grid := k.GridSync()
	if grid {
		wait := "while (atomic_load_explicit(s + 1, memory_order_relaxed) == g) {\n            }"
		if gridBroken {
			wait = "(void)g;"
		}
		l.b.WriteString(strings.Replace(gridPreamble, "JITLLM_GRID_WAIT", wait, 1))
	}
	if ir.UsesTiles(k) {
		// Not in the preamble unconditionally: it is an extra header on every
		// kernel the device compiles, and every other one has no matrix in it.
		l.b.WriteString("#include <metal_simdgroup_matrix>\n")
	}
	if k.Comment != "" {
		fmt.Fprintf(&l.b, "\n// %s\n", k.Comment)
	}
	fmt.Fprintf(&l.b, "\nkernel void %s(", k.Name)
	for i, p := range k.Params {
		if i > 0 {
			l.b.WriteString(",\n" + strings.Repeat(" ", 13+len(k.Name)))
		}
		et := tname(p.Elem)
		if p.IsTile {
			et = tileElemName(p.Tile)
		}
		// A grid barrier's kernel reads, after the barrier, words other
		// threadgroups wrote before it: coherent(device) keeps those loads
		// off a cache that is not coherent across the GPU (MSL 3.2).
		coh := ""
		if grid {
			coh = "coherent(device) "
		}
		fmt.Fprintf(&l.b, "%sdevice %s* %s [[buffer(%d)]]", coh, et, p.Name, i)
	}
	pad := ",\n" + strings.Repeat(" ", 13+len(k.Name))
	fmt.Fprintf(&l.b, "%suint jitllm_tid [[thread_position_in_threadgroup]]", pad)
	// The lane and width attributes only where a kernel asks, so no other
	// kernel's source changes.
	for _, o := range k.Ops {
		if o.Kind == ir.OpLaneID {
			fmt.Fprintf(&l.b, "%suint jitllm_lane [[thread_index_in_simdgroup]]", pad)
			break
		}
	}
	for _, o := range k.Ops {
		if o.Kind == ir.OpSubgroupSize {
			fmt.Fprintf(&l.b, "%suint jitllm_sgsize [[threads_per_simdgroup]]", pad)
			break
		}
	}
	if grid {
		fmt.Fprintf(&l.b, "%suint jitllm_ngroups [[threadgroups_per_grid]]", pad)
	}
	fmt.Fprintf(&l.b, "%suint jitllm_ctaid [[threadgroup_position_in_grid]]) {\n", pad)

	if err := l.block(0, len(k.Ops)); err != nil {
		return "", err
	}
	l.b.WriteString("}\n")
	return l.b.String(), nil
}

func (l *lowerer) line(f string, a ...any) {
	l.b.WriteString(strings.Repeat("    ", l.ind))
	fmt.Fprintf(&l.b, f, a...)
	l.b.WriteByte('\n')
}

// def emits `T vN = expr;` and records vN as the value's name.
func (l *lowerer) def(i int, t ir.Type, f string, a ...any) {
	v := fmt.Sprintf("v%d", i+1)
	l.name[i+1] = v
	l.line("%s %s = %s;", l.tn(ir.Value(i+1), t), v, fmt.Sprintf(f, a...))
}

func (l *lowerer) n(v ir.Value) string { return l.name[v] }

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

func (l *lowerer) block(lo, hi int) error {
	for i := lo; i < hi; i++ {
		o := l.k.Ops[i]
		switch o.Kind {
		case ir.OpLoop:
			end := l.matchEnd(i)
			if end < 0 || end >= hi {
				return fmt.Errorf("msl: unmatched loop at op %d", i)
			}
			if err := l.loop(i, end); err != nil {
				return err
			}
			i = end
		case ir.OpEndLoop:
			return fmt.Errorf("msl: op %d: EndLoop outside a loop", i)
		case ir.OpPhi:
			if l.name[i+1] == "" {
				return fmt.Errorf("msl: op %d: phi outside a loop", i)
			}
		case ir.OpParam:
			l.name[i+1] = l.k.Params[o.Imm].Name
		case ir.OpConst:
			switch o.Type {
			case ir.F32:
				// Imm is the bit pattern, matching ptx and spirv.
				l.name[i+1] = fmt.Sprintf("as_type<float>(%du)", uint32(o.Imm))
			case ir.U32:
				l.name[i+1] = fmt.Sprintf("%du", o.Imm)
			default:
				l.name[i+1] = fmt.Sprintf("%d", o.Imm)
			}
		case ir.OpTID:
			l.name[i+1] = "jitllm_tid"
		case ir.OpCTAID:
			l.name[i+1] = "jitllm_ctaid"
		case ir.OpNTID:
			l.name[i+1] = fmt.Sprintf("%du", l.k.Group[0])
		case ir.OpStore:
			l.line("%s[%s] = %s;", l.n(o.Args[0]), l.idx(o), l.n(o.Args[2]))
		case ir.OpLoad:
			l.def(i, o.Type, "%s[%s]", l.n(o.Args[0]), l.idx(o))
		case ir.OpShared:
			// Declared where it appears; ir.Validate requires it before any
			// loop, since an in-loop threadgroup declaration may be a fresh
			// object per iteration.
			l.name[i+1] = fmt.Sprintf("jitllm_sh%d", i+1)
			l.line("threadgroup %s %s[%d];", tname(o.Type), l.name[i+1], o.Imm)
		case ir.OpBarrier:
			l.line("threadgroup_barrier(mem_flags::mem_threadgroup);")
		case ir.OpGridBarrier:
			l.line("jitllm_grid_barrier((device atomic_uint*)%s, jitllm_tid, jitllm_ngroups);", l.n(o.Args[0]))
		case ir.OpDepTask, ir.OpDepEpoch, ir.OpDepWait, ir.OpDepSignal, ir.OpDepExit:
			// Metal states no forward progress between threadgroups
			// (docs/design/dep-block.md).
			return fmt.Errorf("msl: op %d: %s: dependency counters are not lowered for Metal", i, o.Kind)
		default:
			if err := l.arith(i, o); err != nil {
				return err
			}
		}
	}
	return nil
}

// space is the MSL address space a buffer value lives in: a shared array is
// threadgroup memory, a parameter is device memory.
func (l *lowerer) space(buf ir.Value) string {
	if buf > 0 && int(buf) <= len(l.k.Ops) && l.k.Ops[buf-1].Kind == ir.OpShared {
		return "threadgroup"
	}
	return "device"
}

func (l *lowerer) idx(o ir.Op) string {
	if o.Imm == 0 {
		return l.n(o.Args[1])
	}
	return fmt.Sprintf("%s + %d", l.n(o.Args[1]), o.Imm)
}

func (l *lowerer) arith(i int, o ir.Op) error {
	a, b, c := l.n(o.Args[0]), l.n(o.Args[1]), l.n(o.Args[2])
	bin := func(op string) { l.def(i, o.Type, "%s %s %s", a, op, b) }
	switch o.Kind {
	case ir.OpAdd:
		bin("+")
	case ir.OpSub:
		bin("-")
	case ir.OpMul:
		bin("*")
	case ir.OpDiv:
		bin("/")
	case ir.OpRem:
		bin("%")
	case ir.OpShl:
		bin("<<")
	case ir.OpShr:
		// The operand type is uint, so C's >> is already logical here. It would
		// become arithmetic on a signed operand, which is why OpShr is only
		// ever emitted on U32.
		bin(">>")
	case ir.OpAnd:
		bin("&")
	case ir.OpXor:
		bin("^")
	case ir.OpFunnel:
		// (hi:lo) >> n with the shift split so n == 0 never shifts by 32.
		l.def(i, ir.U32, "((%s >> %s) | ((%s << 1u) << (31u - %s)))", a, c, b, c)
	case ir.OpLt:
		l.def(i, ir.Pred, "%s < %s", a, b)
	case ir.OpSelect:
		l.def(i, o.Type, "(%s ? %s : %s)", a, b, c)
	case ir.OpMin:
		l.def(i, o.Type, "min(%s, %s)", a, b)
	case ir.OpMax:
		l.def(i, o.Type, "max(%s, %s)", a, b)
	case ir.OpExp:
		l.def(i, ir.F32, "exp(%s)", a)
	case ir.OpSqrt:
		l.def(i, ir.F32, "sqrt(%s)", a)
	case ir.OpCvtF32:
		if o.Args[0] != 0 && int(o.Args[0]) < len(l.fc) && l.fc[o.Args[0]] {
			// Already a float: the whole point of carrying the chain.
			l.name[i+1] = a
			return nil
		}
		l.def(i, ir.F32, "float(%s)", a)
	case ir.OpCvtI32:
		l.def(i, ir.I32, "int(%s)", a)
	case ir.OpBitcast:
		if o.Type == ir.F32 {
			l.def(i, ir.F32, "as_type<float>(%s)", a)
		} else {
			l.def(i, ir.U32, "as_type<uint>(%s)", a)
		}
	case ir.OpPackF16:
		l.def(i, ir.U32, "as_type<uint>(half2(%s, %s))", a, b)
	case ir.OpCvtF16H:
		l.def(i, ir.F32, "float(as_type<half2>(%s).x)", a)
	case ir.OpFma:
		l.def(i, ir.F32, "fma(%s, %s, %s)", a, b, c)
	case ir.OpSubF16x2:
		// Packed binary16 pairs, as PTX's sub.rn.f16x2: MSL's half2 is the
		// same two IEEE halves in one 32-bit value, so the bits pass through.
		l.def(i, ir.U32, "as_type<uint>(as_type<half2>(%s) - as_type<half2>(%s))", a, b)
	case ir.OpFmaF16x2:
		l.def(i, ir.U32, "as_type<uint>(fma(as_type<half2>(%s), as_type<half2>(%s), as_type<half2>(%s)))", a, b, c)
	case ir.OpLoadV:
		// One aligned vector access, as PTX's ld.v4: the IR states the
		// address is aligned to the whole vector (see ir.OpLoadV). The
		// components are named .x .. .w of one vector value.
		off, n := ir.VecOf(o)
		vt := fmt.Sprintf("%s%d", tname(o.Type), n)
		v := fmt.Sprintf("v%d", i+1)
		l.line("%s %s = *((%s %s*)(%s + %s + %d));", vt, v, l.space(o.Args[0]), vt, a, b, off)
		l.name[i+1] = v + ".x"
	case ir.OpLoadVGet:
		base := l.n(o.Args[0])
		l.name[i+1] = base[:len(base)-2] + "." + string("xyzw"[o.Imm])
	case ir.OpStoreV:
		off, n := ir.VecOf(o)
		et := l.k.Ops[o.Vargs[0]-1].Type
		vt := fmt.Sprintf("%s%d", tname(et), n)
		parts := make([]string, n)
		for j := range parts {
			parts[j] = l.n(o.Vargs[j])
		}
		l.line("*((%s %s*)(%s + %s + %d)) = %s(%s);", l.space(o.Args[0]), vt, a, b, off, vt,
			strings.Join(parts, ", "))
	case ir.OpDot4:
		// Imm == 1 is ir.Dot4Bounded: both operands within [-127,127].
		bnd := ""
		if o.Imm == 1 {
			bnd = "b"
		}
		if l.fc[i+1] {
			l.def(i, ir.I32, "jitllm_dot4%sf(%s, %s, %s)", bnd, a, b, c)
		} else {
			l.def(i, ir.I32, "jitllm_dot4%s(%s, %s, %s)", bnd, a, b, c)
		}
	case ir.OpShuffleXor:
		// simd_shuffle_xor is the direct equivalent of PTX's shfl.sync.bfly;
		// an Apple simdgroup is 32 lanes, the width ir's mask bound assumes.
		l.def(i, o.Type, "simd_shuffle_xor(%s, ushort(%d))", a, o.Imm)
	case ir.OpSubgroupReduce:
		// simd_sum/simd_max/simd_min reduce across the simdgroup and return
		// the result in every lane; an Apple simdgroup is the 32 lanes the IR
		// defines the op over.
		fn := map[ir.ReduceOp]string{ir.ReduceAdd: "simd_sum", ir.ReduceMax: "simd_max", ir.ReduceMin: "simd_min"}[ir.ReduceOf(o)]
		l.def(i, o.Type, "%s(%s)", fn, a)
	case ir.OpSubgroupBroadcast, ir.OpShuffle:
		// simd_shuffle for both: it takes a per-lane source, so it is also a
		// broadcast when the source is uniform, with no uniformity contract
		// of its own to honour.
		l.def(i, o.Type, "simd_shuffle(%s, ushort(%s & 31u))", a, b)
	case ir.OpLaneID:
		l.name[i+1] = "jitllm_lane"
	case ir.OpSubgroupSize:
		l.name[i+1] = "jitllm_sgsize"
	case ir.OpPrefetch:
		// MSL has no prefetch; the hint lowers to nothing.
	case ir.OpTileLoad, ir.OpTileSplat, ir.OpTileMMA, ir.OpTileStore:
		return l.tile(i, o)
	case ir.OpMMA, ir.OpMMAGet:
		// Declined for two different reasons. MMAS8: the hardware has no
		// integer matrix instruction. MMAF16: simdgroup_matrix exists, but it
		// is an opaque 8x8 tile with no documented lane layout, while ir.OpMMA
		// carries a per-lane fragment contract; it needs the tile ops instead.
		// Either way the tier falls back to the FMA/dp4a tile.
		if sh := ir.MMAShapeOf(o); o.Kind == ir.OpMMA && sh.Kind == ir.MMAF16 {
			return fmt.Errorf("msl: no float matrix lowering: simdgroup_matrix "+
				"exists on this target and is an opaque tile, where %s carries a "+
				"per-lane fragment layout; it needs a tile-shaped op, not an "+
				"encoding", o.Kind)
		}
		return fmt.Errorf("msl: no integer matrix instruction: simdgroup_matrix is "+
			"float only, so %s has no lowering", o.Kind)
	default:
		return fmt.Errorf("msl: no lowering for %s", o.Kind)
	}
	return nil
}

// loop emits a counted for-loop. Loop-carried values become mutable variables
// declared before it and reassigned at the end of the body -- MSL has no phi,
// and the updates are already distinct SSA values so assignment order is
// irrelevant.
func (l *lowerer) loop(start, end int) error {
	var phis []int
	for j, d := start+1, 0; j < end; j++ {
		switch l.k.Ops[j].Kind {
		case ir.OpLoop:
			d++
		case ir.OpEndLoop:
			d--
		case ir.OpPhi:
			if d == 0 {
				phis = append(phis, j)
			}
		}
	}
	for _, j := range phis {
		o := l.k.Ops[j]
		v := fmt.Sprintf("p%d", j+1)
		l.name[j+1] = v
		ty := l.tn(ir.Value(j+1), o.Type)
		if o.Tile != nil {
			// A loop-carried simdgroup matrix (ir.TilePhi): a value type in
			// MSL, so the same declare-then-reassign shape as a scalar.
			t, err := tileT(*o.Tile)
			if err != nil {
				return err
			}
			ty = t
		}
		l.line("%s %s = %s;", ty, v, l.n(o.Args[0]))
	}
	bound := fmt.Sprintf("%du", l.k.Ops[start].Imm)
	if l.k.Ops[start].Imm < 0 {
		bound = l.n(l.k.Ops[start].Args[0]) // LoopN: the count is a value
	}
	l.line("for (uint i%d = 0u; i%d < %s; ++i%d) {", start, start, bound, start)
	l.ind++
	if err := l.block(start+1, end); err != nil {
		return err
	}
	for _, j := range phis {
		u := l.k.Ops[j].Args[1]
		if l.name[u] == "" {
			return fmt.Errorf("msl: op %d: phi update %d was never lowered", j, u)
		}
		l.line("%s = %s;", l.name[j+1], l.name[u])
	}
	l.ind--
	l.line("}")
	return nil
}

// gridPreamble is ir.OpGridBarrier's body, in a kernel that has one: the
// arrive-and-wait counter of ptx/grid.go. MSL's atomics are relaxed only; the
// order comes from atomic_thread_fence at device scope (MSL 3.2), a fence
// before arriving and one after the generation moves, and
// threadgroup_barrier(mem_device) carries it to and from the rest of the
// threadgroup. Apple documents no forward-progress guarantee between
// threadgroups; the launch is bounded by the device's co-residency limit.
const gridPreamble = `
#if __METAL_VERSION__ < 320
#error "a grid barrier needs MSL 3.2 (atomic_thread_fence at device scope, coherent(device))"
#endif
inline void jitllm_grid_barrier(device atomic_uint* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint g = atomic_load_explicit(s + 1, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        uint old = atomic_fetch_add_explicit(s, 1u, memory_order_relaxed);
        if (old == n - 1u) {
            atomic_store_explicit(s, 0u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_fetch_add_explicit(s + 1, 1u, memory_order_relaxed);
        } else {
            JITLLM_GRID_WAIT
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
`

// gridBroken makes every waiter leave without waiting (BreakGridBarrierForTest,
// jitllmtest builds only): the violation the grid barrier's gate must see.
var gridBroken bool
