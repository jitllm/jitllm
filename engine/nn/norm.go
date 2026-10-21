package nn

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// RMSNorm on generated code.
//
// The kernels are cached per width because the width is baked (the trip count
// and the tail are compile-time, as the matvec bakes K). A model uses two or
// three widths: n_embd for the block and output norms, head_dim for q/k norms.
type normKernels struct {
	// The mutex is for writers only. The maps stop changing after the first
	// few calls, and prefill calls this from every pool worker, so readers
	// load an immutable snapshot rather than contending on a lock whose cost
	// grows with the thread count.
	mu   sync.Mutex
	snap atomic.Pointer[normSnap]
}

// normSnap is an immutable snapshot: readers load the pointer and index the
// maps with no lock at all, writers copy-on-write under mu.
type normSnap struct {
	code map[tierN]*cpu.Code
	// konst is [1/n, eps, 1.0] per (width, eps). The kernel bakes the shape and
	// reads the floats, so the kernel cache is not keyed on a float.
	konst map[normKey][]float32
}

type normKey struct {
	n   int
	eps float64
}

// tierN keys a width-baked kernel by the tier it was generated for as well as
// its width: see elemSet for why the tier has to be in every key.
type tierN struct {
	t cpu.Tier
	n int
}

var norms normKernels

func init() {
	norms.snap.Store(&normSnap{code: map[tierN]*cpu.Code{}, konst: map[normKey][]float32{}})
	lns.snap.Store(&lnSnap{code: map[lnKey]*cpu.Code{}, kons: map[lnKons][]float32{}})
}

// elemSet is the elementwise kernels. They are not keyed by width -- each
// takes its length at runtime -- so there is one of each per tier, plus one
// shared constant block.
//
// Per tier because these are package-level (no JIT to record a tier): they ask
// cpu.HostTier on every call, so a forced-SSE run is never handed AVX2 kernels
// an earlier caller mapped.
type elemSet struct {
	softmax *cpu.Code
	logSm   *cpu.Code                   // the softmax's log form, for a reply's logprobs
	actMul  [len(cpu.Gated)]*cpu.Code   // indexed as cpu.Gated
	act     [len(cpu.Ungated)]*cpu.Code // indexed as cpu.Ungated
	sigMul  *cpu.Code                   // sigma(gate) * value, for a gated attention output
	axpy    *cpu.Code
	scale   *cpu.Code
	softcap *cpu.Code
	clamp   *cpu.Code
	pflines *cpu.Code // the fused runs' lookahead (lookahead.go)
	xielu   *cpu.Code // Apertus's ungated activation, its numbers a buffer
	ln      *cpu.Code // x = ln(x), a log-mel feature's log
}

var (
	elemSets   [cpu.NumTiers]elemSet
	elemOnce   tierOnce
	elemConsts = cpu.ActConsts()
	// xieluConsts is the xIELU kernel's block (cpu.XIELUConsts): exp's and
	// its polynomial's.
	xieluConsts = cpu.XIELUConsts()
	// one is alpha for a plain add. A package-level array rather than a stack
	// local so the pointer handed to the kernel is not a Go escape per call.
	one = [1]float32{1}
)

// elemFor returns this host's tier's elementwise kernels, emitting and mapping
// them on first use.
func elemFor() *elemSet {
	t := cpu.HostTier()
	s := &elemSets[t]
	elemOnce.do(t, func() { s.init(cpu.EmittersFor(t)) })
	return s
}

// tierOnce is a sync.Once per tier that does not latch a panic. The fills panic
// by design (mustEmit) when a tier has no kernel; sync.Once would mark itself
// done and later calls would dereference nil instead of naming the kernel.
type tierOnce struct {
	mu   sync.Mutex
	done [cpu.NumTiers]atomic.Bool
}

func (o *tierOnce) do(t cpu.Tier, f func()) {
	if o.done[t].Load() {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.done[t].Load() {
		return
	}
	f()
	o.done[t].Store(true)
}

// LayerNorm kernels, cached per (width, has-bias) for EmitRMSNorm's reason: the
// width is baked and the bias is a property of the file.
var lns struct {
	mu   sync.Mutex
	snap atomic.Pointer[lnSnap]
}

type lnSnap struct {
	code map[lnKey]*cpu.Code
	kons map[lnKons][]float32
}

// lnKey is a kernel: the layer norm, or one of its passes' other uses
// (cpu.RowMode), at a width.
type lnKey struct {
	t    cpu.Tier
	n    int
	bias bool
	mode cpu.RowMode
}

// lnKons is a kernel's constants, [1/n, eps, c]: c is 1 for the layer norm
// and the match, the gaussian's standard-deviation multiple for its cutoff.
type lnKons struct {
	n      int
	eps, c float64
}

func (s *elemSet) init(em *cpu.Emitters) {
	s.axpy = mustEmit("axpy")(em.Axpy())
	s.scale = mustEmit("scale")(em.Scale())
	s.softcap = mustEmit("softcap")(em.Softcap())
	s.clamp = mustEmit("clamp")(em.Clamp())
	s.pflines = mustEmit("prefetch_lines")(em.PrefetchLines())
	s.softmax = mustEmit("softmax")(em.Softmax())
	s.logSm = mustEmit("logsoftmax")(em.LogSoftmax())
	for i, k := range cpu.Gated {
		s.actMul[i] = mustEmit("actmul/" + k.String())(em.ActMul(k))
	}
	// The ungated set is enumerated from cpu.Ungated rather than restated, so
	// a fourth kind cannot be added on one side only.
	for i, k := range cpu.Ungated {
		s.act[i] = mustEmit("act/" + k.String())(em.Act(k))
	}
	s.sigMul = mustEmit("sigmoidmul")(em.SigmoidMul())
	s.xielu = mustEmit("xielu")(em.XIELU())
	s.ln = mustEmit("ln")(em.Ln())
}

// mustEmit maps an emitter's result or stops the process, naming the kernel:
// an emitter error and a Map refusal are both an engine that cannot compute a
// token. It is curried so a call site reads mustEmit("softmax")(em.Softmax()).
func mustEmit(name string) func([]byte, error) *cpu.Code {
	return func(b []byte, err error) *cpu.Code {
		if err != nil {
			panic(fmt.Sprintf("jit: cannot emit the %s kernel: %v", name, err))
		}
		return mustMap(name, b)
	}
}

// mustMap maps generated code or stops the process. There is no interpreted
// path behind any of these kernels.
func mustMap(name string, code []byte) *cpu.Code {
	c, err := cpu.Map(code)
	if err != nil {
		panic(fmt.Sprintf("jit: cannot map the %s kernel: %v", name, err))
	}
	return c
}

// lengthPanic reports a caller handing a kernel a shorter operand than its
// destination -- a programming error, which is the only way these can fail.
func lengthPanic(op string, n, m int) {
	panic(fmt.Sprintf("jit: %s over %d elements with a %d-element operand", op, n, m))
}

// Softmax32JIT normalises row[:n] in place with generated code, at any n >= 1.
// Nothing past n is read or written.
func Softmax32JIT(row []float32, n int) {
	e := elemFor()
	if n <= 0 {
		return
	}
	if len(row) < n {
		lengthPanic("softmax", n, len(row))
	}
	args := cpu.Args{
		Out:  &row[0],
		Scr:  (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	}
	e.softmax.Call(&args)
}

// LogSoftmax32JIT replaces row[:n] with its log-softmax, x - max -
// ln(sum exp(x - max)), with generated code, at any n >= 1. Nothing past n is
// read or written.
func LogSoftmax32JIT(row []float32, n int) {
	e := elemFor()
	if n <= 0 {
		return
	}
	if len(row) < n {
		lengthPanic("logsoftmax", n, len(row))
	}
	args := cpu.Args{
		Out:  &row[0],
		Scr:  (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	}
	e.logSm.Call(&args)
}

// ActMul32JIT computes dst = act(dst) * up with generated code -- for
// ActSwiGLUOAI, the clamped form its kind describes.
func ActMul32JIT(dst, up []float32, k ActKind) {
	e := elemFor()
	i := gatedIndex(k)
	n := len(dst)
	if i < 0 {
		panic(fmt.Sprintf("jit: %v has no gated kernel", k))
	}
	if n == 0 {
		return
	}
	if len(up) < n {
		lengthPanic("actmul", n, len(up))
	}
	args := cpu.Args{
		Out:    &dst[0],
		AScale: &up[0],
		Scr:    (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:      int64(n / cpu.ElemLanes),
		Rows:   int64(n % cpu.ElemLanes),
	}
	e.actMul[i].Call(&args)
}

// Ln32JIT replaces x with ln(x), every element, with generated code; x must be
// positive and finite (a log-mel feature with its floor added).
func Ln32JIT(x []float32) {
	e := elemFor()
	n := len(x)
	if n == 0 {
		return
	}
	args := cpu.Args{
		Out:  &x[0],
		Scr:  (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	}
	e.ln.Call(&args)
}

// SigmoidMul32JIT computes gate[i] = sigma(gate[i]) * val[i] with generated
// code, in place in gate. Note the gate is the in-place argument and val is
// what it gates.
func SigmoidMul32JIT(gate, val []float32) {
	e := elemFor()
	n := len(gate)
	if n == 0 {
		return
	}
	if len(val) < n {
		lengthPanic("sigmoidmul", n, len(val))
	}
	args := cpu.Args{
		Out:    &gate[0],
		AScale: &val[0],
		Scr:    (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:      int64(n / cpu.ElemLanes),
		Rows:   int64(n % cpu.ElemLanes),
	}
	e.sigMul.Call(&args)
}

// Axpy32JIT computes dst[i] += alpha*src[i] with generated code: the residual
// adds, the biases and the MoE's weighted accumulate.
func Axpy32JIT(dst, src []float32, alpha float32) {
	e := elemFor()
	n := len(dst)
	if n == 0 {
		return
	}
	if len(src) < n {
		lengthPanic("axpy", n, len(src))
	}
	p := &one[0]
	if alpha != 1 {
		p = &alpha
	}
	args := cpu.Args{
		Out:    &dst[0],
		AScale: &src[0],
		Scr:    (*byte)(unsafe.Pointer(p)),
		K:      int64(n / cpu.ElemLanes),
		Rows:   int64(n % cpu.ElemLanes),
	}
	e.axpy.Call(&args)
}

// Scale32JIT computes dst[i] *= alpha on generated code.
func Scale32JIT(dst []float32, alpha float32) {
	e := elemFor()
	n := len(dst)
	if n == 0 {
		return
	}
	args := cpu.Args{
		Out:  &dst[0],
		Scr:  (*byte)(unsafe.Pointer(&alpha)),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	}
	e.scale.Call(&args)
}

// Softcap32JIT applies x = c*tanh(x/c) in place with generated code. It is a
// no-op at c == 0, which is every model but gemma2's.
func Softcap32JIT(x []float32, c float32) {
	if c == 0 || len(x) == 0 {
		return
	}
	e := elemFor()
	kc := [2]float32{2 / c, c}
	n := len(x)
	e.softcap.Call(&cpu.Args{
		Out:  &x[0],
		W:    (*byte)(unsafe.Pointer(&kc[0])),
		Scr:  (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	})
}

// Clamp32JIT clamps x to [-c, c] in place with generated code: DBRX's
// clip_qkv on q, k and v. A no-op at c == 0, which is every model but DBRX.
func Clamp32JIT(x []float32, c float32) {
	if c == 0 || len(x) == 0 {
		return
	}
	e := elemFor()
	lh := [2]float32{-c, c}
	n := len(x)
	e.clamp.Call(&cpu.Args{
		Out:  &x[0],
		Scr:  (*byte)(unsafe.Pointer(&lh[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	})
}

// ClampRange32JIT clamps x to [lo, hi] in place with generated code: Gemma 4's
// clipped linears, whose bounds need not be symmetric. The kernel is
// Clamp32JIT's, which takes the two bounds.
func ClampRange32JIT(x []float32, lo, hi float32) {
	if len(x) == 0 {
		return
	}
	e := elemFor()
	lh := [2]float32{lo, hi}
	n := len(x)
	e.clamp.Call(&cpu.Args{
		Out:  &x[0],
		Scr:  (*byte)(unsafe.Pointer(&lh[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	})
}

// ActKind is which activation to apply, re-exported from the assembler so
// model/ need not import it to name one.
type ActKind = cpu.ActKind

const (
	ActSiLU    = cpu.ActSiLU
	ActGELU    = cpu.ActGELU
	ActGELUErf = cpu.ActGELUErf
	// ActSqrtSigmoid is Qwen3.8-Flash-Next's n-gram gate.
	ActSqrtSigmoid = cpu.ActSqrtSigmoid
	ActQuickGELU   = cpu.ActQuickGELU
	ActReLU2       = cpu.ActReLU2
	ActReLU        = cpu.ActReLU
	ActIdentity    = cpu.ActIdentity
	ActSwiGLUOAI   = cpu.ActSwiGLUOAI
	ActXIELU       = cpu.ActXIELU
	// ActSwiGLUClamp and ActSqrtSoftplus are DeepSeek V4's expert activation
	// and router gate.
	ActSwiGLUClamp  = cpu.ActSwiGLUClamp
	ActSqrtSoftplus = cpu.ActSqrtSoftplus
	// ActSitu is Kimi-K3's activation in every FFN.
	ActSitu = cpu.ActSitu
)

// ungatedIndex is k's slot in elemSet.act, or -1 for a kind with no ungated
// kernel. The slot is k's position in cpu.Ungated, not its code.
func ungatedIndex(k ActKind) int {
	for i, g := range cpu.Ungated {
		if g == k {
			return i
		}
	}
	return -1
}

// gatedIndex is k's slot in elemSet.actMul, or -1 for a kind with no gated kernel.
func gatedIndex(k ActKind) int {
	for i, g := range cpu.Gated {
		if g == k {
			return i
		}
	}
	return -1
}

// ElemLanes is cpu.ElemLanes, re-exported so a caller can split a vector on a
// kernel boundary without importing the assembler.
const ElemLanes = cpu.ElemLanes

// Add32JIT computes dst[i] += src[i].
func Add32JIT(dst, src []float32) { Axpy32JIT(dst, src, 1) }

// Act32JIT applies the activation in place, with no gate to multiply by.
func Act32JIT(dst []float32, k ActKind) {
	e := elemFor()
	n := len(dst)
	i := ungatedIndex(k)
	if i < 0 {
		panic(fmt.Sprintf("jit: %v has no ungated kernel", k))
	}
	if n == 0 {
		return
	}
	args := cpu.Args{
		Out:  &dst[0],
		Scr:  (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:    int64(n / cpu.ElemLanes),
		Rows: int64(n % cpu.ElemLanes),
	}
	e.act[i].Call(&args)
}

// XIELU32JIT applies Apertus's xIELU to dst in place with generated code. p
// is the block's four numbers in their effective form -- alpha_p, alpha_n,
// beta, eps -- as the converter writes them (jlm.RoleXIELU);
// cpu/xielu_const.go has the arithmetic.
func XIELU32JIT(dst, p []float32) {
	if len(p) < 4 {
		lengthPanic("xielu parameters", 4, len(p))
	}
	n := len(dst)
	if n == 0 {
		return
	}
	elemFor().xielu.Call(&cpu.Args{
		Out:    &dst[0],
		AScale: &p[0],
		Scr:    (*byte)(unsafe.Pointer(&xieluConsts[0])),
		K:      int64(n / cpu.ElemLanes),
		Rows:   int64(n % cpu.ElemLanes),
	})
}

// LayerNorm32JIT computes y = (x-mean)/sqrt(var+eps)*w + b with generated code.
//
// b may be nil, which selects a kernel that does not read it -- the bias is a
// property of the file, so it is baked rather than branched on per element.
func LayerNorm32JIT(y, x, w, b []float32, eps float64) {
	n := len(x)
	if n == 0 {
		return
	}
	if len(y) < n || len(w) < n || (b != nil && len(b) < n) {
		lengthPanic("layernorm", n, min(len(y), len(w)))
	}
	c, kc := lnCode(lnKey{cpu.HostTier(), n, b != nil, cpu.RowLayerNorm}, lnKons{n, eps, 1})
	args := cpu.Args{
		Out:    &y[0],
		A:      (*int8)(unsafe.Pointer(&x[0])),
		AScale: &w[0],
		Scr:    (*byte)(unsafe.Pointer(&kc[0])),
		K:      int64(n),
	}
	if b != nil {
		args.W = (*byte)(unsafe.Pointer(&b[0]))
	}
	c.Call(&args)
}

// SoftmaxPad is cpu.SoftmaxPad, re-exported so a caller can size a score buffer
// without importing the assembler.
func SoftmaxPad(n int) int { return cpu.SoftmaxPad(n) }

// RMSNorm32JIT computes y = x/sqrt(mean(x^2)+eps) * w with generated code, at
// any width: the kernel bakes n, and the last n%lanes elements with it. It
// cannot decline.
func RMSNorm32JIT(y, x, w []float32, eps float64) {
	n := len(x)
	if n == 0 {
		return
	}
	if len(y) < n || len(w) < n {
		lengthPanic("rmsnorm", n, min(len(y), len(w)))
	}
	k := normKey{n, eps}
	tn := tierN{cpu.HostTier(), n}
	sn := norms.snap.Load()
	c, ok := sn.code[tn]
	kc, ok2 := sn.konst[k]
	if !ok || !ok2 {
		c, kc = normFill(tn, k)
	}
	args := cpu.Args{
		Out: &y[0],
		// Args.A means "the activations"; its Go type is *int8 because every
		// other kernel quantizes them. Still a real Go pointer, so the GC
		// scans it -- the same contract the F32 matvec uses.
		A:      (*int8)(unsafe.Pointer(&x[0])),
		AScale: &w[0],
		Scr:    (*byte)(unsafe.Pointer(&kc[0])),
		K:      int64(n),
	}
	c.Call(&args)
}

// normFill is the slow path: emit what is missing and publish a new snapshot.
//
// Copy-on-write rather than mutating the live maps, because a reader may be
// indexing them without the lock. The maps are tiny and this runs a handful of
// times per process.
func normFill(tn tierN, k normKey) (*cpu.Code, []float32) {
	n := tn.n
	norms.mu.Lock()
	defer norms.mu.Unlock()
	old := norms.snap.Load()
	c, okc := old.code[tn]
	kc, okk := old.konst[k]
	if okc && okk {
		return c, kc // another goroutine filled it between the load and the lock
	}
	next := &normSnap{
		code:  make(map[tierN]*cpu.Code, len(old.code)+1),
		konst: make(map[normKey][]float32, len(old.konst)+1),
	}
	for kk, vv := range old.code {
		next.code[kk] = vv
	}
	for kk, vv := range old.konst {
		next.konst[kk] = vv
	}
	if !okc {
		c = mustEmit("rmsnorm")(cpu.EmittersFor(tn.t).RMSNorm(n))
		next.code[tn] = c
	}
	if !okk {
		kc = []float32{1 / float32(n), float32(k.eps), 1}
		next.konst[k] = kc
	}
	norms.snap.Store(next)
	return c, kc
}

// lnCode looks a row-statistic kernel and its constants up, emitting what is
// missing.
func lnCode(k lnKey, nk lnKons) (*cpu.Code, []float32) {
	sn := lns.snap.Load()
	c, ok := sn.code[k]
	kc, ok2 := sn.kons[nk]
	if !ok || !ok2 {
		c, kc = lnFill(k.n, k, nk)
	}
	return c, kc
}

// GaussTopK32JIT is Gemma 3n's activation sparsity on one gate row, in place:
// x = max(x - (mean(x) + c*std(x)), 0), the population standard deviation,
// with generated code.
func GaussTopK32JIT(x []float32, c float32) {
	n := len(x)
	if n == 0 {
		return
	}
	code, kc := lnCode(lnKey{cpu.HostTier(), n, false, cpu.RowGauss}, lnKons{n, 0, float64(c)})
	code.Call(&cpu.Args{
		Out: &x[0],
		A:   (*int8)(unsafe.Pointer(&x[0])),
		Scr: (*byte)(unsafe.Pointer(&kc[0])),
		K:   int64(n),
	})
}

// MagMatch32JIT is AltUp's magnitude match: y = x * rms(ref) /
// sqrt(max(mean(x^2), eps)), with generated code. y may be x.
func MagMatch32JIT(y, x, ref []float32, eps float64) {
	n := len(x)
	if n == 0 {
		return
	}
	if len(y) < n || len(ref) < n {
		lengthPanic("magmatch", n, min(len(y), len(ref)))
	}
	code, kc := lnCode(lnKey{cpu.HostTier(), n, false, cpu.RowMag}, lnKons{n, eps, 1})
	code.Call(&cpu.Args{
		Out:    &y[0],
		A:      (*int8)(unsafe.Pointer(&x[0])),
		AScale: &ref[0],
		Scr:    (*byte)(unsafe.Pointer(&kc[0])),
		K:      int64(n),
	})
}

// RowStd32JIT standardises x over its unbiased deviation, with generated code:
// y = (x - mean) / (sqrt(sum((x - mean)^2)/(n-1)) + eps), a one-element row's
// deviation 0. NeMo's per-feature normalisation (d1-omni's audio front end).
// y may be x.
func RowStd32JIT(y, x []float32, eps float64) {
	n := len(x)
	if n == 0 {
		return
	}
	if len(y) < n {
		lengthPanic("rowstd", n, len(y))
	}
	code, kc := lnCode(lnKey{cpu.HostTier(), n, false, cpu.RowStd}, lnKons{n, eps, 1})
	code.Call(&cpu.Args{
		Out: &y[0],
		A:   (*int8)(unsafe.Pointer(&x[0])),
		Scr: (*byte)(unsafe.Pointer(&kc[0])),
		K:   int64(n),
	})
}

// lnFill is normFill for LayerNorm; see its comment.
func lnFill(n int, k lnKey, nk lnKons) (*cpu.Code, []float32) {
	lns.mu.Lock()
	defer lns.mu.Unlock()
	old := lns.snap.Load()
	c, okc := old.code[k]
	kc, okk := old.kons[nk]
	if okc && okk {
		return c, kc
	}
	next := &lnSnap{
		code: make(map[lnKey]*cpu.Code, len(old.code)+1),
		kons: make(map[lnKons][]float32, len(old.kons)+1),
	}
	for kk, vv := range old.code {
		next.code[kk] = vv
	}
	for kk, vv := range old.kons {
		next.kons[kk] = vv
	}
	if !okc {
		em := cpu.EmittersFor(k.t)
		if k.mode == cpu.RowLayerNorm {
			c = mustEmit("layernorm")(em.LayerNorm(n, k.bias))
		} else {
			c = mustEmit("rowstat")(em.RowStat(n, k.mode))
		}
		next.code[k] = c
	}
	if !okk {
		// The fourth is 1/(n-1), RowStd's (0 for one element).
		var unb float32
		if n > 1 {
			unb = 1 / float32(n-1)
		}
		kc = []float32{1 / float32(n), float32(nk.eps), float32(nk.c), unb}
		next.kons[nk] = kc
	}
	lns.snap.Store(next)
	return c, kc
}
