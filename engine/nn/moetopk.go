//go:build amd64 || arm64

package nn

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The mixture-of-experts router, generated. Like the greedy argmax it is a
// package-level kernel rather than a JIT's: the router is not a layer.
//
// Two entry points share one kernel. MoETopK32JIT is the pre-DeepSeek-V3
// router (top-k, optionally renormalise) and takes probabilities the caller
// has already gated. MoERouteJIT is V3's (also R1, Kimi-K2, GLM-4-MoE) and
// takes logits: it runs the gating kernel itself, adds the selection bias,
// keeps the best expert groups and applies routed_scaling_factor. The first is
// the second at MoEGate{Norm: norm}, byte for byte
// (TestMoETopKIsMoERouteAtItsDefault).
//
// There is no Go fallback: a tier with no kernel makes engine/model/moe.go return an
// error naming the router's shape.

// MoERoute is the set of buffers one router call fills. A caller owns one and
// reuses it, so a decode token allocates nothing here.
//
// Both orders are outputs because both are read. Sel and Wt are in selection
// order (descending probability), for everything that asks which experts were
// chosen. Ord and OWt are the same experts by ascending id, the order their
// contributions must reach the residual in: the batched path visits the bank
// expert-major, and summing in any other order gives a different float32 sum
// that can flip a near-tied selection in the next layer.
//
// Scr is per caller and must not be shared between concurrent routes: the
// kernel writes the selection through it.
type MoERoute struct {
	Sel []int32   // the chosen ids, SELECTION order
	Ord []int32   // the same ids, ASCENDING
	Wt  []float32 // Sel's probabilities, renormalised
	OWt []float32 // Ord's probabilities, renormalised
	Sum []float32 // one element: the divisor
	Scr []float32 // cpu.MoERouteScratch(n, k, gate) words of kernel scratch
	// Sc is the post-gating score of every expert (sigmoid or softmax of the
	// logits), which MoERouteJIT fills and then selects over. nil on a route
	// built by NewMoERoute, where the caller gates its own buffer. It stays
	// readable after the call so a gate can compare it to libm with a
	// tolerance (engine/nn/moev3_test.go).
	Sc []float32
	// Kons is the kernel's constant block, cpu.MoETopKConstsFor(scale). It is
	// per route because routed_scaling_factor is its only per-model value.
	Kons []float32
	// scale is what Kons was built for. MoERouteJIT refuses a gate whose scale
	// differs: the kernel takes the factor from Kons, not from the gate.
	scale float32
	// Mix is the sparsemixer kernel's constant block,
	// cpu.SparseMixerConsts(eps), for a route built under a sparsemixer gate,
	// and mixEps the eps it carries; nil and 0 otherwise.
	Mix    []float32
	mixEps float32
}

// NewMoERoute allocates a route for k selected experts, for the router every
// pre-V3 mixture uses. MoETopK32JIT is its entry point.
func NewMoERoute(k int) *MoERoute {
	if k <= 0 {
		panic(fmt.Sprintf("jit: a mixture route for k = %d", k))
	}
	return &MoERoute{
		Sel: make([]int32, k),
		Ord: make([]int32, k),
		Wt:  make([]float32, k),
		OWt: make([]float32, k),
		Sum: make([]float32, 1),
		Scr: make([]float32, cpu.MoETopKScratch(k)),
	}
}

// NewMoERouteFor allocates a route for n experts, k of them selected, under
// gate g. MoERouteJIT is its entry point, and a route built by NewMoERoute is
// not interchangeable with one built here: the V3 router needs a larger
// scratch, the score buffer and the constant block.
func NewMoERouteFor(n, k int, g MoEGate) *MoERoute {
	if k <= 0 || n <= 0 || k > n {
		panic(fmt.Sprintf("jit: a mixture route for %d of %d expert(s)", k, n))
	}
	r := &MoERoute{
		Sel:   make([]int32, k),
		Ord:   make([]int32, k),
		Wt:    make([]float32, k),
		OWt:   make([]float32, k),
		Sum:   make([]float32, 1),
		Scr:   make([]float32, cpu.MoERouteScratch(n, k, g.baked())),
		Sc:    make([]float32, n),
		Kons:  cpu.MoETopKConstsFor(g.scale()),
		scale: g.scale(),
	}
	if g.SparseMixer != 0 {
		r.Mix, r.mixEps = cpu.SparseMixerConsts(g.SparseMixer), g.SparseMixer
	}
	return r
}

// MoEGate is a router's configuration: which gating function runs over the
// logits, and everything cpu.MoEGate bakes into the kernel, with the runtime
// values (the bias plane, the scaling factor) beside their flags.
//
// The gating function is a separate generated kernel, not part of the router
// kernel's key: sigmoid and softmax already exist as elementwise kernels on
// every tier, and baking an exp into a selection loop that already holds
// fifteen of sixteen vector registers would save one pass over a few hundred
// floats.
type MoEGate struct {
	// Sigmoid selects sigma(logits) over softmax(logits).
	Sigmoid bool
	// SqrtSoftplus selects sqrt(softplus(logits)), DeepSeek V4's gate, over
	// both.
	SqrtSoftplus bool
	// Bias is e_score_correction_bias, one value per expert, or nil. It moves
	// the selection and never the returned weights. It is not the router's own
	// bias (gpt-oss's, added to the logits before gating); a model can have
	// both.
	Bias []float32
	// NGroup/NGroupUsed are the grouped top-k. 0 or 1 group is ungrouped.
	NGroup, NGroupUsed int
	// Norm renormalises the selected weights by their sum.
	Norm bool
	// Scale is routed_scaling_factor. 0 and 1 both mean no scaling, and the
	// kernel then emits no multiply.
	Scale float32
	// ExpScale is one factor per expert that multiplies its selected weight,
	// last (Gemma 4's router.per_expert_scale), or nil.
	ExpScale []float32
	// SparseMixer, when non-zero, is Phi-3.5-MoE's router: the top two of the
	// raw logits, each weighted by a softmax over the logits within a
	// relative 2*SparseMixer of it (its jitter_eps), not renormalised. It runs
	// the plain top-k kernel at Norm off and then cpu's SparseMixer weights
	// kernel; the gating, bias, groups, renormalisation and scale fields must
	// all be off (jit_old/cpu/sparsemixer_const.go has the contract).
	SparseMixer float32
}

// scale is the factor the constant block carries: never zero, so a route built
// for a model that does not scale still divides and multiplies by the same
// number the kernel would have used.
func (g MoEGate) scale() float32 {
	if g.Scale == 0 {
		return 1
	}
	return g.Scale
}

// baked is the part of the configuration that becomes code.
func (g MoEGate) baked() cpu.MoEGate {
	return cpu.MoEGate{
		Norm:       g.Norm,
		Bias:       g.Bias != nil,
		NGroup:     g.NGroup,
		NGroupUsed: g.NGroupUsed,
		Scale:      g.scale() != 1,
		ExpScale:   g.ExpScale != nil,
	}
}

type moeTopKKey struct {
	t    cpu.Tier
	n, k int
	g    cpu.MoEGate
}

// The kernels, keyed by tier (see elemSet) and by every baked parameter: the
// shape and the MoEGate that decides which phases exist. A model asks for one
// key for its whole life.
var moeTopKs struct {
	mu   sync.Mutex
	snap atomic.Pointer[map[moeTopKKey]*cpu.Code]
}

var moeTopKConsts = cpu.MoETopKConsts()

// moeRouteKonsLen is how many words a route's constant block must hold. The
// check is on every router call, and building the block to measure it was two
// allocations a layer.
var moeRouteKonsLen = len(cpu.MoETopKConstsFor(1))

func init() {
	m := map[moeTopKKey]*cpu.Code{}
	moeTopKs.snap.Store(&m)
}

func moeRouteFor(n, k int, g cpu.MoEGate) *cpu.Code {
	key := moeTopKKey{cpu.HostTier(), n, k, g}
	if c, ok := (*moeTopKs.snap.Load())[key]; ok {
		return c
	}
	return moeTopKFill(key)
}

// moeTopKFill emits and maps one kernel under the lock and publishes a new
// snapshot. A failure is cached as a nil entry rather than retried: an emitter
// that refuses this shape refuses it every token.
func moeTopKFill(key moeTopKKey) *cpu.Code {
	moeTopKs.mu.Lock()
	defer moeTopKs.mu.Unlock()
	old := *moeTopKs.snap.Load()
	if c, ok := old[key]; ok {
		return c
	}
	var c *cpu.Code
	if b, err := cpu.EmittersFor(key.t).MoERoute(key.n, key.k, key.g); err == nil {
		if mapped, err := cpu.MapNamed(b, "moe_topk"); err == nil {
			c = mapped
		}
	}
	next := make(map[moeTopKKey]*cpu.Code, len(old)+1)
	for kk, vv := range old {
		next[kk] = vv
	}
	next[key] = c
	moeTopKs.snap.Store(&next)
	return c
}

// MoETopK32JIT selects the len(r.Sel) largest of p and fills r.
//
// p is the softmax over all experts, which is what llama.cpp's running graph
// computes before the top-k and what the weights were trained with. norm is the
// architecture's renormalisation: true divides by the sum of the chosen
// probabilities (llama, qwen3moe, qwen3next), false leaves the divisor at 1
// (olmoe), and x/1 is x bit for bit, so the two arms share one kernel shape.
//
// It reports false when this tier has no kernel for the shape; the caller
// returns an error naming the router.
func MoETopK32JIT(p []float32, norm bool, r *MoERoute) bool {
	k := len(r.Sel)
	n := len(p)
	if k == 0 || n == 0 || k > n {
		return false
	}
	if len(r.Ord) < k || len(r.Wt) < k || len(r.OWt) < k || len(r.Sum) < 1 ||
		len(r.Scr) < cpu.MoETopKScratch(k) {
		panic(fmt.Sprintf("jit: a mixture route for k = %d handed %d ids, %d weights and %d scratch words",
			k, len(r.Ord), len(r.OWt), len(r.Scr)))
	}
	c := moeRouteFor(n, k, cpu.MoEGate{Norm: norm})
	if c == nil {
		return false
	}
	args := cpu.Args{
		Q32:      &p[0],
		Scr:      (*byte)(unsafe.Pointer(&moeTopKConsts[0])),
		Scratch:  (*byte)(unsafe.Pointer(&r.Scr[0])),
		ASum:     &r.Sel[0],
		AHalfSum: &r.Ord[0],
		Out:      &r.Wt[0],
		Out2:     &r.OWt[0],
		AScale:   &r.Sum[0],
	}
	c.Call(&args)
	return true
}

// The gating kernel, keyed by tier alone: sigma(x) over a runtime length, which
// is cpu.Emitters.Act at kernels.ActSigmoid.
var moeSigmoids struct {
	mu   sync.Mutex
	snap atomic.Pointer[map[cpu.Tier]*cpu.Code]
}

func init() {
	m := map[cpu.Tier]*cpu.Code{}
	moeSigmoids.snap.Store(&m)
}

func moeSigmoidFor(t cpu.Tier) *cpu.Code {
	if c, ok := (*moeSigmoids.snap.Load())[t]; ok {
		return c
	}
	moeSigmoids.mu.Lock()
	defer moeSigmoids.mu.Unlock()
	old := *moeSigmoids.snap.Load()
	if c, ok := old[t]; ok {
		return c
	}
	var c *cpu.Code
	if b, err := cpu.EmittersFor(t).Act(kernels.ActSigmoid); err == nil {
		if mapped, err := cpu.MapNamed(b, "moe_sigmoid"); err == nil {
			c = mapped
		}
	}
	next := make(map[cpu.Tier]*cpu.Code, len(old)+1)
	for kk, vv := range old {
		next[kk] = vv
	}
	next[t] = c
	moeSigmoids.snap.Store(&next)
	return c
}

// MoERouteJIT runs the whole router over the LOGITS: the gating function, the
// selection bias, the grouped top-k, the renormalisation and the scaling
// factor, as two generated kernels and no Go arithmetic.
//
// It takes logits where MoETopK32JIT takes probabilities: a caller must not
// gate first. The gated scores are left in r.Sc.
//
// The bias is added for selection only; the weights in r.Wt and r.OWt come from
// the unbiased scores, as in DeepSeek's reference and llama.cpp.
//
// It reports false when this tier has no kernel for the shape; the caller
// returns an error naming the router.
func MoERouteJIT(logits []float32, g MoEGate, r *MoERoute) bool {
	k, n := len(r.Sel), len(logits)
	if k == 0 || n == 0 || k > n {
		return false
	}
	bg := g.baked()
	if len(r.Ord) < k || len(r.Wt) < k || len(r.OWt) < k || len(r.Sum) < 1 ||
		len(r.Sc) < n || len(r.Kons) < moeRouteKonsLen ||
		len(r.Scr) < cpu.MoERouteScratch(n, k, bg) {
		panic(fmt.Sprintf("jit: a mixture route for %d of %d expert(s) handed %d score(s), "+
			"%d constant(s) and %d scratch word(s) -- NewMoERouteFor(%d, %d, gate) is the constructor",
			k, n, len(r.Sc), len(r.Kons), len(r.Scr), n, k))
	}
	if r.scale != g.scale() {
		panic(fmt.Sprintf("jit: a mixture route built for routed scale %g called with %g -- "+
			"the constant block carries the constructor's", r.scale, g.scale()))
	}
	if bg.Bias && len(g.Bias) < n {
		lengthPanic("moe router bias", n, len(g.Bias))
	}
	if bg.ExpScale && len(g.ExpScale) < n {
		lengthPanic("moe per-expert scale", n, len(g.ExpScale))
	}
	if g.SparseMixer != 0 {
		return sparseMixerRoute(logits, g, r)
	}
	c := moeRouteFor(n, k, bg)
	if c == nil {
		return false
	}

	// The gating function, generated. The copy is a memmove and not arithmetic:
	// both kernels write in place and the logits are the caller's buffer.
	copy(r.Sc[:n], logits)
	if g.SqrtSoftplus {
		Act32JIT(r.Sc[:n], ActSqrtSoftplus)
	} else if g.Sigmoid {
		sig := moeSigmoidFor(cpu.HostTier())
		if sig == nil {
			return false
		}
		sig.Call(&cpu.Args{
			Out:  &r.Sc[0],
			Scr:  (*byte)(unsafe.Pointer(&elemConsts[0])),
			K:    int64(n / cpu.ElemLanes),
			Rows: int64(n % cpu.ElemLanes),
		})
	} else {
		Softmax32JIT(r.Sc, n)
	}

	args := cpu.Args{
		Q32:      &r.Sc[0],
		Scr:      (*byte)(unsafe.Pointer(&r.Kons[0])),
		Scratch:  (*byte)(unsafe.Pointer(&r.Scr[0])),
		ASum:     &r.Sel[0],
		AHalfSum: &r.Ord[0],
		Out:      &r.Wt[0],
		Out2:     &r.OWt[0],
		AScale:   &r.Sum[0],
	}
	if bg.Bias {
		args.Q2 = &g.Bias[0]
	}
	if bg.ExpScale {
		args.AScale2 = &g.ExpScale[0]
	}
	c.Call(&args)
	return true
}
