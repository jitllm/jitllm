//go:build amd64 || arm64

package nn

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// AddDelta generates the gated delta rule for one state width. It is
// provisioned per model, like AddAttn: the width is constant while a model is
// loaded, so it is baked rather than re-read per head.
func (f *JIT) AddDelta(n int) {
	if f.delta != nil {
		return
	}
	f.delta, f.deltaN = mustEmit("gated_delta_n"+itoaN(n))(f.em.GatedDelta(n)), n
}

// AddDeltaChan generates the per-channel decay twin for one state width.
func (f *JIT) AddDeltaChan(n int) {
	if f.deltaC != nil {
		return
	}
	f.deltaC, f.deltaN = mustEmit("gated_delta_chan_n"+itoaN(n))(f.em.GatedDeltaChan(n)), n
}

// GatedDeltaChan is GatedDelta with a per-channel decay: the row is scaled by
// decay[i] rather than by one rate for the head. decay is n floats and is the
// same for every row, because it varies along the key dimension, the
// within-row index here (oracle.GatedDeltaChan has why).
//
// state is read and written.
func (f *JIT) GatedDeltaChan(o, state, k, q, v, decay []float32, gate float32, n int) {
	if f.deltaN != n {
		panic(fmt.Sprintf("jit: GatedDeltaChan at width %d, provisioned for %d", n, f.deltaN))
	}
	if len(o) < n || len(state) < n*n || len(k) < n || len(q) < n || len(v) < n || len(decay) < n {
		lengthPanic("gated_delta_chan", n, min(len(o), len(k), len(q), len(v), len(decay)))
	}
	sc := [2]float32{0, gate}
	args := cpu.Args{
		Out:     &o[0],
		W:       (*byte)(unsafe.Pointer(&state[0])),
		AScale:  &k[0],
		Rows:    int64(n),
		Scr:     (*byte)(unsafe.Pointer(&sc[0])),
		Q32:     &q[0],
		Q2:      &v[0],
		AScale2: &decay[0],
	}
	f.deltaC.Call(&args)
}

// GatedDelta runs one head of the delta rule: n rows of the state are decayed,
// dotted with the key, rank-one updated and dotted with the query.
//
//	state[j] *= decay
//	d         = (v[j] - dot(state[j], k)) * gate
//	state[j] += d*k
//	o[j]      = dot(state[j], q)
//
// state is read and written.
func (f *JIT) GatedDelta(o, state, k, q, v []float32, decay, gate float32, n int) {
	if f.deltaN != n {
		panic(fmt.Sprintf("jit: GatedDelta at width %d, provisioned for %d", n, f.deltaN))
	}
	if len(o) < n || len(state) < n*n || len(k) < n || len(q) < n || len(v) < n {
		lengthPanic("gated_delta", n, min(len(o), len(k), len(q), len(v)))
	}
	sc := [2]float32{decay, gate}
	args := cpu.Args{
		Out:    &o[0],
		W:      (*byte)(unsafe.Pointer(&state[0])),
		AScale: &k[0],
		Rows:   int64(n),
		Scr:    (*byte)(unsafe.Pointer(&sc[0])),
		Q32:    &q[0],
		Q2:     &v[0],
	}
	f.delta.Call(&args)
}

// AddConv1d generates the causal convolution for one (taps, channels).
//
// The weights it expects are plane-major, not the source's layout, and the
// caller transposes them once at load; see cpu.EmitConv1d for why.
func (f *JIT) AddConv1d(taps, chans int) {
	if f.conv != nil {
		return
	}
	f.conv = mustEmit("conv1d_t" + itoaN(taps) + "_c" + itoaN(chans))(f.em.Conv1d(taps, chans))
	f.convTaps, f.convChans = taps, chans
}

// Conv1d reduces the state planes and the new column against the transposed
// weights, writes the result into out and advances the state by one position.
// out may alias x. state is read and written.
func (f *JIT) Conv1d(out, state, wT, x []float32, taps, chans int) {
	if f.convTaps != taps || f.convChans != chans {
		panic(fmt.Sprintf("jit: Conv1d at (%d, %d), provisioned for (%d, %d)", taps, chans, f.convTaps, f.convChans))
	}
	if len(out) < chans || len(x) < chans || len(state) < (taps-1)*chans || len(wT) < taps*chans {
		lengthPanic("conv1d", chans, min(len(out), len(x)))
	}
	args := cpu.Args{
		Out:    &out[0],
		W:      (*byte)(unsafe.Pointer(&state[0])),
		AScale: &wT[0],
		Q32:    &x[0],
	}
	f.conv.Call(&args)
}

// gateOnce guards the gate kernel: unlike the delta rule and the convolution it
// is shape-free -- the vector count is a runtime argument -- so one mapping
// serves every model, per tier (see elemSet for why the tier is in the key).
var (
	gateOnce   tierOnce
	gateK      [cpu.NumTiers]*cpu.Code
	gateConsts = cpu.DeltaGateConsts()
)

// DeltaGate32JIT computes both of a linear block's gates with generated code:
//
//	decay[i] = exp(A[i] * softplus(alpha[i] + dt[i]))
//	beta[i]  = sigmoid(b[i])
//
// They are one call because ssm_ba produces alpha and beta together and the
// same head consumes both.
func DeltaGate32JIT(decay, beta, alpha, dt, b, a []float32) {
	tier := cpu.HostTier()
	gateOnce.do(tier, func() {
		gateK[tier] = mustEmit("delta_gate")(cpu.EmittersFor(tier).DeltaGate())
	})
	n := len(decay)
	if n == 0 {
		return
	}
	if len(beta) < n || len(alpha) < n || len(dt) < n || len(b) < n || len(a) < n {
		lengthPanic("delta_gate", n, min(len(beta), len(alpha), len(dt), len(b), len(a)))
	}
	args := cpu.Args{
		Out:     &decay[0],
		Out2:    &beta[0],
		W:       (*byte)(unsafe.Pointer(&alpha[0])),
		AScale:  &dt[0],
		AScale2: &b[0],
		Q32:     &a[0],
		Scr:     (*byte)(unsafe.Pointer(&gateConsts[0])),
		K:       int64(n / cpu.ElemLanes),
		Rows:    int64(n % cpu.ElemLanes),
	}
	gateK[tier].Call(&args)
}

var (
	boundOnce tierOnce
	boundK    [cpu.NumTiers]*cpu.Code
)

// DeltaDecayBound32JIT computes Kimi-K3's bounded KDA decay with generated
// code:
//
//	decay[i] = exp(lb * sigmoid(-A[i] * (alpha[i] + dt[i])))
//
// fla's lower_bound gate, A being -exp(A_log); beta is DeltaGate32JIT's.
func DeltaDecayBound32JIT(decay, alpha, dt, a []float32, lb float32) {
	tier := cpu.HostTier()
	boundOnce.do(tier, func() {
		boundK[tier] = mustEmit("delta_decay_bound")(cpu.EmittersFor(tier).DeltaDecayBound())
	})
	n := len(decay)
	if n == 0 {
		return
	}
	if len(alpha) < n || len(dt) < n || len(a) < n {
		lengthPanic("delta_decay_bound", n, min(len(alpha), len(dt), len(a)))
	}
	args := cpu.Args{
		Out:     &decay[0],
		W:       (*byte)(unsafe.Pointer(&alpha[0])),
		AScale:  &dt[0],
		AScale2: &lb,
		Q32:     &a[0],
		Scr:     (*byte)(unsafe.Pointer(&gateConsts[0])),
		K:       int64(n / cpu.ElemLanes),
		Rows:    int64(n % cpu.ElemLanes),
	}
	boundK[tier].Call(&args)
}
