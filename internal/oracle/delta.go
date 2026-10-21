package oracle

import "math"

// The hybrid's three recurrent ops, in float64: the reference the generated
// gated delta rule, causal convolution and gates are gated against on every
// tier. The kernels are jit_old/cpu's EmitGatedDelta, EmitConv1d and EmitDeltaGate
// and their SSE twins; the Args layout they take is documented there.

// GatedDelta applies one token of the gated delta rule to one head, n = len(o)
// rows of n, in place on state (n*n, row-major), and writes the head's output.
// Per row j:
//
//	S[j] *= decay
//	sk    = dot(S[j], k)
//	d     = (v[j] - sk) * gate
//	S[j] += d*k
//	o[j]  = dot(S[j], q)
func GatedDelta(o, state, k, q, v []float64, decay, gate float64) {
	n := len(o)
	for j := 0; j < n; j++ {
		row := state[j*n : (j+1)*n]
		var sk float64
		for i := range row {
			row[i] *= decay
			sk += row[i] * k[i]
		}
		d := (v[j] - sk) * gate
		var acc float64
		for i := range row {
			row[i] += d * k[i]
			acc += row[i] * q[i]
		}
		o[j] = acc
	}
}

// Conv1d is the causal convolution over plane-major state and its shift, in
// place on state ((taps-1)*chans) and into out (chans). w is the transposed
// weights, taps planes of chans, the last plane belonging to the new column x:
//
//	out[c]   = x[c]*w[taps-1][c] + sum_{t<taps-1} state[t][c]*w[t][c]
//	state[t] = state[t+1] for t < taps-2, state[taps-2] = x
func Conv1d(out, state, w, x []float64, taps, chans int) {
	for c := 0; c < chans; c++ {
		acc := x[c] * w[(taps-1)*chans+c]
		for t := 0; t < taps-1; t++ {
			acc += state[t*chans+c] * w[t*chans+c]
		}
		out[c] = acc
		for t := 0; t < taps-2; t++ {
			state[t*chans+c] = state[(t+1)*chans+c]
		}
		state[(taps-2)*chans+c] = x[c]
	}
}

// DeltaGate computes a linear block's two gates over len(decay) heads:
//
//	decay[i] = exp(A[i] * softplus(alpha[i] + dt[i]))
//	beta[i]  = sigmoid(b[i])
//
// softplus is written max(z,0) + log1p(exp(-|z|)), which is exact in float64
// for every finite z -- the textbook log1p(exp(z)) overflows above ~709 and
// needs the branch this form removes.
func DeltaGate(decay, beta, alpha, dt, b, a []float64) {
	for i := range decay {
		z := alpha[i] + dt[i]
		sp := math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z)))
		decay[i] = math.Exp(a[i] * sp)
		beta[i] = 1 / (1 + math.Exp(-b[i]))
	}
}

// GatedDeltaChan is GatedDelta with a per-channel decay: the state decays at a
// rate that varies along the row instead of one rate for the whole head. It is
// Kimi-Linear's delta rule (KDA); qwen3next's is the scalar form above.
//
// The decay varies along the key dimension, which is the within-row index i
// here (the state is [v_dim][k_dim], the transpose of the reference), so it is
// the same for every row j: the same loop as GatedDelta, loading decay[i]
// beside state[j][i].
func GatedDeltaChan(o, state, k, q, v, decay []float64, gate float64) {
	n := len(o)
	for j := 0; j < n; j++ {
		row := state[j*n : (j+1)*n]
		var sk float64
		for i := range row {
			row[i] *= decay[i]
			sk += row[i] * k[i]
		}
		d := (v[j] - sk) * gate
		var acc float64
		for i := range row {
			row[i] += d * k[i]
			acc += row[i] * q[i]
		}
		o[j] = acc
	}
}

// GatedSSD applies one token of Mamba-2's selective state update to one head:
// len(o) rows of len(k), in place on state (row-major), and writes the head's
// output. Per row j, in llama.cpp's ssm_scan order:
//
//	S[j] = S[j]*decay + (v[j]*dt)*k
//	o[j] = dot(S[j], q) + d*v[j]
//
// k is B, q is C and v is x; it is GatedDelta with the key dot and the
// subtraction removed, dt in the gate's place and a skip.
func GatedSSD(o, state, k, q, v []float64, decay, dt, d float64) {
	n := len(k)
	for j := range o {
		row := state[j*n : (j+1)*n]
		xdt := v[j] * dt
		var acc float64
		for i := range row {
			row[i] = row[i]*decay + xdt*k[i]
			acc += row[i] * q[i]
		}
		o[j] = acc + d*v[j]
	}
}

// SSDGate computes Mamba-2's two per-head scalars over len(decay) heads:
//
//	dt[i]    = softplus(alpha[i] + bias[i])
//	decay[i] = exp(A[i] * dt[i])
//
// softplus in DeltaGate's form, exact for every finite argument.
func SSDGate(decay, dt, alpha, bias, a []float64) {
	for i := range decay {
		z := alpha[i] + bias[i]
		dt[i] = math.Max(z, 0) + math.Log1p(math.Exp(-math.Abs(z)))
		decay[i] = math.Exp(a[i] * dt[i])
	}
}

// SelScan applies one token of Mamba-1's selective scan to len(o) channels,
// in place on state (row-major, len(b) per channel). Per channel j, in
// llama.cpp's ssm_scan order:
//
//	S[j][s] = S[j][s]*exp(dt[j]*A[j][s]) + (x[j]*dt[j])*B[s]
//	o[j]    = dot(S[j], C) + d[j]*x[j]
//
// It is GatedSSD with one row a head and the decay a vector over the state.
func SelScan(o, state, b, c, x, dt, a, d []float64) {
	n := len(b)
	for j := range o {
		row := state[j*n : (j+1)*n]
		xdt := x[j] * dt[j]
		var acc float64
		for i := range row {
			row[i] = row[i]*math.Exp(dt[j]*a[j*n+i]) + xdt*b[i]
			acc += row[i] * c[i]
		}
		o[j] = acc + d[j]*x[j]
	}
}
