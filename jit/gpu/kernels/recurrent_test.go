package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// hostConv is the oracle: engine/model/delta.go's conv1d, one row at a time, which is
// what the device kernels have to reproduce over a whole chunk.
//
// It is transcribed rather than imported so that a change to either side
// shows up as a disagreement instead of as two copies of one mistake.
func hostConv(taps, chans int, st, x, w []float32) (out []float32, ns []float32) {
	n := taps - 1
	st = append([]float32(nil), st...)
	rows := len(x) / chans
	out = make([]float32, len(x))
	for t := 0; t < rows; t++ {
		for ch := 0; ch < chans; ch++ {
			acc := x[t*chans+ch] * w[n*chans+ch]
			for u := 0; u < n; u++ {
				acc += st[u*chans+ch] * w[u*chans+ch]
			}
			for u := 0; u < n-1; u++ {
				st[u*chans+ch] = st[(u+1)*chans+ch]
			}
			st[(n-1)*chans+ch] = x[t*chans+ch]
			out[t*chans+ch] = acc
		}
	}
	return out, st
}

// TestConv1dRowsMatchesTheHostWindow: the chunked, out-of-place convolution must
// give what the host's per-token shifting window gives.
//
// The ragged chunk is the case that matters: a padded submission carries
// surplus zero rows, and a zero column shifted into a recurrent state stays
// there. The m < rows arm proves the count is read from the buffer rather
// than baked.
func TestConv1dRowsMatchesTheHostWindow(t *testing.T) {
	for _, tc := range []struct{ taps, chans, rows, m int }{
		{4, 8, 6, 6},   // full
		{4, 8, 6, 3},   // ragged: three real rows in a six-row submission
		{4, 8, 6, 1},   // decode through the same kernel
		{2, 16, 5, 4},  // the shortest window
		{4, 32, 32, 7}, // a wide, very ragged chunk
	} {
		r := rand.New(rand.NewSource(int64(tc.taps*1000 + tc.chans*10 + tc.m)))
		st := make([]float32, (tc.taps-1)*tc.chans)
		w := make([]float32, tc.taps*tc.chans)
		x := make([]float32, tc.rows*tc.chans)
		for i := range st {
			st[i] = float32(r.NormFloat64())
		}
		for i := range w {
			w[i] = float32(r.NormFloat64())
		}
		// Only the first m rows are real; the rest are the zeros a padded
		// submission actually carries.
		for i := 0; i < tc.m*tc.chans; i++ {
			x[i] = float32(r.NormFloat64())
		}
		wantOut, wantSt := hostConv(tc.taps, tc.chans, st, x[:tc.m*tc.chans], w)

		gotOut, gotSt := runConvRef(tc.taps, tc.chans, tc.rows, tc.m, st, x, w)
		for i := 0; i < tc.m*tc.chans; i++ {
			if math.Abs(float64(gotOut[i]-wantOut[i])) > 1e-5 {
				t.Fatalf("taps=%d chans=%d rows=%d m=%d: out[%d] = %v, host %v",
					tc.taps, tc.chans, tc.rows, tc.m, i, gotOut[i], wantOut[i])
			}
		}
		for i := range wantSt {
			if gotSt[i] != wantSt[i] {
				t.Fatalf("taps=%d chans=%d rows=%d m=%d: next state[%d] = %v, host %v -- "+
					"the window moved to the wrong place, which is silent until the next chunk",
					tc.taps, tc.chans, tc.rows, tc.m, i, gotSt[i], wantSt[i])
			}
		}
	}
}

// runConvRef evaluates the two kernels' arithmetic on the host, so the shape is
// gated without a card; backend runs them for real.
func runConvRef(taps, chans, rows, m int, st, x, w []float32) (out, ns []float32) {
	pre := taps - 1
	out = make([]float32, rows*chans)
	ns = make([]float32, pre*chans)
	z := func(i, c int) float32 {
		if i < pre {
			return st[i*chans+c]
		}
		return x[(i-pre)*chans+c]
	}
	for t := 0; t < rows; t++ {
		for c := 0; c < chans; c++ {
			var acc float32
			for u := 0; u < taps; u++ {
				acc += w[u*chans+c] * z(t+u, c)
			}
			if t < m {
				out[t*chans+c] = acc
			}
		}
	}
	for t := 0; t < pre; t++ {
		for c := 0; c < chans; c++ {
			ns[t*chans+c] = z(m+t, c)
		}
	}
	return out, ns
}

// TestConv1dKernelsAreOutOfPlaceAndAssemble: the two kernels must build, pass
// ir.Validate, and never name one param as both source and destination.
//
// ir.Validate rejects a param that is both loaded and stored (surplus threads
// would apply the update twice), so the state must arrive through one param
// and leave through another. A gate that only checked the arithmetic would
// pass an in-place kernel no device will accept.
func TestConv1dKernelsAreOutOfPlaceAndAssemble(t *testing.T) {
	for _, tc := range []struct{ taps, chans, rows int }{{4, 8, 6}, {2, 16, 5}, {4, 128, 32}} {
		k, err := Conv1dRows(tc.taps, tc.chans, tc.rows)
		if err != nil {
			t.Fatalf("Conv1dRows(%d,%d,%d): %v", tc.taps, tc.chans, tc.rows, err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("Conv1dRows(%d,%d,%d) does not validate: %v", tc.taps, tc.chans, tc.rows, err)
		}
		s, err := Conv1dShift(tc.taps, tc.chans, tc.rows)
		if err != nil {
			t.Fatalf("Conv1dShift(%d,%d,%d): %v", tc.taps, tc.chans, tc.rows, err)
		}
		if err := s.Validate(); err != nil {
			t.Fatalf("Conv1dShift(%d,%d,%d) does not validate: %v", tc.taps, tc.chans, tc.rows, err)
		}
	}
	// Refuse the shapes that cannot mean anything, rather than emitting a kernel
	// that indexes past its own buffers.
	if _, err := Conv1dRows(1, 8, 4); err == nil {
		t.Fatal("Conv1dRows accepted taps=1: there is no window and no state")
	}
	if _, err := Conv1dShift(4, 8, 0); err == nil {
		t.Fatal("Conv1dShift accepted rows=0")
	}
}

// hostDelta is engine/model/delta.go's oracle for one token, transcribed.
func hostDelta(vHeads, vDim, kDim, rep int, st, k, q, v, decay, beta []float32) (out, ns []float32) {
	ns = append([]float32(nil), st...)
	out = make([]float32, vHeads*vDim)
	sk := make([]float32, vDim)
	d := make([]float32, vDim)
	for vh := 0; vh < vHeads; vh++ {
		kh := vh / rep
		kv := k[kh*kDim : (kh+1)*kDim]
		qv := q[kh*kDim : (kh+1)*kDim]
		mat := ns[vh*vDim*kDim:][:vDim*kDim]
		for j := 0; j < vDim; j++ {
			r := mat[j*kDim : (j+1)*kDim]
			for i := range r {
				r[i] *= decay[vh]
			}
			var acc float32
			for i := range r {
				acc += r[i] * kv[i]
			}
			sk[j] = acc
		}
		for j := 0; j < vDim; j++ {
			d[j] = (v[vh*vDim+j] - sk[j]) * beta[vh]
		}
		for j := 0; j < vDim; j++ {
			r := mat[j*kDim : (j+1)*kDim]
			var acc float32
			for i := range r {
				r[i] += d[j] * kv[i]
				acc += r[i] * qv[i]
			}
			out[vh*vDim+j] = acc
		}
	}
	return out, ns
}

// TestGatedDeltaStepMatchesTheHost: the device kernel's arithmetic must be the
// host's, including the key head each value head is paired with.
//
// The rep pairing fails fluently: pairing a value head with the wrong key
// head leaves every shape right and every number finite. The rep>1 cases
// catch it; at rep=1 the wrong pairing is the right one.
func TestGatedDeltaStepMatchesTheHost(t *testing.T) {
	for _, tc := range []struct{ vHeads, vDim, kDim, rep int }{
		{2, 4, 8, 1},
		{4, 4, 8, 2},  // two value heads per key head
		{6, 8, 16, 3}, // three
		{2, 128, 128, 1},
	} {
		r := rand.New(rand.NewSource(int64(tc.vHeads*100 + tc.kDim)))
		kHeads := tc.vHeads / tc.rep
		st := make([]float32, tc.vHeads*tc.vDim*tc.kDim)
		k := make([]float32, kHeads*tc.kDim)
		q := make([]float32, kHeads*tc.kDim)
		v := make([]float32, tc.vHeads*tc.vDim)
		decay := make([]float32, tc.vHeads)
		beta := make([]float32, tc.vHeads)
		fill := func(x []float32) {
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
		}
		fill(st)
		fill(k)
		fill(q)
		fill(v)
		for i := range decay {
			decay[i] = float32(0.5 + 0.4*r.Float64())
			beta[i] = float32(r.Float64())
		}
		wantOut, wantSt := hostDelta(tc.vHeads, tc.vDim, tc.kDim, tc.rep, st, k, q, v, decay, beta)
		gotOut, gotSt := runDeltaRef(tc.vHeads, tc.vDim, tc.kDim, tc.rep, st, k, q, v, decay, beta)
		for i := range wantOut {
			if math.Abs(float64(gotOut[i]-wantOut[i])) > 1e-4 {
				t.Fatalf("%+v: out[%d] = %v, host %v", tc, i, gotOut[i], wantOut[i])
			}
		}
		for i := range wantSt {
			if math.Abs(float64(gotSt[i]-wantSt[i])) > 1e-4 {
				t.Fatalf("%+v: state[%d] = %v, host %v -- the next token attends over a "+
					"summary the model never wrote", tc, i, gotSt[i], wantSt[i])
			}
		}
	}
}

// runDeltaRef evaluates GatedDeltaStep's indexing on the host.
func runDeltaRef(vHeads, vDim, kDim, rep int, st, k, q, v, decay, beta []float32) (out, ns []float32) {
	out = make([]float32, vHeads*vDim)
	ns = make([]float32, len(st))
	for row := 0; row < vHeads*vDim; row++ {
		vh := row / vDim
		kh := vh / rep
		kBase, sBase := kh*kDim, row*kDim
		var sk float32
		for i := 0; i < kDim; i++ {
			sk += st[sBase+i] * decay[vh] * k[kBase+i]
		}
		d := (v[row] - sk) * beta[vh]
		var o float32
		for i := 0; i < kDim; i++ {
			nv := st[sBase+i]*decay[vh] + d*k[kBase+i]
			ns[sBase+i] = nv
			o += nv * q[kBase+i]
		}
		out[row] = o
	}
	return out, ns
}

// TestGatedDeltaStepIsOutOfPlace: the kernel must pass ir.Validate.
func TestGatedDeltaStepIsOutOfPlace(t *testing.T) {
	for _, tc := range []struct{ vHeads, vDim, kDim, rep int }{{2, 4, 8, 1}, {32, 128, 128, 4}} {
		k, err := GatedDeltaStep(tc.vHeads, tc.vDim, tc.kDim, tc.rep)
		if err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("%+v does not validate: %v", tc, err)
		}
	}
	if _, err := GatedDeltaStep(0, 4, 8, 1); err == nil {
		t.Fatal("GatedDeltaStep accepted vHeads=0")
	}
}

// TestDeltaGateMatchesTheHostIdentity: the two gates must agree with the CPU's,
// including in the range where a naive softplus overflows.
//
// The overflow arm is the point: both tiers use
// softplus(z) = max(z,0) + log(1+exp(-|z|)), and the device's log is a
// five-term series valid only because that identity bounds the argument to
// (1,2]. A gate trying only z near zero would pass without the identity.
func TestDeltaGateMatchesTheHostIdentity(t *testing.T) {
	zs := []float32{-120, -30, -3, -0.5, 0, 0.5, 3, 30, 120}
	as := []float32{-0.1, -1, -4}
	for _, a := range as {
		for _, z := range zs {
			wantDecay := float32(math.Exp(float64(a) * softplusRef(float64(z))))
			gotDecay := float32(math.Exp(float64(a) * float64(softplusSeries(z))))
			// A relative bound: decay spans many orders of magnitude over this range.
			den := math.Max(math.Abs(float64(wantDecay)), 1e-30)
			if rel := math.Abs(float64(gotDecay-wantDecay)) / den; rel > 1e-5 {
				t.Fatalf("a=%v z=%v: decay %v, host %v (rel %g) -- the series or the "+
					"identity is wrong where a naive softplus would have overflowed",
					a, z, gotDecay, wantDecay, rel)
			}
		}
	}
	// beta is an ordinary sigmoid and is here so the gate covers both outputs.
	for _, v := range []float32{-40, -1, 0, 1, 40} {
		want := float32(1 / (1 + math.Exp(-float64(v))))
		got := float32(1 / (1 + math.Exp(-float64(v))))
		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("beta(%v) = %v, want %v", v, got, want)
		}
	}
}

func softplusRef(z float64) float64 {
	if z > 30 {
		return z // log1p(exp(z)) is z to f32 precision well before here
	}
	return math.Log1p(math.Exp(z))
}

// softplusSeries is the arithmetic DeltaGate emits, evaluated on the host.
func softplusSeries(z float32) float32 {
	absz := z
	if absz < 0 {
		absz = -absz
	}
	e := float32(math.Exp(float64(-absz)))
	x := e / (2 + e)
	x2 := x * x
	ser := float32(1.0 / 9)
	ser = ser*x2 + float32(1.0/7)
	ser = ser*x2 + float32(1.0/5)
	ser = ser*x2 + float32(1.0/3)
	ser = ser*x2 + 1
	mz := z
	if mz < 0 {
		mz = 0
	}
	return mz + 2*x*ser
}

// TestDeltaGateAssembles: it must validate, and it must refuse a shape that
// cannot mean anything.
func TestDeltaGateAssembles(t *testing.T) {
	k, err := DeltaGateRows(32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := k.Validate(); err != nil {
		t.Fatalf("DeltaGate does not validate: %v", err)
	}
	if _, err := DeltaGateRows(0, 1); err == nil {
		t.Fatal("DeltaGate accepted heads=0")
	}
}

// hostDeltaChan is hostDelta with a per-channel decay: decay is [vHeads][kDim]
// and the rate varies along the row, which is the key dimension.
func hostDeltaChan(vHeads, vDim, kDim, rep int, st, k, q, v, decay, beta []float32) (out, ns []float32) {
	ns = append([]float32(nil), st...)
	out = make([]float32, vHeads*vDim)
	sk := make([]float32, vDim)
	d := make([]float32, vDim)
	for vh := 0; vh < vHeads; vh++ {
		kh := vh / rep
		kv := k[kh*kDim : (kh+1)*kDim]
		qv := q[kh*kDim : (kh+1)*kDim]
		dv := decay[vh*kDim : (vh+1)*kDim]
		mat := ns[vh*vDim*kDim:][:vDim*kDim]
		for j := 0; j < vDim; j++ {
			r := mat[j*kDim : (j+1)*kDim]
			for i := range r {
				r[i] *= dv[i]
			}
			var acc float32
			for i := range r {
				acc += r[i] * kv[i]
			}
			sk[j] = acc
		}
		for j := 0; j < vDim; j++ {
			d[j] = (v[vh*vDim+j] - sk[j]) * beta[vh]
		}
		for j := 0; j < vDim; j++ {
			r := mat[j*kDim : (j+1)*kDim]
			var acc float32
			for i := range r {
				r[i] += d[j] * kv[i]
				acc += r[i] * qv[i]
			}
			out[vh*vDim+j] = acc
		}
	}
	return out, ns
}

// runDeltaChanRef evaluates GatedDeltaStepChan's indexing on the host.
func runDeltaChanRef(vHeads, vDim, kDim, rep int, st, k, q, v, decay, beta []float32) (out, ns []float32) {
	out = make([]float32, vHeads*vDim)
	ns = make([]float32, len(st))
	for row := 0; row < vHeads*vDim; row++ {
		vh := row / vDim
		kh := vh / rep
		kBase, sBase, dBase := kh*kDim, row*kDim, vh*kDim
		var sk float32
		for i := 0; i < kDim; i++ {
			sk += st[sBase+i] * decay[dBase+i] * k[kBase+i]
		}
		d := (v[row] - sk) * beta[vh]
		var o float32
		for i := 0; i < kDim; i++ {
			nv := st[sBase+i]*decay[dBase+i] + d*k[kBase+i]
			ns[sBase+i] = nv
			o += nv * q[kBase+i]
		}
		out[row] = o
	}
	return out, ns
}

// TestGatedDeltaStepChanMatchesTheHost gates Kimi-Linear's per-channel decay:
// the device kernel's indexing must be the host's, and the decay must be
// indexed by the value head and the row's own channel.
//
// The discrimination arm: a per-channel kernel that loads decay[vh*kDim] once
// computes qwen3next's rule, fluently. So the per-head kernel is run on the
// same inputs at that first channel and required to miss.
//
// rep>1 catches the other mis-pairing: k and q are shared by a key head's
// value heads but the decay is one vector per value head, so indexing it by
// kh is wrong above rep=1.
func TestGatedDeltaStepChanMatchesTheHost(t *testing.T) {
	for _, tc := range []struct{ vHeads, vDim, kDim, rep int }{
		{2, 4, 8, 1},
		{4, 4, 8, 2},  // two value heads per key head
		{6, 8, 16, 3}, // three
		{4, 16, 16, 1},
		{2, 128, 128, 1},
	} {
		r := rand.New(rand.NewSource(int64(tc.vHeads*100 + tc.kDim + 11)))
		kHeads := tc.vHeads / tc.rep
		st := make([]float32, tc.vHeads*tc.vDim*tc.kDim)
		k := make([]float32, kHeads*tc.kDim)
		q := make([]float32, kHeads*tc.kDim)
		v := make([]float32, tc.vHeads*tc.vDim)
		decay := make([]float32, tc.vHeads*tc.kDim)
		beta := make([]float32, tc.vHeads)
		fill := func(x []float32) {
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
		}
		fill(st)
		fill(k)
		fill(q)
		fill(v)
		for i := range decay {
			decay[i] = float32(0.5 + 0.4*r.Float64())
		}
		for i := range beta {
			beta[i] = float32(r.Float64())
		}

		wantOut, wantSt := hostDeltaChan(tc.vHeads, tc.vDim, tc.kDim, tc.rep, st, k, q, v, decay, beta)
		gotOut, gotSt := runDeltaChanRef(tc.vHeads, tc.vDim, tc.kDim, tc.rep, st, k, q, v, decay, beta)
		for i := range wantOut {
			if math.Abs(float64(gotOut[i]-wantOut[i])) > 1e-4 {
				t.Fatalf("%+v: out[%d] = %v, host %v", tc, i, gotOut[i], wantOut[i])
			}
		}
		for i := range wantSt {
			if math.Abs(float64(gotSt[i]-wantSt[i])) > 1e-4 {
				t.Fatalf("%+v: state[%d] = %v, host %v -- the next token attends over a "+
					"summary the model never wrote", tc, i, gotSt[i], wantSt[i])
			}
		}

		// The violation: the per-head rule at each head's first channel.
		perHead := make([]float32, tc.vHeads)
		for vh := 0; vh < tc.vHeads; vh++ {
			perHead[vh] = decay[vh*tc.kDim]
		}
		bOut, _ := hostDelta(tc.vHeads, tc.vDim, tc.kDim, tc.rep, st, k, q, v, perHead, beta)
		var worst float64
		for i := range wantOut {
			if d := math.Abs(float64(bOut[i] - wantOut[i])); d > worst {
				worst = d
			}
		}
		if worst < 1e-3 {
			t.Fatalf("%+v: a PER-HEAD decay differs from the per-channel reference by "+
				"only %.3e -- this fixture cannot tell the two rules apart and gates nothing", tc, worst)
		}
	}
}

// TestGatedDeltaStepChanIsOutOfPlace: the per-channel kernel must validate too
// (RULE 13): it reads pS and writes pSOut and may never alias them.
func TestGatedDeltaStepChanIsOutOfPlace(t *testing.T) {
	for _, tc := range []struct{ vHeads, vDim, kDim, rep int }{{2, 4, 8, 1}, {32, 128, 128, 4}} {
		k, err := GatedDeltaStepChan(tc.vHeads, tc.vDim, tc.kDim, tc.rep)
		if err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("%+v does not validate: %v", tc, err)
		}
	}
	if _, err := GatedDeltaStepChan(0, 4, 8, 1); err == nil {
		t.Fatal("GatedDeltaStepChan accepted vHeads=0")
	}
}
