//go:build amd64 || arm64

package cpu

import (
	"math"
	"runtime"
	"testing"
	"unsafe"
)

// hcMixRef is the mixer in float64, transformers' DeepseekV4HyperConnection
// arithmetic (hc_const.go).
func hcMixRef(mix, scale, base []float32, eps float64, iters int, head bool) []float64 {
	h := HCStreams
	sig := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	af := func(i int) float64 { return float64(mix[i])*float64(scale[i]) + float64(base[i]) }
	out := make([]float64, (2+h)*h)
	for i := 0; i < h; i++ {
		out[i] = sig(af(i)) + eps
		out[h+i] = 2 * sig(af(h+i))
	}
	if head {
		return out[:h]
	}
	cm := out[2*h:]
	for i := 0; i < h; i++ {
		mx, sum := math.Inf(-1), 0.0
		for j := 0; j < h; j++ {
			mx = math.Max(mx, af(2*h+i*h+j))
		}
		for j := 0; j < h; j++ {
			cm[i*h+j] = math.Exp(af(2*h+i*h+j) - mx)
			sum += cm[i*h+j]
		}
		for j := 0; j < h; j++ {
			cm[i*h+j] = cm[i*h+j]/sum + eps
		}
	}
	norm := func(cols bool) {
		for a := 0; a < h; a++ {
			sum := 0.0
			at := func(b int) *float64 {
				if cols {
					return &cm[b*h+a]
				}
				return &cm[a*h+b]
			}
			for b := 0; b < h; b++ {
				sum += *at(b)
			}
			for b := 0; b < h; b++ {
				*at(b) /= sum + eps
			}
		}
	}
	if iters > 0 {
		norm(true)
		for it := 1; it < iters; it++ {
			norm(false)
			norm(true)
		}
	}
	return out
}

// hcTiers is every tier this host can execute the two kernels on: the host
// tier, and the SSE tier wherever it is a second one (on an SSE-only host the
// host tier is it).
func hcTiers() []*Emitters {
	ts := []*Emitters{hostTable()}
	if runtime.GOARCH == "amd64" && HostTier() != TierSSE {
		ts = append(ts, EmittersFor(TierSSE))
	}
	return ts
}

// TestHCMixMatchesReference holds the mixer to float64 on every tier this
// host runs, at the rounds DeepSeek V4 ships (20), one round and none, with
// logits wide enough that the softmax is far from uniform. The mixer's output
// must be the reference's, and the rounds must move it (a kernel that ignored
// them would pass every count but one).
func TestHCMixMatchesReference(t *testing.T) {
	const eps = 1e-6
	n := (2 + HCStreams) * HCStreams
	mix, scale, base := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range mix {
		mix[i] = float32(math.Sin(float64(i)*1.37) * 4)
		scale[i] = float32(1 + 0.3*math.Cos(float64(i)))
		base[i] = float32(0.5 * math.Sin(float64(i)*0.7))
	}
	kons := HCMixConsts(eps)
	for _, em := range hcTiers() {
		var prev []float32
		for _, iters := range []int{20, 1, 0} {
			code, err := em.HCMix(iters, false)
			if err != nil {
				t.Fatal(err)
			}
			c := onHost(t)(code, nil)
			out := make([]float32, n+8)
			for i := range out {
				out[i] = 7
			}
			c.Call(&Args{Out: &out[0], Q32: &mix[0], AScale: &scale[0], Q2: &base[0],
				Scr: (*byte)(unsafe.Pointer(&kons[0]))})
			want := hcMixRef(mix, scale, base, eps, iters, false)
			for i, w := range want {
				if d := math.Abs(float64(out[i]) - w); d > 2e-6*math.Max(1, math.Abs(w)) {
					t.Errorf("%v rounds %d: out[%d] = %g, want %g", em.Tier, iters, i, out[i], w)
				}
			}
			for i := n; i < n+8; i++ {
				if out[i] != 7 {
					t.Errorf("%v rounds %d: wrote out[%d] past the mixer", em.Tier, iters, i)
				}
			}
			if prev != nil && out[2*HCStreams] == prev[2*HCStreams] {
				t.Errorf("%v: rounds %d and the count before agree on the mixer", em.Tier, iters)
			}
			prev = out
		}
		// The head's weights alone, from padded eight-float buffers.
		code, err := em.HCMix(20, true)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, 2*HCStreams+4)
		for i := range out {
			out[i] = 7
		}
		onHost(t)(code, nil).Call(&Args{Out: &out[0], Q32: &mix[0], AScale: &scale[0], Q2: &base[0],
			Scr: (*byte)(unsafe.Pointer(&kons[0]))})
		for i, w := range hcMixRef(mix, scale, base, eps, 0, true) {
			if d := math.Abs(float64(out[i]) - w); d > 2e-6 {
				t.Errorf("%v head: out[%d] = %g, want %g", em.Tier, i, out[i], w)
			}
		}
		if out[2*HCStreams] != 7 {
			t.Errorf("%v head: wrote past its eight floats", em.Tier)
		}
	}
}

// TestColPoolMatchesReference holds the pool to float64 at every ragged width
// from one lane to a few vectors past, and slot counts from one to the CSA
// overlap's eight and past, with a guard after the output: a tail kernel that
// is right at whole vectors and wrong at seven is the shape this gate exists
// for.
func TestColPoolMatchesReference(t *testing.T) {
	kons := ActConsts()
	for _, em := range hcTiers() {
		code, err := em.ColPool()
		if err != nil {
			t.Fatal(err)
		}
		c := onHost(t)(code, nil)
		for w := 1; w <= 3*ElemLanes+3; w++ {
			for _, s := range []int{1, 2, 4, 8, 9} {
				kv, gate := make([]float32, s*w), make([]float32, s*w)
				for i := range kv {
					kv[i] = float32(math.Sin(float64(i)*0.61) * 3)
					gate[i] = float32(math.Cos(float64(i)*0.29) * 5)
				}
				const guard = 8
				out := make([]float32, w+guard)
				for i := range out {
					out[i] = 7
				}
				c.Call(&Args{Out: &out[0], Q32: &kv[0], Q2: &gate[0], Scr: (*byte)(unsafe.Pointer(&kons[0])),
					K: int64(w / ElemLanes), Rows: int64(w % ElemLanes), Cols: int64(s), RowStr: int64(4 * w)})
				for ch := 0; ch < w; ch++ {
					mx := math.Inf(-1)
					for j := 0; j < s; j++ {
						mx = math.Max(mx, float64(gate[j*w+ch]))
					}
					// The bound is relative to the terms' magnitude, |kv| weighted:
					// exp's polynomial carries a few ulps per term, and the
					// terms' signs may cancel down to a far smaller result.
					sum, acc, mag := 0.0, 0.0, 0.0
					for j := 0; j < s; j++ {
						e := math.Exp(float64(gate[j*w+ch]) - mx)
						sum += e
						acc += e * float64(kv[j*w+ch])
						mag += e * math.Abs(float64(kv[j*w+ch]))
					}
					if want := acc / sum; math.Abs(float64(out[ch])-want) > 2e-6*math.Max(1, mag/sum) {
						t.Errorf("%v width %d slots %d: out[%d] = %g, want %g", em.Tier, w, s, ch, out[ch], want)
					}
				}
				for i := w; i < w+guard; i++ {
					if out[i] != 7 {
						t.Errorf("%v width %d slots %d: wrote out[%d] past the width", em.Tier, w, s, i)
					}
				}
			}
		}
	}
}
