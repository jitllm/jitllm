package jit_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// TestComposedGPUSampleDraw holds graph.SampleDraw on every GPU device to
// its definition (jit.SampleDrawRef) at the full candidate count a head
// samples from: the token of every draw whose best score clears the next by
// a margin float32 cannot cross, over random logits, k, temperatures and
// cuts.
func TestComposedGPUSampleDraw(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
	}
	const n, K, H = 1000, 256, 16
	g, in := jit.SampleDrawGraph(n, K, H)
	if err := graph.Verify(g, graph.Facts{}); err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		t.Run(d.API(), func(t *testing.T) {
			_, codes, err := jit.Default().Form(jit.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{},
				Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
			if err != nil {
				t.Fatal(err)
			}
			args := make([]uint32, 2+H)
			args[1] = math.Float32bits(1)
			for i := 2; i < len(args); i++ {
				args[i] = n
			}
			rng := rand.New(rand.NewSource(29))
			decisive, cases := 0, 0
			for c := range 120 {
				l := make([]float32, n)
				lw := make([]uint32, n)
				for i := range l {
					l[i] = float32(rng.NormFloat64() * 3)
					lw[i] = math.Float32bits(l[i])
				}
				k := 1 + rng.Intn(K)
				invT := 1 / (0.3 + 1.5*rng.Float64())
				topp, minp := math.Inf(1), 0.0
				if c%3 != 0 {
					topp = 0.3 + 0.7*rng.Float64()
				}
				if c%2 == 0 {
					minp = 0.1 * rng.Float64()
				}
				seed, pos := rng.Uint32(), rng.Uint32()
				want, _, margin := jit.SampleDrawRef(l, K, k, invT, topp, minp, seed, pos)
				vals := map[graph.ValueID][]uint32{
					in["logits"]: lw, in["sample.args"]: append([]uint32(nil), args...), in["sample.penalty"]: {math.Float32bits(1)},
					in["sample.draw"]: {math.Float32bits(float32(k)), math.Float32bits(float32(invT)),
						math.Float32bits(float32(topp)), math.Float32bits(float32(minp))},
					in["sample.counter"]: {seed, pos}, in["token"]: poisoned(1),
				}
				run := newPassRunner(d)
				words, err := run.run(codes, 0, len(codes), vals, map[string][]uint32{"rows.n": {1}}, nil, nil)
				run.close()
				if err != nil {
					t.Fatal(err)
				}
				cases++
				if margin < 1e-3 {
					continue
				}
				decisive++
				if len(words) != 1 || words[0] != want {
					t.Fatalf("case %d (k %d, 1/T %.3f, top-p %.3f, min-p %.3f): drew %v, want %d (margin %.3g)", c, k, invT, topp, minp, words, want, margin)
				}
			}
			if decisive < cases*9/10 {
				t.Fatalf("%d of %d draws decisive: the comparison sees too little", decisive, cases)
			}
			t.Logf("%d of %d draws decisive and right at %d candidates", decisive, cases, K)
		})
	}
}
