package composer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUPickedHead forms the head projecting the rows ids pick
// (graph.Head.Pick), with the projection's bias and without it, and holds
// each picked logit to the oracle's whole head at its id; the picks repeat
// ids and reach the last row. The violation: the biased head's pick formed
// without its bias must part from the biased oracle.
func TestComposedCPUPickedHead(t *testing.T) { gateComposedCPUPickedHead(t, hostArm()) }
func gateComposedCPUPickedHead(t *testing.T, cpuOn cpuArm) {
	const P, bound = 24, 1e-9
	rng := rand.New(rand.NewSource(17))
	h := newHeadWeights(t, rng)
	x := make([]float32, headEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 3)
	}
	ids := make([]uint32, P)
	for i := range ids {
		ids[i] = uint32(rng.Intn(headVocab))
	}
	ids[0], ids[1], ids[2] = headVocab-1, 0, ids[3]
	pick := func(s graph.Head) ([]float32, error) {
		s.Pick = P
		g := s.Build()
		return formHeadOnCPUWith(t, cpuOn, g, 1, x, h, map[graph.ValueID][]uint32{g.In[1]: ids})
	}
	at := func(whole []float32) []float32 {
		v := make([]float32, P)
		for i, id := range ids {
			v[i] = whole[id]
		}
		return v
	}
	for _, bias := range []bool{false, true} {
		s := graph.Head{NEmbd: headEmbd, NVocab: headVocab, Eps: headEps, Output: jlm.RoleOutput, Bias: bias}
		got, err := pick(s)
		if err != nil {
			t.Fatalf("bias %v: %v", bias, err)
		}
		e := cpuNMSE(got, at(headWant(t, h, s, x, 1)))
		t.Logf("bias %v: NMSE %.3e against the oracle's whole head at %d picked ids", bias, e, P)
		if math.IsNaN(e) || !(e < bound) {
			t.Errorf("bias %v: NMSE %.3e against the oracle, bound %.0e", bias, e, bound)
		}
	}
	biased := graph.Head{NEmbd: headEmbd, NVocab: headVocab, Eps: headEps, Output: jlm.RoleOutput, Bias: true}
	unbiased := biased
	unbiased.Bias = false
	got, err := pick(unbiased)
	if err != nil {
		t.Fatal(err)
	}
	e := cpuNMSE(got, at(headWant(t, h, biased, x, 1)))
	t.Logf("violation, the bias dropped: NMSE %.3e from the biased oracle", e)
	if math.IsNaN(e) || !(e > 1e-4) {
		t.Errorf("the pick without its bias parts from the biased oracle by %.3e: the gate cannot see the bias", e)
	}
}
