package convert

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestK3JoinsAMixedQuantization joins a KDA block's q, k and v as a mixed
// quantization stores them (Kimi-K3-0.40B's Q4_K_M: attn_q at Q4_K, attn_k
// and attn_v at Q8_0; here q at F16): the join is Q8_0, the Q8_0 parts come
// back bit for bit and the other within Q8_0's rounding. Refusing the mix,
// as the join did, leaves every mixed quantization of Kimi-K3 unconvertible.
func TestK3JoinsAMixedQuantization(t *testing.T) {
	const k, rows = 64, 8
	r := rand.New(rand.NewSource(1))
	draw := func() []float32 {
		v := make([]float32, k*rows)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		return v
	}
	qv := draw()
	f16 := make([]byte, 2*len(qv))
	for i, x := range qv {
		binary.LittleEndian.PutUint16(f16[2*i:], f32ToF16(x))
	}
	kb, vb := quantizeQ8Rows(draw(), k, rows), quantizeQ8Rows(draw(), k, rows)
	part := func(ty jlm.Type, b []byte) *jlm.Tensor {
		e := &jlm.Tensor{Type: ty, NDim: 2, Data: b, Name: "part"}
		e.Dims[0], e.Dims[1] = k, rows
		return e
	}
	e, err := kdaJoinRows([]*jlm.Tensor{part(jlm.TypeF16, f16), part(jlm.TypeQ8, kb), part(jlm.TypeQ8, vb)},
		jlm.RoleAttnQKV, k, rows)
	if err != nil {
		t.Fatal(err)
	}
	if e.Type != jlm.TypeQ8 || e.Dims[0] != k || e.Dims[1] != 3*rows {
		t.Fatalf("joined as %v %v, want Q8_0 {%d, %d}", e.Type, e.Dims[:2], k, 3*rows)
	}
	n := len(kb)
	if !bytes.Equal(e.Data[n:2*n], kb) || !bytes.Equal(e.Data[2*n:], vb) {
		t.Fatal("a Q8_0 part did not come back bit for bit")
	}
	got := make([]float32, k*rows)
	if err := quant.Dequant32(quant.Q8_0, e.Data[:n], got); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, k*rows)
	if err := quant.Dequant32(quant.F16, f16, want); err != nil {
		t.Fatal(err)
	}
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	if nmse := num / den; !(nmse < 1e-4) || math.IsNaN(nmse) {
		t.Fatalf("the F16 part re-stored as Q8_0 reads NMSE %.3e", nmse)
	}

	// One type stays bytes back to back.
	e, err = kdaJoinRows([]*jlm.Tensor{part(jlm.TypeQ8, kb), part(jlm.TypeQ8, vb), part(jlm.TypeQ8, kb)},
		jlm.RoleAttnQKV, k, rows)
	if err != nil {
		t.Fatal(err)
	}
	if e.Type != jlm.TypeQ8 || !bytes.Equal(e.Data, append(append(append([]byte(nil), kb...), vb...), kb...)) {
		t.Fatal("one type was not joined byte for byte")
	}
}
