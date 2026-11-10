package convert

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestMixedExpertTypesStackAsF32 stacks a block whose gate experts are one
// Q8_0 and one F16 sheet: the gate bank is F32, each expert's values in id
// order exactly as its own type decodes them, and the up bank, uniform,
// keeps its Q8_0 bytes. Per-expert tensors left in the container -- what a
// mixed block became before -- run on no device.
func TestMixedExpertTypesStackAsF32(t *testing.T) {
	c := &jlm.Config{NEmbd: 32, NFFNExp: 2, NExpert: 2}
	q8 := func(seed byte) *jlm.Tensor {
		b := make([]byte, 2*34)
		for r := range 2 {
			binary.LittleEndian.PutUint16(b[34*r:], 0x3c00+uint16(seed)) // a scale near 1
			for i := range 32 {
				b[34*r+2+i] = byte(int8(int(seed) + 3*i - 40 + r))
			}
		}
		return &jlm.Tensor{Type: jlm.TypeQ8, NDim: 2, Dims: [4]uint64{32, 2}, Data: b, Index: -1,
			Name: "blk.0.ffn_x.weight"}
	}
	f16 := &jlm.Tensor{Type: jlm.TypeF16, NDim: 2, Dims: [4]uint64{32, 2}, Data: make([]byte, 2*64), Index: -1,
		Name: "blk.0.ffn_gate.1.weight"}
	for i := range 64 {
		binary.LittleEndian.PutUint16(f16.Data[2*i:], 0x3800+uint16(i)) // 0.5 and up
	}
	gate0, up0, up1 := q8(1), q8(5), q8(9)
	experts := map[bankKey][]*jlm.Tensor{
		{jlm.RoleExpGateBank, 0}: {gate0, f16},
		{jlm.RoleExpUpBank, 0}:   {up0, up1},
	}
	out, err := stackGGUFExperts(experts, &Config{Config: c})
	if err != nil {
		t.Fatal(err)
	}
	var gate, up *jlm.Tensor
	for i := range out {
		switch out[i].Role {
		case jlm.RoleExpGateBank:
			gate = &out[i]
		case jlm.RoleExpUpBank:
			up = &out[i]
		default:
			t.Fatalf("%s reached the container as %v: a mixture's experts are banks", out[i].Name, out[i].Role)
		}
	}
	if gate == nil || up == nil {
		t.Fatalf("the block's banks: gate %v, up %v -- a mixed bank was left per-expert", gate != nil, up != nil)
	}
	if gate.Type != jlm.TypeF32 || gate.NDim != 3 || gate.Dims[2] != 2 {
		t.Fatalf("gate bank: %v %d-D %v, want F32 3-D over 2 experts", gate.Type, gate.NDim, gate.Dims[:gate.NDim])
	}
	got, err := gate.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	var want []float32
	for _, x := range []*jlm.Tensor{gate0, f16} {
		src, _ := jlm.SourceType(x.Type)
		f := make([]float32, 64)
		if err := quant.Dequant32(src, x.Data, f); err != nil {
			t.Fatal(err)
		}
		want = append(want, f...)
	}
	if len(got) != 4*len(want) {
		t.Fatalf("gate bank holds %d bytes, want %d", len(got), 4*len(want))
	}
	for i, w := range want {
		if g := math.Float32frombits(binary.LittleEndian.Uint32(got[4*i:])); g != w {
			t.Fatalf("gate bank value %d (expert %d) = %v, want %v as its own type decodes it", i, i/64, g, w)
		}
	}
	if up.Type != jlm.TypeQ8 {
		t.Fatalf("the uniform up bank was widened to %v", up.Type)
	}
	ub, err := up.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ub, append(append([]byte{}, up0.Data...), up1.Data...)) {
		t.Fatal("the uniform up bank is not its experts' bytes in id order")
	}
}
