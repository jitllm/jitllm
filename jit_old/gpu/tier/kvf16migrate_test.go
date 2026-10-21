package tier

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// TestF16HistoryTravelsRounded moves a block's history onto a device whose V
// pages are packed binary16 (Config.KVF16) and home again: K comes back
// exactly, and V as quant.EncodeHalf then DecodeHalf round it, bit for bit --
// the narrowing and widening on the way are the host's generated kernels
// (nn.NarrowF16, nn.WidenF16), held here to the Go reference they replaced.
// The positions are not a whole number of vectors, so both kernels run a tail.
func TestF16HistoryTravelsRounded(t *testing.T) {
	devs := realDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host -- this gate proved nothing")
	}
	for _, dv := range devs {
		t.Run(dv.API(), func(t *testing.T) { f16HistoryTravelsRounded(t, dv) })
	}
}

func f16HistoryTravelsRounded(t *testing.T, dv backend.Device) {
	p := fakePlan()
	g, err := New([]Slot{{Dev: dv, Bytes: 1 << 30}}, WithDeviceTune(TuneOff),
		WithConfig(func(c *Config) { c.KVF16 = true }))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	const pos = 37
	kvDim := p.NKVHead * p.HeadDim
	k, v := make([]float32, pos*kvDim), make([]float32, pos*kvDim)
	for i := range k {
		k[i] = float32(i%13)*0.3 - 1.7
		v[i] = float32(i%29)*0.01171875*float32(1+i%3) - 0.13 + float32(i)*1e-7
	}
	wantK := append([]float32(nil), k...)
	wantV := make([]float32, len(v))
	for i, x := range v {
		wantV[i] = float32(quant.DecodeHalf(quant.EncodeHalf(x)))
	}
	if !g.PrepLayer(0, p, blocks(p, 1)[0]) {
		t.Fatalf("block 0 declined: %s", g.Err())
	}
	if !g.ReserveKV(pos) || !g.MigrateKV(0, k, v, pos, true) {
		t.Fatalf("the history did not go up: %s", g.Err())
	}
	clear(k)
	clear(v)
	if !g.MigrateKV(0, k, v, pos, false) {
		t.Fatalf("the history did not come home: %s", g.Err())
	}
	rounded := 0
	for i := range k {
		if math.Float32bits(k[i]) != math.Float32bits(wantK[i]) {
			t.Fatalf("k[%d] came home %v, went up %v", i, k[i], wantK[i])
		}
		if math.Float32bits(v[i]) != math.Float32bits(wantV[i]) {
			t.Fatalf("v[%d] came home %v, want %v (rounded through binary16)", i, v[i], wantV[i])
		}
		if wantV[i] != float32(i%29)*0.01171875*float32(1+i%3)-0.13+float32(i)*1e-7 {
			rounded++
		}
	}
	if rounded == 0 {
		t.Fatalf("no value was rounded: the gate cannot tell a packed V from a float32 one")
	}
}
