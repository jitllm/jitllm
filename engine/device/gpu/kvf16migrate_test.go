package gpu

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/nn"
)

// TestF16HistoryTravelsRounded moves a block's history onto a device whose V
// pages are packed binary16 (Config.KVF16) and home again: K comes back
// exactly, and V as quant.EncodeHalf then DecodeHalf round it, bit for bit --
// the narrowing and widening on the way are the card's own, composed
// (graph.OpHalves, graph.OpWidenHalves, kvhalves.go), and must have run.
// The positions are not a whole number of pages, so a page runs part full.
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
	if n, _ := g.Stats().ServedPart(nn.ConvertedHalves); n == 0 {
		t.Fatal("the history travelled without the card's own binary16 conversions")
	}
}

// TestTypedHistoryTravelsRounded moves a block's binary16 history onto the
// card, whose pages hold it as the CPU's do (a typed pool: keys and values
// binary16 rows by position), and home again, at rows of 32 floats, of 64
// and of 256, one position and several pages' worth: K and V each come back
// as quant.EncodeHalf then DecodeHalf round it, bit for bit, through the
// card's own conversions.
//
// The tier closes the device it is given, so each case opens its own.
func TestTypedHistoryTravelsRounded(t *testing.T) {
	ran := 0
	for _, sh := range [][2]int{{1, 32}, {2, 32}, {2, 128}} {
		for _, pos := range []int{1, 37, 130} {
			for _, dv := range realDevices(t) {
				ran++
				t.Run(fmt.Sprintf("%s/%dkv%dd/%dpos", dv.API(), sh[0], sh[1], pos), func(t *testing.T) {
					typedHistoryTravelsRounded(t, dv, sh[0], sh[1], pos)
				})
			}
		}
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host -- this gate proved nothing")
	}
}

func typedHistoryTravelsRounded(t *testing.T, dv backend.Device, kvh, dim, pos int) {
	p := &nn.LayerPlan{NEmbd: 128, NHead: 2 * kvh, NKVHead: kvh, HeadDim: dim, NRot: dim, NFFN: 256,
		MaxSeq: 512, RMSEps: 1e-5, RopeBase: 10000, Spec: &nn.BlockSpec{}}
	textPlan(p)
	if pool := kvPoolOf(p); !pool.Half || pool.Row != 0 {
		t.Fatalf("the plan's history is %+v, not binary16 rows: the gate would move floats", pool)
	}
	g, err := New([]Slot{{Dev: dv, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	kvDim := kvh * dim
	k, v := make([]float32, pos*kvDim), make([]float32, pos*kvDim)
	for i := range k {
		k[i] = float32(i%13)*0.3 - 1.7 + float32(i)*1e-6
		v[i] = float32(i%29)*0.01171875*float32(1+i%3) - 0.13 + float32(i)*1e-7
	}
	round := func(x []float32) []float32 {
		r := make([]float32, len(x))
		for i, f := range x {
			r[i] = float32(quant.DecodeHalf(quant.EncodeHalf(f)))
		}
		return r
	}
	wantK, wantV := round(k), round(v)
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
	for i := range k {
		if math.Float32bits(k[i]) != math.Float32bits(wantK[i]) || math.Float32bits(v[i]) != math.Float32bits(wantV[i]) {
			t.Fatalf("position %d element %d came home k %v v %v, want %v and %v", i/kvDim, i%kvDim, k[i], v[i], wantK[i], wantV[i])
		}
	}
	if n, _ := g.Stats().ServedPart(nn.ConvertedHalves); n == 0 {
		t.Fatal("the history travelled without the card's own binary16 conversions")
	}
}
