package tier

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestAContainerWeightIsNotRepackedOnAUnifiedDevice offers a block whose
// weights come from a container -- already in the device layout, Packed set
// and Data its payload span, as model.bindOne binds them -- to a unified
// device with an arena budget, which is what the command line gives a
// streamed placement. The arena packs GGUF bytes: handed a container's
// payload, it reads it as GGUF blocks, so it must not be asked at all, and the
// bytes the device holds must be the container's own.
func TestAContainerWeightIsNotRepackedOnAUnifiedDevice(t *testing.T) {
	d := &impDev{unified: true, align: int(kernels.HostAlign())}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	g.SetArena(1 << 30)
	dt := g.devs[0]
	if !dt.importing() {
		t.Fatal("the device is not importing, so the arena branch is never reached: this gate measures nothing")
	}

	p := arenaPlan()
	w, _ := containerBlock(t, p)
	if !dt.PrepLayer(0, p, w) {
		t.Fatalf("a container block was declined on a unified device: %s", dt.Err())
	}
	if st := g.ArenaStats(); st.Packs+st.Hits+st.Declines != 0 {
		t.Fatalf("the arena was asked to pack a container weight: %d packs, %d hits, %d declines",
			st.Packs, st.Hits, st.Declines)
	}

	dt.mu.Lock()
	defer dt.mu.Unlock()
	l := dt.layers[0]
	for i, pair := range []struct {
		r *resident
		x nn.Weight
	}{{l.wq, w.Wq}, {l.wk, w.Wk}, {l.wv, w.Wv}, {l.wo, w.Wo}, {l.gate, w.Gate}, {l.up, w.Up}, {l.down, w.Down}} {
		if pair.r == nil || !pair.r.ok {
			t.Fatalf("tensor %d is not resident", i)
		}
		for j, want := range [][]byte{pair.x.Packed.QS, pair.x.Packed.D, pair.x.Packed.SC} {
			if len(want) == 0 {
				continue
			}
			buf := []backend.Buf{pair.r.qs, pair.r.d, pair.r.sc}[j]
			got := make([]byte, len(want))
			if err := buf.Read(got); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("tensor %d plane %d: the device holds bytes that are not the container's", i, j)
			}
		}
	}
}

// containerBlock is a block in the form a .jlm hands the tier: the packer's
// output is the layout a container stores, Packed carries its spans and Data
// is the payload span, as model.bindOne binds them. It returns the device
// bytes the block's weights take.
func containerBlock(t *testing.T, p *nn.LayerPlan) (*nn.LayerWeights, uint64) {
	t.Helper()
	w := blockWeights(p)
	fill(w, 20261002)
	var n uint64
	for _, x := range []*nn.Weight{&w.Wq, &w.Wk, &w.Wv, &w.Wo, &w.Gate, &w.Up, &w.Down} {
		q, ok := quantOf(x.T)
		if !ok {
			t.Fatalf("%v has no device format", x.T)
		}
		qs, dw, sc, err := kernels.PackWeights(q, x.Data, x.Rows, x.K)
		if err != nil {
			t.Fatal(err)
		}
		x.Packed = &nn.Packed{QS: u32b(qs), D: u32b(dw), SC: u32b(sc)}
		x.Data = x.Packed.QS
		n += uint64(len(qs)+len(dw)+len(sc)) * 4
	}
	return w, n
}

// TestAContainerBlockIsPricedInFullOnAUnifiedDevice: the arena's headroom
// pays only for what the arena would hold, and it holds no container weight,
// so a container block is priced at its whole upload. Priced at the arena's
// headroom it was admitted with room for none of it, and its tensors then
// declined one by one -- the half-placed block the whole-block price exists
// to prevent.
func TestAContainerBlockIsPricedInFullOnAUnifiedDevice(t *testing.T) {
	p := arenaPlan()
	w, weights := containerBlock(t, p)
	d := &impDev{unified: true, align: int(kernels.HostAlign())}
	g, err := New([]Slot{{Dev: d, Bytes: weights / 2}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	g.SetArena(1 << 30)
	dt := g.devs[0]
	dt.mu.Lock()
	need, _ := dt.blockBytes([]nn.Weight{w.Wq, w.Wk, w.Wv, w.Wo, w.Gate, w.Up, w.Down})
	dt.mu.Unlock()
	if need != weights {
		t.Fatalf("a container block is priced at %d bytes on a unified device with an arena; "+
			"its upload is %d", need, weights)
	}
	used := g.Bytes()
	if dt.PrepLayer(0, p, w) {
		t.Fatal("a block twice the budget was admitted")
	}
	dt.mu.Lock()
	defer dt.mu.Unlock()
	for k, r := range dt.res {
		if r != nil && r.ok {
			t.Fatalf("a declined block left tensor %+v resident", k)
		}
	}
	if dt.used != used {
		t.Fatalf("a declined block left %d bytes charged", dt.used-used)
	}
}
