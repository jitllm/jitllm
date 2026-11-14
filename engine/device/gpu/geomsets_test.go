//go:build linux

package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/nn"
)

// TestNonCausalBlocksOfTwoGeometriesRunOnOneCard places a tower whose blocks
// attend at two geometries -- 4 heads of 64 and 2 heads of 128 over one
// residual -- on one card. Each block runs in its geometry's set, a picture's
// rows are reserved in both, a call over every block is cut where the set
// changes and runs, and the card holds nothing once it closes. Declined by
// name, as the second geometry's blocks were, they would run on the CPU.
func TestNonCausalBlocksOfTwoGeometriesRunOnOneCard(t *testing.T) {
	a := visionPlan()
	a.MaxSeq = 4096
	b := visionPlan()
	b.MaxSeq = 4096
	b.NHead, b.NKVHead, b.HeadDim = 2, 2, 128
	b.Graph = visionBlockGraph(b)
	plans := []*nn.LayerPlan{a, a, b, b}
	ws := append(visionBlocks(a, 2), visionBlocks(b, 2)...)
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	dt := g.devs[0]
	for li, p := range plans {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("tower block %d (%d heads of %d) declined: %s", li, p.NHead, p.HeadDim, g.Err())
		}
	}
	ka, kb := dt.layers[0].geo, dt.layers[2].geo
	if ka == kb || dt.layers[1].geo != ka || dt.layers[3].geo != kb {
		t.Fatalf("blocks of two geometries run in sets %v %v %v %v", ka, dt.layers[1].geo, kb, dt.layers[3].geo)
	}
	g.prewarmPromptWidths()
	const n = 3000
	if !g.ReserveRows(0, n) {
		t.Fatalf("a picture of %d rows refused: %s", n, g.Err())
	}
	for _, k := range []geoKey{ka, kb} {
		if c := dt.visOf(k).cap; c < n {
			t.Fatalf("set %v holds %d rows for a picture of %d", k, c, n)
		}
	}
	g.mu.Lock()
	rs, ok := g.runs(0, len(plans))
	g.mu.Unlock()
	if !ok || len(rs) != 2 {
		t.Fatalf("a call over both geometries is %d runs (%v), want 2", len(rs), ok)
	}
	cs := make([]float32, a.NEmbd)
	if !g.Layers(0, len(plans), 0, n, make([]float32, n*a.NEmbd), cs, nil, nil) {
		t.Fatalf("a picture of %d rows did not run: %s", n, g.Err())
	}
	g.Close()
	if live := d.allocBytes - d.freedBytes; live != 0 {
		t.Fatalf("%d bytes left on the card after Close", live)
	}
}
