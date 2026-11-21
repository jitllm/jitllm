package gpu

import (
	"fmt"
	"strings"
	"testing"
)

// TestADeclineReportsItsOwnReason runs two sessions on one device: the
// first's offer is declined, then the second's is, for another reason. Each
// session's Err is the reason its own call was declined for, the first's
// unmoved by the second's refusal after it: a decline read through the
// device's last error would report the second session's reason as the
// first's.
func TestADeclineReportsItsOwnReason(t *testing.T) {
	p := fakePlan()
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	ws := blocks(p, 2)
	g.HoldModel(p.Model)
	if !g.PrepLayer(0, p, ws[0]) {
		t.Fatalf("block 0 declined: %s", g.Err())
	}
	a, b := g.Attach().(*gpuSession), g.Attach().(*gpuSession)
	pa, pb := *p, *p
	pa.Model, pb.Model = p.Model+1, p.Model+2
	if a.PrepLayer(1, &pa, ws[1]) {
		t.Fatalf("model %d's block placed on a tier holding model %d's", pa.Model, p.Model)
	}
	own := a.Err()
	if want := fmt.Sprintf("model %d's block 1", pa.Model); !strings.Contains(own, want) {
		t.Fatalf("the declined session reports %q, which does not name its offer (%s)", own, want)
	}
	if b.PrepLayer(1, &pb, ws[1]) {
		t.Fatalf("model %d's block placed on a tier holding model %d's", pb.Model, p.Model)
	}
	if got := g.Err(); got == own {
		t.Fatalf("the second session's refusal left the device's last error at the first's (%q): the gate tells nothing apart", got)
	}
	if got := a.Err(); got != own {
		t.Errorf("after another session's refusal the declined session reports %q, not its own %q", got, own)
	}
	if got, want := b.Err(), fmt.Sprintf("model %d's block 1", pb.Model); !strings.Contains(got, want) {
		t.Errorf("the second session reports %q, which does not name its offer (%s)", got, want)
	}
}
