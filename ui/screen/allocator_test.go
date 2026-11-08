package screen

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// The map must actually draw its lanes; the formatter gates pass whether or
// not anything renders them.
func TestTheMemoryMapDrawsALanePerTier(t *testing.T) {
	sh := app.NewShell(nil, nil)
	sh.Store.BlockMap.Set(splitBlocks(48, 4))
	sh.Store.Alloc.Set(app.Allocation{
		NBlocks: 48, CPUBlocks: 44, DeviceBlocks: 4,
		HostUsed: 6 << 30, HostBudget: 8 << 30, Dense: 182_100_000, PageBytes: 128 << 20,
		Devices: []app.DeviceUse{{
			Name: "0:NVIDIA GeForce RTX 3050 Ti [cuda]", Short: "cuda:0", Blocks: 4,
			Used: 1_610_612_736, Limit: 2_147_483_648, Pool: "sm_86", Weights: 512 << 20,
		}},
	})

	got := drawPanel(t, sh, primitives.VBox(memoryMap(sh)))
	for _, want := range []string{"file (.jlm)", "host memory", "6.00 GiB", "cuda:0", "RTX 3050 Ti", "4 block(s)", "1 seam"} {
		if !strings.Contains(got, want) {
			t.Errorf("the map never drew %q:\n%s", want, got)
		}
	}
}

// And an absent device must draw no lane.
func TestTheMemoryMapHasNoLaneForAnAbsentDevice(t *testing.T) {
	sh := app.NewShell(nil, nil)
	sh.Store.BlockMap.Set(splitBlocks(48, 0))
	sh.Store.Alloc.Set(app.Allocation{NBlocks: 48, HostUsed: 1 << 30})

	got := drawPanel(t, sh, primitives.VBox(memoryMap(sh)))
	if !strings.Contains(got, "host memory") {
		t.Errorf("the host lane is missing:\n%s", got)
	}
	if strings.Contains(got, "[cuda]") || strings.Contains(got, "[vulkan]") {
		t.Errorf("a device lane was drawn with no devices:\n%s", got)
	}
}

// With nothing loaded the map draws nothing; the line above it says so.
func TestTheMemoryMapIsEmptyWhenNothingIsLoaded(t *testing.T) {
	sh := app.NewShell(nil, nil)
	if got := drawPanel(t, sh, primitives.VBox(memoryMap(sh))); got != "" {
		t.Errorf("the map drew %q with nothing loaded", got)
	}
}

// splitBlocks is n resident blocks, the first dev of them on device 0.
func splitBlocks(n, dev int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(widgets.PlaceHost | widgets.PlaceResident)
		if i < dev {
			b[i] = byte(widgets.OnDevice(0))
		}
	}
	return b
}

func drawPanel(t *testing.T, sh *app.Shell, w widget.Widget) string {
	t.Helper()
	ctx := mockCtx()
	widget.MountTree(w, ctx)
	w.Layout(ctx, geometry.Constraints{MaxWidth: 900, MaxHeight: geometry.Infinity})
	c := &uitest.MockCanvas{}
	w.Draw(ctx, c)
	return drawnText(c)
}

// A lane's caption must be drawn beside its name, not on top of it: the canvas
// does not clip text. Only the geometry shows it.
func TestAMemoryLaneDoesNotOverprintItsName(t *testing.T) {
	sh := app.NewShell(nil, nil)
	sh.Store.BlockMap.Set(splitBlocks(16, 0))
	sh.Store.Alloc.Set(app.Allocation{NBlocks: 16, HostUsed: 2 << 30, HostBudget: 8 << 30})
	w := primitives.VBox(memoryMap(sh))
	ctx := mockCtx()
	widget.MountTree(w, ctx)
	w.Layout(ctx, geometry.Constraints{MaxWidth: 1000, MaxHeight: 800})
	c := &uitest.MockCanvas{}
	w.Draw(ctx, c)

	var name, caption geometry.Rect
	var sawName, sawCaption bool
	see := func(text string, at geometry.Rect) {
		switch {
		case text == "host memory":
			name, sawName = at, true
		case strings.Contains(text, "8.00 GiB budget"):
			caption, sawCaption = at, true
		}
	}
	for _, x := range c.Texts {
		see(x.Text, x.Bounds)
	}
	for _, x := range c.StyledTexts {
		see(x.Text, x.Bounds)
	}
	if !sawName || !sawCaption {
		t.Fatalf("setup: the host row was not drawn (name %v, caption %v)", sawName, sawCaption)
	}
	if name.Width() <= 0 || caption.Min.X < name.Max.X {
		t.Errorf("the caption %v overprints the name %v", caption, name)
	}
}
