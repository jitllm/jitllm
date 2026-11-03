package widgets

import (
	"testing"

	"github.com/gogpu/gg"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/render"
	"github.com/gogpu/ui/widget"
)

// Every icon must paint. gg/svg parses a document it then draws nothing of --
// a stroke on the root <svg> is not inherited -- so parsing is not the test.
func TestEveryIconPaints(t *testing.T) {
	for name, ic := range map[string]Icon{
		"chat": IconChat(), "discover": IconDiscover(), "models": IconModels(), "convert": IconConvert(),
		"machine": IconMachine(), "sun": IconSun(), "moon": IconMoon(), "copy": IconCopy(), "retry": IconRetry(),
		"image": IconImage(), "file": IconFile(), "send": IconSend(), "stop": IconStop(), "download": IconDownload(),
		"think": IconThink(), "sliders": IconSliders(), "trash": IconTrash(), "plus": IconPlus(), "memory": IconMemory(), "gauge": IconGauge(), "gpu": IconGPU(), "settings": IconSettings(), "folder": IconFolder(),
	} {
		dc := gg.NewContext(24, 24)
		cv := render.NewCanvas(dc, 24, 24)
		cv.(widget.SVGRenderer).RenderSVG(ic, geometry.NewRect(0, 0, 24, 24), widget.Hex(0xFFFFFF))
		img, n := dc.Image(), 0
		for y := 0; y < 24; y++ {
			for x := 0; x < 24; x++ {
				if _, _, _, a := img.At(x, y).RGBA(); a > 0 {
					n++
				}
			}
		}
		dc.Close()
		if n < 20 {
			t.Errorf("%s painted %d pixels", name, n)
		}
	}
}
