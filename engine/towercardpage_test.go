package engine

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestATowerPagesOnTheCard is the paging principle for a tower on a card
// listed alone (WithDevices(card), the CPU not listed): its blocks placed
// streamed on the card (Place.Stream) page through the card's slots under a
// budget half the tower below what the resident card spends, and two
// encodes give the resident card's rows; the CPU's frames read no page and
// hold none. SmolVLM-256M's tower is real: a fixture's is a few hundred
// kilobytes, which the card's slack absorbs.
func TestATowerPagesOnTheCard(t *testing.T) {
	text, mm := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf"), testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	path := jlmOfPair(t, text, mm)
	img := image.NewRGBA(image.Rect(0, 0, 300, 200))
	for y := range 200 {
		for x := range 300 {
			img.Set(x, y, color.RGBA{uint8(x * 255 / 299), uint8(y * 255 / 199), uint8((x + y) % 256), 255})
		}
	}
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			type arm struct {
				rows, again []float32
				st          gpu.Stats
				tower       uint64
			}
			encode := func(budget uint64, stream bool) (arm, error) {
				ds := dev
				if budget > 0 {
					ds += "=" + strconv.FormatUint(budget, 10)
				}
				g, err := openGPU(t, gpu.WithDevices(ds), gpu.WithDeviceTune(gpu.TuneOff))
				if err != nil || g == nil {
					noDevice(t, ds, err)
				}
				defer g.Close()
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				opts := []Option{WithKVF16(false), WithDevices(dev)}
				if stream {
					pl := Placement{Blocks: map[int]Place{}}
					for li := m.TowerBase; li < m.TowerBase+len(m.Tower.Blocks); li++ {
						pl.Blocks[li] = Place{On: dev, Stream: true}
					}
					opts = append(opts, WithPlacement(pl))
				}
				s := testInstance(t, m, 64, opts...)
				defer s.Close()
				var a arm
				if err := s.SetDevice(g); err != nil {
					return a, err
				}
				for li := m.TowerBase; li < m.TowerBase+len(m.Tower.Blocks); li++ {
					a.tower += m.Container.PageBytes(li)
				}
				rows, err := s.EncodePicture(img)
				if err != nil {
					return a, fmt.Errorf("%w (%s)", err, g.Err())
				}
				a.rows = slices.Clone(rows)
				if rows, err = s.EncodePicture(img); err != nil {
					return a, err
				}
				a.again = slices.Clone(rows)
				if nb, on := len(m.Tower.Blocks), s.tower.s.GPULayers(); on != nb {
					t.Fatalf("budget %d: %d of the tower's %d blocks on %s, the CPU not listed (%s)", budget, on, nb, dev, g.Err())
				}
				if n, held := s.fr.ChunksIn(), s.fr.ResidentPages(); n != 0 || held != 0 {
					t.Fatalf("budget %d: the CPU is not listed, and its frames read %d chunk(s) and hold %d page(s)", budget, n, held)
				}
				a.st = g.Stats()
				return a, nil
			}
			want, err := encode(0, false)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(want.again, want.rows) {
				t.Fatal("the resident card's second encode differs from its first")
			}
			if want.st.PageOuts != 0 {
				t.Fatalf("the unbudgeted card paged %d blocks out", want.st.PageOuts)
			}
			// A card spends elastically -- its reserved widths and history shrink
			// before a block pages -- so the budget steps down by half the
			// tower until the streamed tower pages; an encode that fails
			// before it does is the violation of the principle.
			var got arm
			var budget uint64
			for k := uint64(1); ; k++ {
				if k*want.tower/2 >= want.st.BudgetUsed {
					t.Fatalf("no budget down to %d bytes paged the streamed tower", want.st.BudgetUsed-(k-1)*want.tower/2)
				}
				budget = want.st.BudgetUsed - k*want.tower/2
				if got, err = encode(budget, true); err != nil {
					t.Fatalf("streamed at %d bytes (%d resident, the tower %d), the picture failed before the tower paged: %v",
						budget, want.st.BudgetUsed, want.tower, err)
				}
				if got.st.PageOuts > 0 {
					break
				}
				t.Logf("streamed at %d bytes: %d slots, no page-out", budget, got.st.Slots)
			}
			e, _ := nmseOf(got.rows, want.rows)
			e2, _ := nmseOf(got.again, want.rows)
			if math.IsNaN(e) || math.IsNaN(e2) || e > 1e-6 || e2 > 1e-6 {
				t.Fatalf("paged on %s: NMSE %.3e and %.3e from the resident card's rows", dev, e, e2)
			}
			t.Logf("%s streamed at %d bytes (%d resident, the tower %d): %d slots, %d page-ins, %d page-outs, rows %.3e and %.3e from the resident card's (bit for bit: %v)",
				dev, budget, want.st.BudgetUsed, want.tower, got.st.Slots, got.st.PageIns, got.st.PageOuts, e, e2, slices.Equal(got.rows, want.rows))
		})
	}
}
