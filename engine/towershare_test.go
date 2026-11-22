package engine

import (
	"fmt"
	"image"
	"image/draw"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
)

// TestASharedPictureRestoresPastIt: a picture prompt a sharing session ran
// and decoded past (ShareKV, a server's generate) restores whole for the next
// session that asks it -- every position but the last, past the picture --
// encodes no picture and decodes the tokens a cold run does, on the CPU and
// with every block on each listed card, the first session let go by Close or
// by Reset (a server's session starting over). The violation: a picture a
// pixel apart restores nothing past its start.
func TestASharedPictureRestoresPastIt(t *testing.T) {
	m, img := openSmolVLM(t)
	defer m.Close()
	other := image.NewNRGBA(img.Bounds())
	draw.Draw(other, other.Bounds(), img, img.Bounds().Min, draw.Src)
	other.Pix[0] ^= 0x40
	msgs := []model.ChatMessage{{Role: "user", Content: "What is in this picture?", Images: 1}}
	type result struct {
		restored, imgAt, n, encodes int
		toks                        []int32
	}
	run := func(t *testing.T, g *gpu.GPU, ns string, shared *SharedStore, pic image.Image, reset bool) result {
		t.Helper()
		st := testInstance(t, m, 2048, sharedCPU(t))
		defer st.Close()
		if g != nil {
			if err := st.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
		}
		mustKey(t, st, ns)
		if err := st.ShareKV(shared); err != nil {
			t.Fatal(err)
		}
		sp, err := m.PictureSpan(pic)
		if err != nil {
			t.Fatal(err)
		}
		spans, err := m.ChatSpans(msgs, []model.Span{sp}, true)
		if err != nil {
			t.Fatal(err)
		}
		var r result
		for _, s := range spans {
			if s.Rows() {
				break
			}
			r.imgAt += s.N(m.Cfg.NEmbd)
		}
		before := st.tower.encodes
		lg, err := st.PrefillCachedMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		r.encodes = st.tower.encodes - before
		r.restored, r.n = st.KVRestored(), model.SpanPositions(spans, m.Cfg.NEmbd)
		for range 6 {
			id := Greedy(lg)
			r.toks = append(r.toks, id)
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		if reset {
			st.Reset()
		}
		return r
	}
	arms := func(t *testing.T, g *gpu.GPU) {
		// Each run's prompt is its namespace's: a card names its pages for the
		// tier, past any one store, and a picture run in one variant would
		// otherwise be a hit in the next.
		cold := run(t, g, "tower/cold", NewSharedStore(), img, false)
		for _, reset := range []bool{false, true} {
			ns, shared := fmt.Sprintf("tower/shared/reset=%v", reset), NewSharedStore()
			first := run(t, g, ns, shared, img, reset)
			again := run(t, g, ns, shared, img, false)
			t.Logf("reset=%v: %d positions (picture from %d); the second run restored %d and encoded %d, tokens %v (cold %v)",
				reset, first.n, first.imgAt, again.restored, again.encodes, again.toks, cold.toks)
			if again.restored < first.n-1 {
				t.Errorf("reset=%v: the second run restored %d of the prompt's %d positions, want every one but the last",
					reset, again.restored, first.n)
			}
			if again.encodes != 0 {
				t.Errorf("reset=%v: the second run encoded %d picture(s), want none: its rows were restored", reset, again.encodes)
			}
			if !slices.Equal(again.toks, cold.toks) {
				t.Errorf("reset=%v: the restored run decodes %v, a cold one %v", reset, again.toks, cold.toks)
			}
			diff := run(t, g, ns, shared, other, false)
			if diff.restored > diff.imgAt {
				t.Errorf("VIOLATION: reset=%v: a picture a pixel apart restored %d positions, past its start at %d",
					reset, diff.restored, diff.imgAt)
			}
		}
	}
	t.Run("host", func(t *testing.T) { arms(t, nil) })
	for _, spec := range stepDevices() {
		g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec),
			gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			arms(t, g)
		})
	}
}
