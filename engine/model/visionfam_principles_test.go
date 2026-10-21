package model

import (
	"fmt"
	"image"
	"image/color"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// Pixtral's and Llama 4's vision under the five principles: the tower pages
// under a budget of one vision page with the same bits, a decode after a
// picture allocates nothing on the host or a device, and a second turn about
// the same picture restores past it from the prefix cache without running a
// tower block or asking the image cache. Relocation is each family's
// OnDeviceMatchesHost; JIT and no Go compute are the package-wide gates; the
// warm encode is TestWarmImageEncodeDoesNotAllocate (towerFamilies).

// visionFam is one of these families' fixtures: its model, its picture, and
// how a chat lays a picture out.
type visionFam struct {
	name string
	open func(t testing.TB, opts ...Option) (*Model, image.Image)
	// spans is msgs with img laid out in the family's own markers.
	spans func(t *testing.T, m *Model, msgs []ChatMessage, img image.Image) []Span
	// encode is the tower's rows for img on a fresh vision State.
	encode func(t *testing.T, m *Model, img image.Image) []float32
}

// pictureSpans is msgs with img laid out by ChatSpansImages: a family whose
// picture is one span between its markers.
func pictureSpans(t *testing.T, m *Model, msgs []ChatMessage, img image.Image) []Span {
	st := m.NewState(8)
	p, err := st.Picture(img)
	st.Close()
	if err != nil {
		t.Fatal(err)
	}
	spans, err := m.ChatSpansImages(msgs, []Image{{Picture: p}}, true)
	if err != nil {
		t.Fatal(err)
	}
	return spans
}

// pictureEncode is the tower's rows for img, preprocessed and encoded on a
// fresh vision State.
func pictureEncode(t *testing.T, m *Model, img image.Image) []float32 {
	s := m.Tower().testState()
	defer s.Close()
	px, err := s.PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...)
}

var visionFams = []visionFam{
	{
		name: "gemma3n",
		open: func(t testing.TB, opts ...Option) (*Model, image.Image) {
			m, g := openGemma3nV(t, opts...)
			return m, goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
		},
		spans:  pictureSpans,
		encode: pictureEncode,
	},
	{
		name: "phi4",
		open: func(t testing.TB, opts ...Option) (*Model, image.Image) {
			m, g := openPhi4VFixture(t, opts...)
			return m, goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
		},
		spans:  pictureSpans,
		encode: pictureEncode,
	},
	{
		name: "gemma4",
		open: func(t testing.TB, opts ...Option) (*Model, image.Image) {
			m, g := openGemma4V(t, opts...)
			return m, goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
		},
		spans:  pictureSpans,
		encode: pictureEncode,
	},
	{
		name: "pixtral",
		open: func(t testing.TB, opts ...Option) (*Model, image.Image) {
			m, g := openPixtral(t, opts...)
			return m, goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
		},
		spans: func(t *testing.T, m *Model, msgs []ChatMessage, img image.Image) []Span {
			st := m.NewState(8)
			p, err := st.Picture(img)
			st.Close()
			if err != nil {
				t.Fatal(err)
			}
			spans, err := m.ChatSpansImages(msgs, []Image{{Picture: p}}, true)
			if err != nil {
				t.Fatal(err)
			}
			return spans
		},
		encode: func(t *testing.T, m *Model, img image.Image) []float32 {
			s := m.Tower().testState()
			defer s.Close()
			px, err := s.PreprocessImage(img)
			if err != nil {
				t.Fatal(err)
			}
			out, err := s.Encode(px)
			if err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out...)
		},
	},
	{
		name: "llama4",
		open: func(t testing.TB, opts ...Option) (*Model, image.Image) {
			m, g := openLlama4V(t, opts...)
			return m, goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
		},
		spans: func(t *testing.T, m *Model, msgs []ChatMessage, img image.Image) []Span {
			rows, cols, ps, err := m.Tower().Llama4Pieces(img)
			if err != nil {
				t.Fatal(err)
			}
			pic, err := m.Llama4Spans(rows, cols, ps)
			if err != nil {
				t.Fatal(err)
			}
			spans, err := m.ChatSpansParts(msgs, [][]Span{pic}, true)
			if err != nil {
				t.Fatal(err)
			}
			return spans
		},
		encode: func(t *testing.T, m *Model, img image.Image) []float32 {
			_, _, ps, err := m.Tower().Llama4Pieces(img)
			if err != nil {
				t.Fatal(err)
			}
			return l4vEncode(t, m.Tower(), ps)
		},
	},
}

// otherPicture is img with one pixel changed: another picture.
func otherPicture(img image.Image) image.Image {
	b := img.Bounds()
	o := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			o.Set(x, y, img.At(x, y))
		}
	}
	o.Set(b.Min.X+3, b.Min.Y+5, color.RGBA{255, 0, 255, 255})
	return o
}

func TestVisionFamTowerPagesUnderABudget(t *testing.T) {
	for _, f := range visionFams {
		t.Run(f.name, func(t *testing.T) {
			m, img := f.open(t)
			full := f.encode(t, m, img)
			vpage := m.container.PageBytes(int(m.container.H.NBlocks))
			n := m.Tower().Cfg.NLayer
			m.Close()
			mb, _ := f.open(t, WithPageBudget(vpage))
			defer mb.Close()
			got := f.encode(t, mb, img)
			_, ev := mb.container.Faults()
			if ev == 0 {
				t.Fatalf("a budget of one vision page against a %d-page tower evicted nothing", n)
			}
			if !slices.Equal(got, full) {
				t.Fatalf("the tower's rows under a one-page budget differ from the unbudgeted ones")
			}
			t.Logf("one page of %d KiB against a %d-page tower: %d eviction(s), bit-identical", vpage>>10, n, ev)
		})
	}
}

func TestVisionFamDecodeAfterAPictureDoesNotAllocate(t *testing.T) {
	for _, f := range visionFams {
		t.Run(f.name, func(t *testing.T) {
			m, img := f.open(t)
			defer m.Close()
			msgs := []ChatMessage{{Role: "user", Content: "What is this?", Images: 1}}
			spans := f.spans(t, m, msgs, img)
			run := func(t *testing.T, gpu *tier.GPU) {
				// Past the device's launch-sequence capture, so the window
				// replays it, and inside one 128-position score grain and one
				// 256-position KV page, so it crosses neither (crossing a grain
				// re-captures the launch graph, and the host commits a page at
				// every 256, both once per span rather than per token):
				// decodeAllocs' rule, with the prompt's length in front of it.
				// The window starts 136 into a page, the second grain's
				// eighth position.
				const n = 48
				pos := SpanPositions(spans, m.Cfg.NEmbd)
				warm := 96 + (256+136-(pos+96)%256)%256
				s := m.NewState(pos + warm + n + 8)
				defer s.Close()
				if gpu != nil {
					if err := s.SetDevice(gpu); err != nil {
						t.Fatal(err)
					}
					if s.GPULayers() == 0 {
						t.Skip("no text block placed on the device -- this arm proved nothing")
					}
				}
				if _, err := s.PrefillMixed(spans...); err != nil {
					t.Fatal(err)
				}
				for i := range warm {
					if _, err := s.Forward(int32(5 + i)); err != nil {
						t.Fatal(err)
					}
				}
				var c0 tier.Stats
				if gpu != nil {
					c0 = gpu.Stats()
				}
				r0 := m.container.Reads()
				w := countAllocs(func() {
					for i := range n {
						if _, err := s.Forward(int32(3 + i%64)); err != nil {
							t.Fatal(err)
						}
					}
				})
				captures := 0
				if gpu != nil {
					captures = gpu.Stats().Captures - c0.Captures
				}
				allocVerdict(t, fmt.Sprintf("after a picture, %d of %d blocks placed", s.GPULayers(), m.Cfg.NLayer),
					w, n, m.container.Reads()-r0, captures)
			}
			t.Run("host", func(t *testing.T) { run(t, nil) })
			for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
				t.Run(spec, func(t *testing.T) {
					gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
					if err != nil || gpu == nil {
						noDevice(t, spec, err)
					}
					defer gpu.Close()
					run(t, gpu)
				})
			}
		})
	}
}

// TestVisionFamTwoTurnChatRestoresPastThePicture: a second turn about the
// same picture restores the first turn's prompt from the prefix cache -- the
// picture's rows, any rows the family writes between them, and the text after
// -- runs no tower block and asks the image cache nothing, and decodes what a
// cold run does; another picture restores nothing past its start.
func TestVisionFamTwoTurnChatRestoresPastThePicture(t *testing.T) {
	for _, f := range visionFams {
		t.Run(f.name, func(t *testing.T) {
			m, img := f.open(t)
			defer m.Close()
			// The question runs past a whole chunk beyond the picture, so the
			// chunk turn 2 restores up to ends after it: Ministral's template
			// puts a system prompt ahead that would otherwise leave the
			// boundary inside the picture.
			turn1 := []ChatMessage{{Role: "user", Images: 1,
				Content: "What is in this picture? Name the colours, the shapes, and where each of them sits."}}
			turn2 := append(append([]ChatMessage(nil), turn1...),
				ChatMessage{Role: "assistant", Content: "A disc."},
				ChatMessage{Role: "user", Content: "What colour is it?"})
			type result struct {
				restored, imgAt, imgEnd, n int
				lookups, blocks            int64
				toks                       []int32
			}
			run := func(store KVStore, msgs []ChatMessage, pic image.Image) result {
				spans := f.spans(t, m, msgs, pic)
				st := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + 16)
				defer st.Close()
				st.SetKVStore(store)
				mustKey(t, st, "vision-families/two-turn")
				var r result
				r.imgAt = -1
				at := 0
				for _, sp := range spans {
					if sp.rows() {
						if r.imgAt < 0 {
							r.imgAt = at
						}
						r.imgEnd = at + sp.n(m.Cfg.NEmbd)
					}
					at += sp.n(m.Cfg.NEmbd)
				}
				h0, m0 := m.ImageCacheStats()
				lg, err := st.PrefillCachedMixed(spans...)
				if err != nil {
					t.Fatal(err)
				}
				h1, m1 := m.ImageCacheStats()
				r.lookups = h1 - h0 + m1 - m0
				if st.visState != nil {
					r.blocks = st.visState.vis.hostBlocks + st.visState.vis.devBlocks
				}
				r.restored, r.n = st.KVRestored(), at
				for range 8 {
					id := Greedy(lg)
					r.toks = append(r.toks, id)
					if lg, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				return r
			}
			cold := run(NewMemStore(), turn2, img)
			store := NewMemStore()
			t1 := run(store, turn1, img)
			t2 := run(store, turn2, img)
			if want := t1.n / kvChunk * kvChunk; t2.restored < want || t2.restored <= t1.imgEnd {
				t.Fatalf("turn 2 restored %d positions of turn 1's %d (want %d, past the picture at %d..%d)",
					t2.restored, t1.n, want, t1.imgAt, t1.imgEnd)
			}
			if !slices.Equal(t2.toks, cold.toks) {
				t.Fatalf("the restored turn decodes %v, a cold one %v", t2.toks, cold.toks)
			}
			if t2.lookups != 0 || t2.blocks != 0 {
				t.Fatalf("turn 2 restored the picture and still asked the image cache %d time(s) and ran %d "+
					"tower block(s)", t2.lookups, t2.blocks)
			}
			diff := run(store, turn2, otherPicture(img))
			if diff.restored > diff.imgAt {
				t.Fatalf("a different picture restored %d positions, past its start at %d", diff.restored, diff.imgAt)
			}
			t.Logf("turn 1 %d positions (picture %d..%d); turn 2 restored %d of %d with no tower block and no "+
				"image-cache lookup, tokens %v = cold; another picture restored %d",
				t1.n, t1.imgAt, t1.imgEnd, t2.restored, t2.n, t2.toks, diff.restored)
		})
	}
}
