package engine

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// towerSources are the audit models a picture tower converts with, the
// geometry sweep's: a GGUF beside its projector, at most 3 GiB.
func towerSources(t *testing.T) []auditModel {
	var out []auditModel
	for _, src := range auditModels(t) {
		if !src.hf && src.mmproj != "" && src.size <= 3<<30 {
			out = append(out, src)
		}
	}
	return out
}

// pairContainer is src's text model and projector in one container.
func pairContainer(t *testing.T, src auditModel) string {
	t.Helper()
	path, err := convertPair(src.path, src.mmproj)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	return path
}

// principlePicture is a picture of no tower's square: a gradient in each
// channel, wider than tall.
func principlePicture() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 300, 200))
	for y := range 200 {
		for x := range 300 {
			img.Set(x, y, color.RGBA{uint8(x * 255 / 299), uint8(y * 255 / 199), uint8((x + y) % 256), 255})
		}
	}
	return img
}

// towerFixtures are the stated towers the principle gates run: every
// fixture's (synth-*), one a projector family.
func towerFixtures(t *testing.T) []auditModel {
	var out []auditModel
	for _, src := range towerSources(t) {
		if strings.HasPrefix(filepath.Base(src.name), "synth-") {
			out = append(out, src)
		}
	}
	if len(out) == 0 {
		t.Fatal("no tower fixture is here (see docs/testing.md)")
	}
	return out
}

// sameWords fails t where got and want differ in a word.
func sameWords(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: value %d is %v, want %v", what, i, got[i], want[i])
		}
	}
}

// TestTowersPageWithTheirRows is the paging principle for every stated
// tower: a picture encoded twice under a page budget of the tower's largest
// page -- each tower block evicting the one before it, and read again by the
// second encode -- gives the rows the resident tower gives, bit for bit, and
// the second encode read the tower's pages again where the resident one read
// none. The violation: a resident tower block's page overwritten, the next
// encode's rows are other rows -- the rows are read from the pages the
// budgeted encodes read again.
func TestTowersPageWithTheirRows(t *testing.T) {
	img := principlePicture()
	ran := 0
	for _, src := range towerFixtures(t) {
		t.Run(src.name, func(t *testing.T) {
			path := pairContainer(t, src)
			// encode is two encodes of img on a fresh instance under budget
			// (0: the model resident) and the pages the second read; with
			// spoil, the first tower block's first matrix's page overwritten
			// between them.
			encode := func(budget uint64, spoil bool) (first, second []float32, read int64) {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				if m.Tower == nil {
					t.Fatal("the container states no picture tower")
				}
				var opts []Option
				if budget > 0 {
					opts = append(opts, WithPageBudget(budget))
				}
				s := testInstance(t, m, 64, opts...)
				defer s.Close()
				rows, err := s.EncodePicture(img)
				if err != nil {
					t.Fatal(err)
				}
				first = slices.Clone(rows)
				if spoil {
					l := &m.Layers[m.TowerBase]
					if len(l.Mats) == 0 {
						t.Fatal("the tower's first block holds no matrix")
					}
					qs, _, _ := s.fr.Span(l.Mats[0].T.E)
					if len(qs) == 0 {
						t.Fatal("the tower's first block is not resident after an encode")
					}
					for i := range qs {
						qs[i] ^= 0x5a
					}
				}
				_, in0, _ := m.PageStats(s.fr)
				if rows, err = s.EncodePicture(img); err != nil {
					t.Fatal(err)
				}
				_, in1, _ := m.PageStats(s.fr)
				return first, slices.Clone(rows), in1 - in0
			}
			want, again, read := encode(0, false)
			if read != 0 {
				t.Fatalf("the resident tower read %d pages on its second encode", read)
			}
			sameWords(t, "the resident tower's second encode", again, want)
			if _, spoilt, _ := encode(0, true); slices.Equal(spoilt, want) {
				t.Fatal("VIOLATION: a resident tower block's page overwritten encodes the same rows: the rows are not read from the pages")
			}

			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			page, nt, chain := uint64(0), len(m.Tower.Blocks), m.Tower.Chain
			// The tower's own largest page: a text block's can hold the whole
			// tower, which then never evicts.
			for li := m.TowerBase; li < m.TowerBase+nt; li++ {
				page = max(page, m.Container.PageBytes(li))
			}
			m.Close()
			if nt < 2 && !chain {
				t.Fatalf("a tower of %d block evicts nothing under one page", nt)
			}
			got, got2, read := encode(page, false)
			sameWords(t, "under one page", got, want)
			sameWords(t, "under one page, again", got2, want)
			if read == 0 {
				t.Fatalf("under a budget of one page (%d bytes) the second encode read no page: the tower stayed resident and the arm proved nothing", page)
			}
			t.Logf("%d tower blocks: the rows bit for bit under one page of %d bytes, %d pages read again; a spoilt resident page moves them", nt, page, read)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no tower ran")
	}
}

// TestTowersMoveWithTheirRows is the relocation principle for every stated
// tower whose blocks are the instance's (a chain's are forms placed with the
// chain, TestStatedTowerRunsOnTheCard): its blocks placed on a card, then
// moved to the CPU (MoveBlocks) and back between encodes. The entry and the
// head stay where they were placed: on the CPU the blocks' rows are within
// the band of the CPU device's (RULE 11c), and not the card's; back on the
// card they are the card's first encode's bit for bit; on the card, within
// the same band. A grey picture past the band is the violation.
func TestTowersMoveWithTheirRows(t *testing.T) {
	img := principlePicture()
	ran := 0
	for _, src := range towerFixtures(t) {
		t.Run(src.name, func(t *testing.T) {
			m, err := model.Open(pairContainer(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Tower.Chain {
				t.Skip("a chain's blocks are forms placed with the chain (TestStatedTowerRunsOnTheCard)")
			}
			host := testInstance(t, m, 64)
			want, err := host.EncodePicture(img)
			if err != nil {
				t.Fatal(err)
			}
			want = slices.Clone(want)
			grey := image.NewGray(img.Bounds())
			for i := range grey.Pix {
				grey.Pix[i] = 128
			}
			blank, err := host.EncodePicture(grey)
			if err != nil {
				t.Fatal(err)
			}
			blank = slices.Clone(blank)
			host.Close()
			const bound = 1e-3
			if bad, _ := nmseOf(blank, want); !(bad > 10*bound) {
				t.Fatalf("VIOLATION: a grey picture's rows sit at NMSE %.3e from the picture's, within ten times the bound %.0e", bad, bound)
			}
			nb := len(m.Tower.Blocks)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					defer g.Close()
					s := testInstance(t, m, 64)
					defer s.Close()
					s.SetDeviceFallback(false)
					ts := s.tower.s
					if err := ts.SetDeviceLayers(g, nb); err != nil {
						t.Fatal(err)
					}
					if n := ts.GPULayers(); n != nb {
						t.Fatalf("%d of the tower's %d blocks placed on %s: %s", n, nb, dev, g.Err())
					}
					card, err := s.EncodePicture(img)
					if err != nil {
						t.Fatal(err)
					}
					card = slices.Clone(card)
					if e, _ := nmseOf(card, want); math.IsNaN(e) || e > bound {
						t.Fatalf("the tower on %s: NMSE %.3e from the CPU device's rows (bound %.0e)", dev, e, bound)
					}
					base := m.TowerBase
					mv, err := ts.MoveBlocks(base, base+nb, nn.CPU)
					if err != nil {
						t.Fatal(err)
					}
					if len(mv.Moved) != nb || ts.GPULayers() != 0 {
						t.Fatalf("%d of the tower's %d blocks moved to the CPU, %d left on %s", len(mv.Moved), nb, ts.GPULayers(), dev)
					}
					back, err := s.EncodePicture(img)
					if err != nil {
						t.Fatal(err)
					}
					if e, _ := nmseOf(back, want); math.IsNaN(e) || e > bound {
						t.Fatalf("the tower's blocks moved to the CPU: NMSE %.3e from the CPU device's rows (bound %.0e)", e, bound)
					}
					if slices.Equal(back, card) {
						t.Fatal("the rows after the move to the CPU are the card's bit for bit: no block ran on the CPU")
					}
					if mv, err = ts.MoveBlocks(base, base+nb, dev); err != nil {
						t.Fatal(err)
					}
					if len(mv.Moved) != nb || ts.GPULayers() != nb {
						t.Fatalf("%d of the tower's %d blocks moved back to %s, %d on it", len(mv.Moved), nb, dev, ts.GPULayers())
					}
					again, err := s.EncodePicture(img)
					if err != nil {
						t.Fatal(err)
					}
					sameWords(t, "moved back to "+dev, again, card)
					if st := g.Stats(); st.UnformedAtStep != 0 {
						t.Fatalf("%d of the tower's parts formed at a step on %s, not at the placement", st.UnformedAtStep, dev)
					}
					e, _ := nmseOf(card, want)
					t.Logf("%s: %d tower blocks to the CPU and back; the card's rows bit for bit after, the card %.3e from the CPU device's", dev, nb, e)
				})
			}
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no tower ran")
	}
}

// TestEveryTowerRestoresPastThePicture is the prefix cache past a picture for
// every stated tower, a hybrid's (synth-qwen35vl) among them: a prompt that
// extends one carrying a picture restores past the picture's rows from the
// first's pages -- a hybrid where it sealed its summary, at the first
// prompt's end -- encodes no picture, and decodes the tokens a cold run does.
// Against the violations nothing past the picture restores: its rows with no
// name, and a picture one pixel apart.
func TestEveryTowerRestoresPastThePicture(t *testing.T) {
	img := principlePicture()
	other := image.NewNRGBA(img.Bounds())
	draw.Draw(other, other.Bounds(), img, img.Bounds().Min, draw.Src)
	other.Pix[0] ^= 0x40
	for _, src := range towerFixtures(t) {
		t.Run(src.name, func(t *testing.T) {
			m, err := model.Open(pairContainer(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			pre := m.Vocab.Encode("Look at this picture:", true)
			post := m.Vocab.Encode(" It is a gradient of colours from one corner to the other, with no words in it and nothing drawn over it.", false)
			more := m.Vocab.Encode(" What colour is its top left corner?", false)
			type result struct {
				restored, imgAt, imgEnd, n, encodes int
				toks                                []int32
			}
			run := func(store KVStore, extend bool, pic image.Image, named bool) result {
				t.Helper()
				st := testInstance(t, m, 1024)
				defer st.Close()
				if st.tower == nil {
					t.Fatal("the instance runs no tower")
				}
				st.SetKVStore(store)
				mustKey(t, st, "tower/restore")
				sp, err := m.PictureSpan(pic)
				if err != nil {
					t.Fatal(err)
				}
				if !named {
					rows, err := st.EncodePicture(pic)
					if err != nil {
						t.Fatal(err)
					}
					sp = model.Span{Embd: slices.Clone(rows)}
				}
				spans := []model.Span{{Tokens: pre}, sp, {Tokens: post}}
				if extend {
					spans = append(spans, model.Span{Tokens: more})
				}
				r := result{imgAt: len(pre)}
				r.imgEnd = r.imgAt + sp.N(m.Cfg.NEmbd)
				before := st.tower.encodes
				lg, err := st.PrefillCachedMixed(spans...)
				if err != nil {
					t.Fatal(err)
				}
				r.encodes = st.tower.encodes - before
				r.restored, r.n = st.KVRestored(), model.SpanPositions(spans, m.Cfg.NEmbd)
				for range 8 {
					id := Greedy(lg)
					r.toks = append(r.toks, id)
					if lg, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				return r
			}
			cold := run(NewMemStore(), true, img, true)
			if cold.encodes != 1 {
				t.Fatalf("the cold prompt encoded %d pictures, want its one", cold.encodes)
			}
			store := NewMemStore()
			t1 := run(store, false, img, true)
			t2 := run(store, true, img, true)
			if t2.restored <= t1.imgEnd {
				t.Fatalf("the extended prompt restored %d positions of the first's %d, not past the picture at %d..%d",
					t2.restored, t1.n, t1.imgAt, t1.imgEnd)
			}
			if t2.encodes != 0 {
				t.Fatalf("the extended prompt restored the picture and still encoded %d", t2.encodes)
			}
			if !slices.Equal(t2.toks, cold.toks) {
				t.Fatalf("the restored prompt decodes %v, a cold one %v", t2.toks, cold.toks)
			}
			if diff := run(store, true, other, true); diff.restored > diff.imgAt {
				t.Fatalf("VIOLATION: a picture a pixel apart restored %d positions, past its start at %d", diff.restored, diff.imgAt)
			}
			if unnamed := run(store, true, img, false); unnamed.restored > unnamed.imgAt {
				t.Fatalf("VIOLATION: unnamed rows restored %d positions, past the picture's start at %d", unnamed.restored, unnamed.imgAt)
			}
			t.Logf("the first prompt %d positions (picture %d..%d); the extended one restored %d of %d, no picture encoded, tokens a cold run's",
				t1.n, t1.imgAt, t1.imgEnd, t2.restored, t2.n)
		})
	}
}

// TestDecodeAfterAPictureDoesNotAllocate is the allocation principle past a
// picture, for every stated tower: a prompt carrying one, its rows the
// tower's, then tokens decoded after it, the warm ones making no engine heap
// allocation (RULE 2f). The counter must see the prefill's, which encodes the
// picture and grows its buffers.
func TestDecodeAfterAPictureDoesNotAllocate(t *testing.T) {
	img := principlePicture()
	for _, src := range towerFixtures(t) {
		t.Run(src.name, func(t *testing.T) {
			m, err := model.Open(pairContainer(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			pspan, err := m.PictureSpan(img)
			if err != nil {
				t.Fatal(err)
			}
			spans := []model.Span{{Tokens: m.Vocab.Encode("Describe the picture:", true)}, pspan,
				{Tokens: m.Vocab.Encode(" It shows", false)}}
			const warm, n = 16, 32
			s := testInstance(t, m, model.SpanPositions(spans, m.Cfg.NEmbd)+warm+n+8)
			defer s.Close()
			var lg []float32
			cold := countAllocs(func() {
				if lg, err = s.PrefillMixed(spans...); err != nil {
					t.Fatal(err)
				}
			})
			if cold.engine == 0 {
				t.Fatal("VIOLATION: the prefill, which encodes the picture and grows its buffers, counted no allocation")
			}
			step := func() {
				if lg, err = s.Forward(int32(argmax(lg))); err != nil {
					t.Fatal(err)
				}
			}
			for range warm {
				step()
			}
			got := countAllocs(func() {
				for range n {
					step()
				}
			})
			if got.engine != 0 {
				t.Fatalf("%d warm tokens after a picture made %d engine allocations:\n%s", n, got.engine, got.where)
			}
			t.Logf("the prefill %d engine allocations; %d warm tokens after the picture %d", cold.engine, n, got.engine)
		})
	}
}
