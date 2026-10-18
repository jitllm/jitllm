package model

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// clefPictureTol bounds a Clef request with pictures or a video against the
// reference at the engine's arithmetic: both sides round every tower
// activation to q8, from f32 values summed in different orders, so a value at
// a rounding edge goes a step apart. The text-only fixture reads 1.7e-6 and
// the pictures 5.5e-4, a sixtieth of what leaving them out moves.
const clefPictureTol = 2e-3

// TestClefMediaMatchesReference holds Clef requests carrying media to Clef's
// own code on a random fixture whose toy tower has the release's position
// table (clefgold.py fixture-vl, the release's Qwen3VLProcessor): two
// pictures (pictures.json), and a video of ten frames with no frame rate
// (video.json: four sampled at the 24 frames a second the reference assumes,
// two temporal pairs of different frames). Each is held row for row -- the
// media's markers, stamps and rows after "STATE:\n", a newline, then the
// state -- span for span, and logit for logit at the engine's arithmetic (the
// reference's tower inputs and matrices rounded as the container's are,
// logits_a8), on the host and with every block on each device. The
// violations: the media left out; for the video, each pair's second frame
// read as its first (the still-picture path), which the second temporal
// plane must see.
func TestClefMediaMatchesReference(t *testing.T) {
	text := testmodels.Path(filepath.Join("clef", "fixture-vl", "clef-fixture-vl.gguf"))
	proj := testmodels.Path(filepath.Join("clef", "fixture-vl", "mmproj-clef-fixture-vl.gguf"))
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- run scripts/clefgold.py fixture-vl (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Tower() == nil || m.Decision() != jlm.DecisionClef || m.Tower().patchW1.e == nil {
		t.Fatal("the container carries no tower, no Clef head or no second temporal plane, so this gate proves nothing")
	}
	for _, name := range []string{"pictures", "video"} {
		t.Run(name, func(t *testing.T) {
			body, err := os.ReadFile(filepath.Join("testdata", "decision", name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			req, err := ParseDecisionRequest(body)
			if err != nil {
				t.Fatal(err)
			}
			imgs := make([]image.Image, len(req.Images))
			for i, u := range req.Images {
				if imgs[i], err = DecodeImageDataURL(u); err != nil {
					t.Fatal(err)
				}
			}
			var vids []VideoFrames
			for _, v := range req.Videos {
				vf := VideoFrames{FPS: v.FPS}
				for _, u := range v.Frames {
					f, err := DecodeImageDataURL(u)
					if err != nil {
						t.Fatal(err)
					}
					vf.Frames = append(vf.Frames, f)
				}
				vids = append(vids, vf)
			}
			if len(imgs)+len(vids) == 0 {
				t.Fatal("the request carries no media, so this gate proves nothing")
			}
			g := readClefGoldOf(t, "clef-fixture-vl", name)
			if len(g.LogitsA8) == 0 {
				t.Fatal("the golden has no logits_a8: regenerate it with scripts/clefgold.py fixture-vl")
			}
			a8 := g
			a8.Logits = g.LogitsA8
			run := func(t *testing.T, gpu *tier.GPU, imgs []image.Image, vids []VideoFrames) (*clefPrompt, []float32) {
				d, err := m.NewDecider(4096)
				if err != nil {
					t.Fatal(err)
				}
				defer d.Close()
				if gpu != nil {
					if err := d.SetDevice(gpu, -1); err != nil {
						t.Fatal(err)
					}
					if d.DeviceBlocks() != m.Cfg.NLayer {
						t.Fatalf("the device took %d of %d blocks (%s)", d.DeviceBlocks(), m.Cfg.NLayer, gpu.Err())
					}
				}
				if err := d.Prepare(req.State, req.Questions, len(imgs)); err != nil {
					t.Fatal(err)
				}
				if err := d.checkVideos(len(vids), len(imgs)); err != nil {
					t.Fatal(err)
				}
				if len(imgs) > 0 {
					if d.pics, err = d.cutPictures(imgs); err != nil {
						t.Fatal(err)
					}
				}
				if len(vids) > 0 {
					if d.vids, err = d.cutVideos(vids); err != nil {
						t.Fatal(err)
					}
				}
				p, err := d.clefPrompt(req.State, req.Questions)
				if err != nil {
					t.Fatal(err)
				}
				out := make([]float32, len(p.opts))
				if err := d.clefForward(p, out); err != nil {
					t.Fatal(err)
				}
				return p, out
			}
			check := func(t *testing.T, gpu *tier.GPU) (float64, float64) {
				p, got := run(t, gpu, imgs, vids)
				if len(p.ids) != len(g.IDs) {
					t.Fatalf("the prompt is %d rows, the reference's %d", len(p.ids), len(g.IDs))
				}
				for i := range p.ids {
					if p.ids[i] != g.IDs[i] {
						t.Fatalf("row %d is %d, the reference's %d", i, p.ids[i], g.IDs[i])
					}
				}
				at := 0
				for qi, q := range g.Questions {
					if s := p.qs[qi]; s.lo != q.Span[0] || s.hi != q.Span[1] {
						t.Errorf("%s: span [%d,%d), the reference's %v", q.ID, s.lo, s.hi, q.Span)
					}
					for _, o := range q.Options {
						if s := p.opts[at]; s.lo != o[0] || s.hi != o[1] {
							t.Errorf("%s option %d: span [%d,%d), the reference's %v", q.ID, at, s.lo, s.hi, o)
						}
						at++
					}
				}
				return clefWorst(t, a8, got), clefWorst(t, g, got)
			}
			clean, f32 := check(t, nil)
			t.Logf("host: %d picture(s), %d video(s): worst logit difference from Clef's own code %.2e at the "+
				"engine's arithmetic, %.2e against its f32 tower", len(imgs), len(vids), clean, f32)
			if clean > clefPictureTol {
				t.Fatalf("host: worst logit %.2e from the reference, tolerance %.0e", clean, clefPictureTol)
			}
			_, bare := run(t, nil, nil, nil)
			if w := clefWorstLoose(a8, bare); !(w > 10*max(clean, clefFixtureTol)) {
				t.Errorf("the media left out move the logits by %.2e: the gate cannot see them", w)
			} else {
				t.Logf("violation, the media left out: %.2e", w)
			}
			if len(vids) > 0 {
				m.Tower().Cfg.fault = faultVideoFirstFrame
				_, still := run(t, nil, imgs, vids)
				m.Tower().Cfg.fault = towerFaultNone
				w := clefWorstLoose(a8, still)
				t.Logf("violation, each pair's second frame read as its first: %.2e", w)
				if !(w > 10*max(clean, clefFixtureTol)) {
					t.Errorf("the second frame read as the first moves the logits by %.2e against a clean %.2e", w, clean)
				}
			}
			for _, spec := range stepDevices() {
				gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || gpu == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					w, _ := check(t, gpu)
					t.Logf("%s, every block: worst logit %.2e from the reference", spec, w)
					if w > clefPictureTol {
						t.Fatalf("%s: worst logit %.2e", spec, w)
					}
				})
			}
		})
	}
}
