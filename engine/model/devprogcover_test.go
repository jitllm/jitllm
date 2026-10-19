package model

import (
	"cmp"
	"fmt"
	"image"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDeviceProgramCoverage reports, for every synthetic fixture whose
// blocks a device takes, whether its first device block runs as its device program, and
// where it does not, why: the work list the block encoder's deletion (D5)
// drives to empty. A report until then; at D5 it fails while any fixture
// still runs the encoder.
func TestDeviceProgramCoverage(t *testing.T) {
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda")}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, "cuda", err)
	}
	defer g.Close()
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	reasons := map[string][]string{}
	covered, tried := 0, 0
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".jlm")
		if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
			continue
		}
		m, err := Open(p, noTune)
		if err != nil {
			continue
		}
		func() {
			defer m.Close()
			if m.IsEncoder() {
				return
			}
			s := m.NewState(32)
			defer s.Close()
			if err := s.SetDevice(g); err != nil || s.GPULayers() == 0 {
				return
			}
			tried++
			if _, err := s.Prefill(m.Vocab.Encode("Hello", true)); err != nil {
				reasons["prefill: "+err.Error()] = append(reasons["prefill: "+err.Error()], name)
				return
			}
			// Every block the device holds: compiled means each one's
			// program holds primitives alone. The first that does not names
			// the fixture's reason.
			why := ""
			for _, r := range s.DeviceBlocks() {
				for li := r[0]; li < r[1] && why == ""; li++ {
					prog, w := g.BlockProgram(li)
					if c := g.BlockComposite(li); prog != "" && c != "" {
						w = "composite op " + c
					}
					if prog == "" && w == "" {
						w = fmt.Sprintf("(no program, no reason: block %d of runs %v, %d device blocks; declines %v; device %q)",
							li, s.DeviceBlocks(), s.GPULayers(), s.DeviceDeclines(), s.devWhy())
					}
					why = w
				}
			}
			if why == "" {
				covered++
				return
			}
			reasons[why] = append(reasons[why], name)
		}()
		g.ReleaseLayers(0, 1<<20)
	}
	keys := make([]string, 0, len(reasons))
	for k := range reasons {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b string) int { return cmp.Compare(len(reasons[b]), len(reasons[a])) })
	t.Logf("%d of %d fixtures run every device block as a device program of primitives", covered, tried)
	for _, k := range keys {
		t.Logf("%3d  %s: %s", len(reasons[k]), k, strings.Join(reasons[k], " "))
	}
}

// programWhy is the first reason one of blocks [lo, hi) does not run as a
// device program of primitives, empty when every one does.
func programWhy(g *tier.GPU, lo, hi int) string {
	for li := lo; li < hi; li++ {
		prog, w := g.BlockProgram(li)
		if c := g.BlockComposite(li); prog != "" && c != "" {
			w = "composite op " + c
		}
		if prog == "" && w == "" {
			w = fmt.Sprintf("(no program, no reason: block %d)", li)
		}
		if w != "" {
			return w
		}
	}
	return ""
}

// TestDeviceProgramCoverageBeyondText is TestDeviceProgramCoverage's report
// for the blocks that are not a decoder's: every merged fixture's vision
// tower, and every embedding encoder's blocks. The compiler lowers by
// structure, so a tower block and an encoder block are on its work list as a
// text block is.
func TestDeviceProgramCoverageBeyondText(t *testing.T) {
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda")}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, "cuda", err)
	}
	defer g.Close()
	img := image.NewRGBA(image.Rect(0, 0, 96, 80))
	for i := range img.Pix {
		img.Pix[i] = uint8(i * 37)
	}
	reasons := map[string][]string{}
	covered, tried := 0, 0
	note := func(name, why string) {
		tried++
		if why == "" {
			covered++
			return
		}
		reasons[why] = append(reasons[why], name)
	}
	towers, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*-vlm.jlm"))
	for _, p := range towers {
		name := "tower " + strings.TrimSuffix(filepath.Base(p), "-vlm.jlm")
		m, err := Open(p, noTune)
		if err != nil {
			reasons["open: "+err.Error()] = append(reasons["open: "+err.Error()], name)
			continue
		}
		func() {
			defer m.Close()
			s := m.NewState(64)
			defer s.Close()
			if err := s.SetDevice(g); err != nil {
				reasons["device: "+err.Error()] = append(reasons["device: "+err.Error()], name)
				return
			}
			vs, err := s.Vision()
			if err != nil {
				reasons["vision: "+err.Error()] = append(reasons["vision: "+err.Error()], name)
				return
			}
			// A picture the tower takes: the test picture, or a smaller one
			// where its patch budget is smaller; an LFM2 tower by pieces.
			if _, _, pcs, perr := m.Tower().LFM2Pieces(img); perr == nil && len(pcs) > 0 {
				_, err = vs.encodePicture(pcs[0])
			} else {
				for _, im := range []image.Image{img, image.NewRGBA(image.Rect(0, 0, 32, 32))} {
					var px []float32
					if px, err = vs.PreprocessImage(im); err == nil {
						if _, err = vs.EncodeImage(px); err == nil {
							break
						}
					}
				}
			}
			if err != nil {
				reasons["encode: "+err.Error()] = append(reasons["encode: "+err.Error()], name)
				return
			}
			why := ""
			for _, r := range vs.DeviceBlocks() {
				if why = programWhy(g, r[0], r[1]); why != "" {
					break
				}
			}
			if vs.GPUBlocks() == 0 {
				why = "the device took no tower block"
			}
			note(name, why)
		}()
		g.ReleaseLayers(0, 1<<20)
	}
	encoders, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "embed", "*.jlm"))
	for _, p := range encoders {
		name := "encoder " + strings.TrimSuffix(filepath.Base(p), ".jlm")
		m, err := Open(p, noTune)
		if err != nil {
			continue
		}
		func() {
			defer m.Close()
			if !m.IsEncoder() {
				return // a decoder embedding model: TestDeviceProgramCoverage's
			}
			e, err := m.NewEmbedder()
			if err != nil {
				return
			}
			defer e.Close()
			if err := e.SetDeviceLayers(g, -1); err != nil {
				reasons["device: "+err.Error()] = append(reasons["device: "+err.Error()], name)
				return
			}
			if _, err := e.Embed(m.EmbedIDs("Hello there")); err != nil {
				reasons["embed: "+err.Error()] = append(reasons["embed: "+err.Error()], name)
				return
			}
			why := programWhy(g, 0, e.DeviceBlocks())
			if e.DeviceBlocks() == 0 {
				why = "the device took no encoder block"
			}
			note(name, why)
		}()
		g.ReleaseLayers(0, 1<<20)
	}
	keys := make([]string, 0, len(reasons))
	for k := range reasons {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b string) int { return cmp.Compare(len(reasons[b]), len(reasons[a])) })
	t.Logf("%d of %d towers and encoders run every device block as a device program of primitives", covered, tried)
	for _, k := range keys {
		t.Logf("%3d  %s: %s", len(reasons[k]), k, strings.Join(reasons[k], " "))
	}
}
