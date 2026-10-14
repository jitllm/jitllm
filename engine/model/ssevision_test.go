//go:build amd64 && linux && jitllmtest

package model

import (
	"github.com/jitllm/jitllm/format/jlm"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/jit/cpu"
)

// TestEveryTowerRunsSSE runs every tower family's preprocessing and encode on
// the SSE tier against the AVX2 tier, on the runner's one block body: the
// tier is forced before the model opens (its JIT builds the kernels), no AVX2
// kernel may be mapped while it is on, and SSE-tier kernels must be.
//
// The Pillow and torchvision resamplers are integer arithmetic
// (cpu.EmitResample), so a picture preprocessed on either tier is the same
// bits; the bilinear families resize on the generated F32 matvec, which has no
// FMA on SSE, and are held to a float bound. The encode is held to the bound
// the device is held to over a whole tower: the tiers quantize the activations
// to int8 from floats a few ulps apart, and a few flips compound over the
// blocks. The picture encoded is the preprocessed one (the first piece of a
// slicing tower's), so a bilinear family's arms read pixels a few ulps apart.
func TestEveryTowerRunsSSE(t *testing.T) {
	ssemRequireComplete(t)
	if cpu.HostTier() != cpu.TierAVX2 {
		t.Skip("no AVX2 tier on this host to hold the SSE tier to")
	}
	photo := decodeImage(t, mcvPhoto)
	type arm struct {
		pixels      [][]float32
		after0, out []float32
	}
	run := func(t *testing.T, f towerFamily) arm {
		m, tw := f.open(t)
		defer m.Close()
		st := m.NewState(64)
		defer st.Close()
		var a arm
		var first *Picture
		if tw.Cfg.PosBuckets > 0 {
			lay, err := tw.Layout(photo.Bounds().Dx(), photo.Bounds().Dy())
			if err != nil {
				t.Fatal(err)
			}
			ps, err := tw.PicturePieces(photo, lay)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range ps {
				a.pixels = append(a.pixels, p.Pixels)
			}
			first = ps[0]
		} else if tw.Cfg.Kind == jlm.ProjLFM2 {
			_, _, ps, err := tw.LFM2Pieces(photo)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range ps {
				a.pixels = append(a.pixels, p.Pixels)
			}
			first = ps[0]
		} else {
			p, err := st.Picture(photo)
			if err != nil {
				t.Fatal(err)
			}
			a.pixels, first = [][]float32{p.Pixels}, p
		}
		vs, err := st.Vision()
		if err != nil {
			t.Fatal(err)
		}
		vs.vis.afterBlock = func(li int, x []float32) {
			if li == 0 && a.after0 == nil {
				a.after0 = append([]float32(nil), x...)
			}
		}
		out, err := vs.encodePicture(first)
		if err != nil {
			t.Fatal(err)
		}
		a.out = append([]float32(nil), out...)
		return a
	}
	for _, f := range towerFamilies() {
		t.Run(f.name, func(t *testing.T) {
			ref := run(t, f)
			old := cpu.ForceTierForTest(cpu.TierSSE)
			defer cpu.ForceTierForTest(old)
			before := cpu.MappedByTier()
			got := run(t, f)
			after := cpu.MappedByTier()
			if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
				t.Fatalf("%d AVX2 kernels were mapped during the SSE arm -- it did not run the SSE tier", d)
			}
			if after[cpu.TierSSE] == before[cpu.TierSSE] {
				t.Fatal("the SSE arm mapped no SSE-tier kernel -- it ran nothing it claims to")
			}
			bilinear := f.name == "smolvlm" || f.name == "llava"
			for i := range ref.pixels {
				if !bilinear {
					if !slices.Equal(got.pixels[i], ref.pixels[i]) {
						t.Fatalf("piece %d: the SSE tier's integer resample is not the AVX2 tier's bits", i)
					}
					continue
				}
				if nm, _ := nmseOf(got.pixels[i], ref.pixels[i]); math.IsNaN(nm) || nm > 1e-10 {
					t.Fatalf("piece %d: the SSE tier's bilinear resize reads NMSE %.3e against AVX2's", i, nm)
				}
			}
			nm, worst := nmseOf(got.out, ref.out)
			n0, _ := nmseOf(got.after0, ref.after0)
			t.Logf("%s: pixels %s; after block 0 NMSE %.3e, the encode NMSE %.3e (max|d| %.4f) against the AVX2 tier",
				f.name, map[bool]string{true: "within float rounding", false: "bit-identical"}[bilinear], n0, nm, worst)
			// One block alone, before the tiers' int8 roundings compound, is the
			// bound that sees a wrong graph; the whole tower's is the device's.
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 2e-2 || !(n0 <= 1e-4) {
				t.Fatalf("%s: the SSE tier reads NMSE %.3e after block 0 and %.3e over the tower against the "+
					"AVX2 tier's", f.name, n0, nm)
			}
		})
	}
}
