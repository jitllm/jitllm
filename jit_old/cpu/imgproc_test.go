//go:build amd64 || arm64

package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestResampleHMatchesTheReference gates the horizontal resampling pass
// (EmitResampleH) on every tier this host executes against its formula
// written out: ragged row widths, window counts and tap counts, several rows
// at strides that are not the width, negative taps and sums past both ends of
// 0..255. The input's last row ends at a PROT_NONE page, so a fourth byte
// read past a pixel faults, and the output carries guards between rows and
// after the last. Two violations -- the reference shifting one bit less, and
// reading every window one pixel later -- must each fail it.
func TestResampleHMatchesTheReference(t *testing.T) {
	type shape struct{ w, ow, cols, rows int }
	var shapes []shape
	for _, w := range []int{1, 2, 3, 5, 8, 9, 17} {
		for _, ow := range []int{1, 2, 4, 7, 13} {
			for cols := 1; cols <= min(w, 6); cols += 2 {
				for _, rows := range []int{1, 3} {
					shapes = append(shapes, shape{w, ow, cols, rows})
				}
			}
		}
	}
	ref := func(out, in []uint8, s shape, inStr, outStr int, w []int32, lo []int64, prec uint, shift int) {
		for r := 0; r < s.rows; r++ {
			for x := 0; x < s.ow; x++ {
				for c := 0; c < 3; c++ {
					acc := int32(1) << (prec - 1)
					for j := 0; j < s.cols; j++ {
						at := r*inStr + int(lo[x]) + 3*j + c + 3*shift
						if at >= len(in) {
							at = len(in) - 1
						}
						acc += w[x*s.cols+j] * int32(in[at])
					}
					out[r*outStr+3*x+c] = uint8(min(max(acc>>prec, 0), 255))
				}
			}
		}
	}
	for _, tr := range imgTiers() {
		for _, violate := range []int{0, 1, 2} {
			bad := 0
			for _, prec := range []uint{7, 13, 22} {
				b, err := tr.em.ResampleH(int(prec))
				c := imgMap(t, tr, "resample_h", b, err)
				r := rand.New(rand.NewSource(int64(prec)))
				scr := ResampleConsts(int(prec))
				for _, s := range shapes {
					inStr, outStr := 3*s.w+4, 3*s.ow+2
					src := make([]uint8, (s.rows-1)*inStr+3*s.w)
					for i := range src {
						src[i] = uint8(r.Intn(256))
					}
					in := imgGuard(t, src)
					w := make([]int32, s.ow*s.cols)
					for i := range w {
						w[i] = int32(r.Intn(1<<(prec+1))) - 1<<prec/2
					}
					lo := make([]int64, s.ow)
					for i := range lo {
						lo[i] = 3 * int64(r.Intn(s.w-s.cols+1))
					}
					const guard = 0xA5
					got := make([]uint8, (s.rows-1)*outStr+3*s.ow+8)
					for i := range got {
						got[i] = guard
					}
					want := append([]uint8(nil), got...)
					rp, shift := prec, 0
					switch violate {
					case 1:
						rp = prec - 1
					case 2:
						shift = 1
					}
					ref(want, in, s, inStr, outStr, w, lo, rp, shift)
					c.Call(&Args{
						Out:     (*float32)(unsafe.Pointer(&got[0])),
						AScale:  (*float32)(unsafe.Pointer(&in[0])),
						W:       (*byte)(unsafe.Pointer(&w[0])),
						AScale2: (*float32)(unsafe.Pointer(&lo[0])),
						Cols:    int64(s.cols),
						K:       int64(s.ow),
						Rows:    int64(s.rows),
						RowStr:  int64(inStr),
						OutStr:  int64(outStr),
						Scr:     (*byte)(unsafe.Pointer(&scr[0])),
					})
					for i := range got {
						inPixel := i < len(got)-8 && i%outStr < 3*s.ow
						if !inPixel && got[i] != guard {
							t.Fatalf("%s prec %d %+v: byte %d is not the output's and was written", tr.name, prec, s, i)
						}
						if inPixel && got[i] != want[i] {
							bad++
							if violate == 0 {
								t.Fatalf("%s prec %d %+v: byte %d is %d, want %d", tr.name, prec, s, i, got[i], want[i])
							}
						}
					}
				}
				c.Close()
			}
			if violate > 0 && bad == 0 {
				t.Fatalf("%s: violation %d agreed everywhere -- this gate proves nothing", tr.name, violate)
			}
		}
		t.Logf("%s: %d shapes at precisions 7/13/22 exact, guarded; both violations differ", tr.name, len(shapes))
	}
}

// TestPixLUTMatchesTheReference gates the per-channel table lookup
// (EmitPixLUT): every sample count to 40, whole pixels and the one or two
// samples after them, the samples at a PROT_NONE page, guards after the
// output. Reading the next channel's table must fail it.
func TestPixLUTMatchesTheReference(t *testing.T) {
	for _, tr := range imgTiers() {
		b, err := tr.em.PixLUT()
		c := imgMap(t, tr, "pixlut", b, err)
		r := rand.New(rand.NewSource(3))
		lut := make([]float32, 3*pixLUTSize)
		for i := range lut {
			lut[i] = r.Float32()*4 - 2
		}
		for _, violate := range []bool{false, true} {
			bad := 0
			for n := 1; n <= 40; n++ {
				src := make([]uint8, n)
				for i := range src {
					src[i] = uint8(r.Intn(256))
				}
				in := imgGuard(t, src)
				guard := float32(math.NaN())
				got := make([]float32, n+3)
				for i := range got {
					got[i] = guard
				}
				c.Call(&Args{
					Out:    &got[0],
					W:      &in[0],
					AScale: &lut[0],
					K:      int64(n / 3),
					Rows:   int64(n % 3),
				})
				for i := n; i < len(got); i++ {
					if !math.IsNaN(float64(got[i])) {
						t.Fatalf("%s n %d: wrote past the output at %d", tr.name, n, i)
					}
				}
				for i := 0; i < n; i++ {
					ch := i % 3
					if violate {
						ch = (ch + 1) % 3
					}
					if want := lut[pixLUTSize*ch+int(in[i])]; math.Float32bits(got[i]) != math.Float32bits(want) {
						bad++
						if !violate {
							t.Fatalf("%s n %d: sample %d is %v, want %v", tr.name, n, i, got[i], want)
						}
					}
				}
			}
			if violate && bad == 0 {
				t.Fatalf("%s: the next channel's table agreed everywhere -- this gate proves nothing", tr.name)
			}
		}
		c.Close()
		t.Logf("%s: every count to 40 bit-exact, guarded; the channel violation differs", tr.name)
	}
}

// TestCopy32MatchesTheReference gates the strided copy (EmitCopy32): rows and
// elements from zero, strides that gather, scatter and transpose, guards
// between the strided outputs and after them, the source at a PROT_NONE page.
// Swapping the source's two strides must fail it.
func TestCopy32MatchesTheReference(t *testing.T) {
	type shape struct{ rows, k, rowStr, el, outStr, outEl int }
	shapes := []shape{
		{1, 1, 4, 4, 4, 4}, {1, 9, 36, 4, 36, 4}, {3, 5, 60, 12, 20, 4}, // gather a channel
		{4, 7, 28, 4, 4, 16}, {7, 4, 16, 4, 4, 28}, // transposes
		{1, 11, 44, 4, 4, 33 * 4}, {2, 3, 40, 8, 64, 16}, {5, 1, 8, 4, 12, 4},
		{0, 3, 4, 4, 4, 4}, {3, 0, 4, 4, 4, 4},
	}
	for _, tr := range imgTiers() {
		b, err := tr.em.Copy32()
		c := imgMap(t, tr, "copy32", b, err)
		r := rand.New(rand.NewSource(5))
		for _, violate := range []bool{false, true} {
			bad := 0
			for _, s := range shapes {
				ext := func(rows, k, rs, es int) int {
					if rows == 0 || k == 0 {
						return 4
					}
					return (rows-1)*rs + (k-1)*es + 4
				}
				srcN := max(ext(s.rows, s.k, s.rowStr, s.el), ext(s.rows, s.k, s.el, s.rowStr))
				src := make([]byte, srcN)
				for i := range src {
					src[i] = byte(r.Intn(256))
				}
				in := imgGuard(t, src)
				const guard = 0x5A
				got := make([]byte, ext(s.rows, s.k, s.outStr, s.outEl)+16)
				for i := range got {
					got[i] = guard
				}
				want := append([]byte(nil), got...)
				rs, es := s.rowStr, s.el
				if violate {
					rs, es = es, rs
				}
				for y := 0; y < s.rows; y++ {
					for i := 0; i < s.k; i++ {
						copy(want[y*s.outStr+i*s.outEl:y*s.outStr+i*s.outEl+4], in[y*rs+i*es:])
					}
				}
				c.Call(&Args{
					Out:    (*float32)(unsafe.Pointer(&got[0])),
					AScale: (*float32)(unsafe.Pointer(&in[0])),
					Rows:   int64(s.rows),
					K:      int64(s.k),
					RowStr: int64(s.rowStr),
					Cols:   int64(s.el),
					OutStr: int64(s.outStr),
					DStr:   int64(s.outEl),
				})
				for i := range got {
					if got[i] != want[i] {
						bad++
						if !violate {
							t.Fatalf("%s %+v: byte %d is %#x, want %#x", tr.name, s, i, got[i], want[i])
						}
					}
				}
			}
			if violate && bad == 0 {
				t.Fatalf("%s: swapped strides agreed everywhere -- this gate proves nothing", tr.name)
			}
		}
		c.Close()
		t.Logf("%s: %d shapes byte-exact, guarded; the swapped strides differ", tr.name, len(shapes))
	}
}
