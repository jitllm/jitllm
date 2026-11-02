//go:build amd64 || arm64

package cpu

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// pixelCase is one decoded picture and how its rows reach PixelRow.
type pixelCase struct {
	name string
	img  image.Image
	f    PixFmt
	sub  int
	// row returns row y's source bytes (the luma's for YCbCr), its chroma
	// rows and its first x.
	row func(y int) (src, cb, cr []byte, x0 int)
}

// pixelCases builds every format PixelRow reads at ragged sizes, random
// content -- an RGBA or RGBA64 whose colour exceeds its alpha included, which
// a decoder never writes and the stdlib path still converts -- and YCbCr at
// every subsampling, whole and as a sub-image at odd offsets.
func pixelCases(r *rand.Rand) []pixelCase {
	var cs []pixelCase
	fill := func(b []byte) {
		for i := range b {
			b[i] = byte(r.Intn(256))
		}
	}
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {5, 2}, {9, 4}, {17, 3}, {33, 2}} {
		w, h := sz[0], sz[1]
		rect := image.Rect(0, 0, w, h)
		rgba := image.NewRGBA(rect)
		fill(rgba.Pix)
		cs = append(cs, pixelCase{fmt.Sprintf("rgba%dx%d", w, h), rgba, PixRGBA, 0, func(y int) ([]byte, []byte, []byte, int) {
			return rgba.Pix[y*rgba.Stride : y*rgba.Stride+4*w], nil, nil, 0
		}})
		nrgba := image.NewNRGBA(rect)
		fill(nrgba.Pix)
		cs = append(cs, pixelCase{fmt.Sprintf("nrgba%dx%d", w, h), nrgba, PixNRGBA, 0, func(y int) ([]byte, []byte, []byte, int) {
			return nrgba.Pix[y*nrgba.Stride : y*nrgba.Stride+4*w], nil, nil, 0
		}})
		gray := image.NewGray(rect)
		fill(gray.Pix)
		cs = append(cs, pixelCase{fmt.Sprintf("gray%dx%d", w, h), gray, PixGray, 0, func(y int) ([]byte, []byte, []byte, int) {
			return gray.Pix[y*gray.Stride : y*gray.Stride+w], nil, nil, 0
		}})
		r64 := image.NewRGBA64(rect)
		fill(r64.Pix)
		cs = append(cs, pixelCase{fmt.Sprintf("rgba64_%dx%d", w, h), r64, PixRGBA64, 0, func(y int) ([]byte, []byte, []byte, int) {
			return r64.Pix[y*r64.Stride : y*r64.Stride+8*w], nil, nil, 0
		}})
		for _, ratio := range []image.YCbCrSubsampleRatio{image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422,
			image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio440, image.YCbCrSubsampleRatio411,
			image.YCbCrSubsampleRatio410} {
			full := image.NewYCbCr(image.Rect(0, 0, w+3, h+3), ratio)
			fill(full.Y)
			fill(full.Cb)
			fill(full.Cr)
			sub, sy := yccShifts(ratio)
			for _, part := range []image.Rectangle{rect, image.Rect(3, 1, w+3, h+1), image.Rect(1, 3, w+1, h+3)} {
				p := full.SubImage(part).(*image.YCbCr)
				cs = append(cs, pixelCase{fmt.Sprintf("ycbcr%v@%v", ratio, part), p, PixYCbCr, sub,
					func(y int) ([]byte, []byte, []byte, int) {
						ay := p.Rect.Min.Y + y
						yo := p.YOffset(p.Rect.Min.X, ay)
						co := (ay>>sy - p.Rect.Min.Y>>sy) * p.CStride
						n := (p.Rect.Max.X-1)>>sub - p.Rect.Min.X>>sub + 1
						return p.Y[yo : yo+w], p.Cb[co : co+n], p.Cr[co : co+n], p.Rect.Min.X
					}})
			}
		}
	}
	return cs
}

// yccShifts is a subsampling ratio's horizontal and vertical chroma shifts.
func yccShifts(r image.YCbCrSubsampleRatio) (sx, sy int) {
	switch r {
	case image.YCbCrSubsampleRatio422:
		return 1, 0
	case image.YCbCrSubsampleRatio420:
		return 1, 1
	case image.YCbCrSubsampleRatio440:
		return 0, 1
	case image.YCbCrSubsampleRatio411:
		return 2, 0
	case image.YCbCrSubsampleRatio410:
		return 2, 1
	}
	return 0, 0
}

// TestPixelRowMatchesTheStdlib gates PixelRow on every tier this host
// executes against what the processors did before it existed: image.Image's
// At(x, y).RGBA() per pixel, composited over white into bytes as toRGB8 did
// (c>>8 for an opaque pixel, (c + 0xffff - a)>>8 otherwise) or kept as the
// 16-bit floats the bilinear path read. Every format and subsampling, ragged
// widths, each source row and chroma row ending at a PROT_NONE page, guards
// after every output row. Reading each pixel's right neighbour must fail it.
func TestPixelRowMatchesTheStdlib(t *testing.T) {
	scr := PixelConsts()
	for _, tr := range imgTiers() {
		r := rand.New(rand.NewSource(11))
		cases := pixelCases(r)
		for _, violate := range []bool{false, true} {
			bad, rows := 0, 0
			for _, pc := range cases {
				b := pc.img.Bounds()
				w := b.Dx()
				for _, o := range []PixOut{PixOver8, PixPlanes} {
					code, err := tr.em.PixelRow(pc.f, pc.sub, o)
					c := imgMap(t, tr, "pixelrow", code, err)
					for y := 0; y < b.Dy(); y++ {
						src, cb, cr, x0 := pc.row(y)
						src = imgGuard(t, src)
						args := &Args{W: &src[0], K: int64(w), Cols: int64(x0), Scr: (*byte)(unsafe.Pointer(&scr[0]))}
						if pc.f == PixYCbCr {
							cb, cr = imgGuard(t, cb), imgGuard(t, cr)
							args.AScale = (*float32)(unsafe.Pointer(&cb[0]))
							args.Q32 = (*float32)(unsafe.Pointer(&cr[0]))
						}
						at := func(x int) (uint32, uint32, uint32, uint32) {
							if violate {
								x = min(x+1, w-1)
							}
							return pc.img.At(b.Min.X+x, b.Min.Y+y).RGBA()
						}
						switch o {
						case PixOver8:
							const guard = 0xA5
							got := make([]byte, 3*w+8)
							for i := range got {
								got[i] = guard
							}
							args.Out = (*float32)(unsafe.Pointer(&got[0]))
							c.Call(args)
							for x := 0; x < w; x++ {
								cr, cg, cb, ca := at(x)
								var want [3]byte
								for i, v := range [3]uint32{cr, cg, cb} {
									if ca == 0xFFFF {
										want[i] = uint8(v >> 8)
									} else {
										want[i] = uint8((v + 0xFFFF - ca) >> 8)
									}
								}
								if [3]byte(got[3*x:3*x+3]) != want {
									bad++
									if !violate {
										t.Fatalf("%s %s row %d pixel %d: % x, want % x", tr.name, pc.name, y, x, got[3*x:3*x+3], want)
									}
								}
							}
							for i := 3 * w; i < len(got); i++ {
								if got[i] != guard {
									t.Fatalf("%s %s row %d: wrote past the row at %d", tr.name, pc.name, y, i)
								}
							}
						case PixPlanes:
							var planes [3][]float32
							for i := range planes {
								planes[i] = make([]float32, w+4)
								for j := range planes[i] {
									planes[i][j] = float32(math.NaN())
								}
							}
							args.Out, args.Out2, args.AScale2 = &planes[0][0], &planes[1][0], &planes[2][0]
							c.Call(args)
							for x := 0; x < w; x++ {
								cr, cg, cb, _ := at(x)
								for i, v := range [3]uint32{cr, cg, cb} {
									if planes[i][x] != float32(v) {
										bad++
										if !violate {
											t.Fatalf("%s %s row %d pixel %d plane %d: %v, want %v",
												tr.name, pc.name, y, x, i, planes[i][x], float32(v))
										}
									}
								}
							}
							for i := range planes {
								for j := w; j < len(planes[i]); j++ {
									if !math.IsNaN(float64(planes[i][j])) {
										t.Fatalf("%s %s row %d: wrote past plane %d at %d", tr.name, pc.name, y, i, j)
									}
								}
							}
						}
						rows++
					}
					c.Close()
				}
			}
			if violate && bad == 0 {
				t.Fatalf("%s: the neighbouring pixel agreed everywhere -- this gate proves nothing", tr.name)
			}
			if !violate {
				t.Logf("%s: %d pictures, %d rows each way, bit-exact against At().RGBA()", tr.name, len(cases), rows)
			}
		}
	}
}

// TestNRGBADivisionIsExact pins the one arithmetic PixelRow does in float:
// c*257*a/255, truncated, through a float32 quotient, for every (c, a).
func TestNRGBADivisionIsExact(t *testing.T) {
	for c := uint32(0); c < 256; c++ {
		for a := uint32(0); a < 256; a++ {
			x := c * 257 * a
			if got := uint32(float32(x) / 255); got != x/255 {
				t.Fatalf("c %d a %d: %d, want %d", c, a, got, x/255)
			}
		}
	}
}
