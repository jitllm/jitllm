//go:build amd64 || arm64

package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// axisMode is one reference's position-table axis: its nine coordinate steps
// for an output size and a table side, and its filter.
type axisMode struct {
	name  string
	f     AxisFilter
	chain func(size, side int) [9]float32
	// ref is the reference's own float32 code, every product and sum rounded
	// by an explicit conversion so no host fuses them (arm64 Go would).
	ref func(i, size, side int) (taps [4]int32, w [4]float32)
}

func axisModes() []axisMode {
	cubicAt := func(ix float32, side int) (taps [4]int32, w [4]float32) {
		fl := float32(math.Floor(float64(ix)))
		t := ix - fl
		const A = float32(-0.75)
		conv1 := func(x float32) float32 {
			return float32(float32(float32(float32((A+2)*x)-(A+3))*x)*x) + 1
		}
		conv2 := func(x float32) float32 {
			return float32(float32(float32(float32(float32(A*x)-5*A)*x)+8*A)*x) - 4*A
		}
		w = [4]float32{conv2(t + 1), conv1(t), conv1(1 - t), conv2(2 - t)}
		for k := range taps {
			taps[k] = int32(min(max(int(fl)-1+k, 0), side-1))
		}
		return taps, w
	}
	return []axisMode{
		{"bilinear", AxisBilinear,
			func(size, side int) [9]float32 {
				return [9]float32{0, float32(side - 1), float32(max(size-1, 1)), 1, 0, 0, 1, 0, 1}
			},
			func(i, size, side int) (taps [4]int32, w [4]float32) {
				src := float32(float32(i)*float32(side-1)) / float32(max(size-1, 1))
				fl := float32(math.Floor(float64(src)))
				f := int(fl)
				taps[0], taps[1] = int32(min(max(f, 0), side-1)), int32(min(max(f+1, 0), side-1))
				abs := func(x float32) float32 { return float32(math.Abs(float64(x))) }
				w[0] = max(1-abs(src-fl-0), 0)
				w[1] = max(1-abs(src-fl-1), 0)
				return
			}},
		{"grid_sample", AxisCubic,
			func(size, side int) [9]float32 {
				return [9]float32{0.5, 1, float32(size), 2, -1, 1, float32(side), -1, 2}
			},
			func(i, size, side int) ([4]int32, [4]float32) {
				norm := float32(float32((float32(i)+0.5)/float32(size))*2) - 1
				ix := float32(float32(float32(norm+1)*float32(side))-1) / 2
				return cubicAt(ix, side)
			}},
		{"interpolate", AxisCubic,
			func(size, side int) [9]float32 {
				return [9]float32{0.5, float32(side) / float32(size), 1, 1, -0.5, 0, 1, 0, 1}
			},
			func(i, size, side int) ([4]int32, [4]float32) {
				scale := float32(side) / float32(size)
				return cubicAt(float32(scale*(float32(i)+0.5))-0.5, side)
			}},
		// HunyuanVL's: F.interpolate's bilinear with align_corners False.
		{"interpolate_linear", AxisLinear,
			func(size, side int) [9]float32 {
				return [9]float32{0.5, float32(side) / float32(size), 1, 1, -0.5, 0, 1, 0, 1}
			},
			func(i, size, side int) (taps [4]int32, w [4]float32) {
				scale := float32(side) / float32(size)
				src := max(float32(scale*(float32(i)+0.5))-0.5, 0)
				f := int(src)
				l := src - float32(f)
				taps[0], taps[1] = int32(f), int32(min(f+1, side-1))
				w[0], w[1] = 1-l, l
				return
			}},
	}
}

// TestAxisTapsMatchesTheReference gates AxisTaps on every tier this host
// executes against each reference's float32 code: Qwen3-VL's bilinear,
// GLM-4.xV's grid_sample bicubic, Kimi-VL's interpolate bicubic and
// HunyuanVL's interpolate bilinear, at every
// output size to 20 and a few past it, tables smaller and larger than the
// grid, every weight and index bit-exact, a guard after the last plane. The
// same arithmetic in float64, rounded once at the end -- the port that looks
// right -- must fail it.
func TestAxisTapsMatchesTheReference(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 37, 64}
	sides := []int{1, 2, 3, 7, 16, 48}
	for _, tr := range imgTiers() {
		for _, m := range axisModes() {
			b, err := tr.em.AxisTaps(m.f)
			c := imgMap(t, tr, "axistaps", b, err)
			taps := AxisTapCount(m.f)
			for _, violate := range []bool{false, true} {
				bad := 0
				for _, size := range sizes {
					for _, side := range sides {
						scr := AxisConsts(m.chain(size, side), int32(side-1))
						guard := float32(math.NaN())
						w := make([]float32, taps*size+4)
						ix := make([]int32, taps*size+4)
						for i := range w {
							w[i] = guard
							ix[i] = -7777
						}
						c.Call(&Args{
							Out:  &w[0],
							Out2: (*float32)(unsafe.Pointer(&ix[0])),
							K:    int64(size),
							Scr:  (*byte)(unsafe.Pointer(&scr[0])),
						})
						for i := taps * size; i < len(w); i++ {
							if !math.IsNaN(float64(w[i])) || ix[i] != -7777 {
								t.Fatalf("%s %s size %d side %d: wrote past the planes at %d", tr.name, m.name, size, side, i)
							}
						}
						for i := 0; i < size; i++ {
							wt, ww := m.ref(i, size, side)
							if violate {
								wt, ww = axisF64(m, i, size, side)
							}
							for k := 0; k < taps; k++ {
								gw, gi := w[k*size+i], ix[k*size+i]
								if math.Float32bits(gw) != math.Float32bits(ww[k]) || gi != wt[k] {
									bad++
									if !violate {
										t.Fatalf("%s %s size %d side %d output %d tap %d: (%d, %v), want (%d, %v)",
											tr.name, m.name, size, side, i, k, gi, gw, wt[k], ww[k])
									}
								}
							}
						}
					}
				}
				if violate && bad == 0 {
					t.Fatalf("%s %s: the float64 port agreed everywhere -- this gate proves nothing", tr.name, m.name)
				}
				if violate {
					t.Logf("%s %s: exact at %d sizes x %d sides; the float64 port differs at %d", tr.name, m.name,
						len(sizes), len(sides), bad)
				}
			}
			c.Close()
		}
	}
}

// axisF64 is mode m's arithmetic in float64, rounded to float32 at the end.
func axisF64(m axisMode, i, size, side int) (taps [4]int32, w [4]float32) {
	ch := m.chain(size, side)
	x := float64(i)
	for k, v := range ch {
		switch k {
		case 0, 4, 5, 7:
			x += float64(v)
		case 1, 3, 6:
			x *= float64(v)
		default:
			x /= float64(v)
		}
	}
	if m.f == AxisLinear {
		x = math.Max(x, 0)
	}
	fl := math.Floor(x)
	t := x - fl
	if m.f == AxisLinear {
		w[0], w[1] = float32(1-t), float32(t)
		taps[0], taps[1] = int32(min(max(int(fl), 0), side-1)), int32(min(max(int(fl)+1, 0), side-1))
		return
	}
	if m.f == AxisBilinear {
		w[0], w[1] = float32(1-t), float32(1-math.Abs(t-1))
		taps[0], taps[1] = int32(min(max(int(fl), 0), side-1)), int32(min(max(int(fl)+1, 0), side-1))
		return
	}
	const A = -0.75
	c1 := func(x float64) float64 { return ((A+2)*x-(A+3))*x*x + 1 }
	c2 := func(x float64) float64 { return ((A*x-5*A)*x+8*A)*x - 4*A }
	for k, v := range []float64{c2(t + 1), c1(t), c1(1 - t), c2(2 - t)} {
		w[k] = float32(v)
		taps[k] = int32(min(max(int(fl)-1+k, 0), side-1))
	}
	return
}

// TestLerpGridMatchesTheReference gates LerpGrid: every grid to 9 x 9 and a
// wider one, each weight the float32 product and each index ht*side + wt,
// guards after both planes. Indexing the table column-first must fail it.
func TestLerpGridMatchesTheReference(t *testing.T) {
	for _, tr := range imgTiers() {
		b, err := tr.em.LerpGrid()
		c := imgMap(t, tr, "lerpgrid", b, err)
		r := rand.New(rand.NewSource(7))
		for _, violate := range []bool{false, true} {
			bad := 0
			for _, gh := range []int{1, 2, 3, 5, 9, 17} {
				for _, gw := range []int{1, 2, 3, 4, 5, 8, 9, 23} {
					side := 3 + r.Intn(40)
					hw, ww := make([]float32, gh), make([]float32, gw)
					ht, wt := make([]int32, gh), make([]int32, gw)
					for i := range hw {
						hw[i], ht[i] = r.Float32()*2-0.5, int32(r.Intn(side))
					}
					for i := range ww {
						ww[i], wt[i] = r.Float32()*2-0.5, int32(r.Intn(side))
					}
					w := make([]float32, gh*gw+4)
					ix := make([]int32, gh*gw+4)
					for i := range w {
						w[i] = float32(math.NaN())
						ix[i] = -7777
					}
					c.Call(&Args{
						Out:     &w[0],
						Out2:    (*float32)(unsafe.Pointer(&ix[0])),
						AScale:  &ww[0],
						AScale2: (*float32)(unsafe.Pointer(&wt[0])),
						Q32:     &hw[0],
						Q2:      (*float32)(unsafe.Pointer(&ht[0])),
						K:       int64(gw),
						Rows:    int64(gh),
						Cols:    int64(side),
					})
					for i := gh * gw; i < len(w); i++ {
						if !math.IsNaN(float64(w[i])) || ix[i] != -7777 {
							t.Fatalf("%s %dx%d: wrote past the planes at %d", tr.name, gh, gw, i)
						}
					}
					for y := 0; y < gh; y++ {
						for x := 0; x < gw; x++ {
							want, wantI := hw[y]*ww[x], ht[y]*int32(side)+wt[x]
							if violate {
								wantI = wt[x]*int32(side) + ht[y]
							}
							got, gotI := w[y*gw+x], ix[y*gw+x]
							if math.Float32bits(got) != math.Float32bits(want) || gotI != wantI {
								bad++
								if !violate {
									t.Fatalf("%s %dx%d at (%d, %d): (%v, %d), want (%v, %d)", tr.name, gh, gw, y, x,
										got, gotI, want, wantI)
								}
							}
						}
					}
				}
			}
			if violate && bad == 0 {
				t.Fatalf("%s: the column-first index agreed everywhere -- this gate proves nothing", tr.name)
			}
		}
		c.Close()
	}
}
