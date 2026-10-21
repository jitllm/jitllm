//go:build amd64 || arm64

package cpu

import (
	"math/rand"
	"runtime"
	"testing"
	"unsafe"
)

// imgTier is one tier a gate in this package executes on this host: the
// host's own, and on amd64 the SSE tier beside AVX2.
type imgTier struct {
	name string
	em   *Emitters
}

func imgTiers() []imgTier {
	tiers := []imgTier{{"native", Native()}}
	if runtime.GOARCH == "amd64" && HostTier() != TierSSE {
		tiers = append(tiers, imgTier{"sse", EmittersFor(TierSSE)})
	}
	return tiers
}

// imgMap maps an emitted kernel, running an SSE one through the VEX-leak gate
// first.
func imgMap(t *testing.T, tr imgTier, name string, b []byte, err error) *Code {
	t.Helper()
	if err != nil {
		t.Fatalf("%s %s: %v", tr.name, name, err)
	}
	if tr.name == "sse" {
		sseKernelCheck(t, name, b)
	}
	c, err := Map(b)
	if err != nil {
		t.Fatalf("%s %s: %v", tr.name, name, err)
	}
	return c
}

// TestResampleMatchesTheReference gates the image resampling pass
// (EmitResample) on every tier this host executes against the fixed-point
// formula written out: ragged widths around every vector boundary, one to
// nine taps, the precisions both processors use, negative taps (bicubic's
// lobes) and sums that land below 0 and above 255, uint8 in and out, with
// guards after the output and the source rows at a stride that is not the
// width. Against a violation -- the reference shifting one bit less -- it
// must fail.
func TestResampleMatchesTheReference(t *testing.T) {
	ref := func(out, in []uint8, stride int, taps []int32, prec uint) {
		for x := range out {
			acc := int32(1) << (prec - 1)
			for j, w := range taps {
				acc += w * int32(in[j*stride+x])
			}
			v := acc >> prec
			out[x] = uint8(min(max(v, 0), 255))
		}
	}
	for _, tr := range imgTiers() {
		for _, violate := range []bool{false, true} {
			bad := 0
			for _, prec := range []uint{7, 13, 22} {
				b, err := tr.em.Resample(int(prec))
				c := imgMap(t, tr, "resample", b, err)
				r := rand.New(rand.NewSource(int64(prec)))
				scr := ResampleConsts(int(prec))
				for n := 1; n <= 3*ElemLanes+3; n++ {
					for ntap := 1; ntap <= 9; ntap += 2 {
						stride := n + 5
						in := make([]uint8, (ntap-1)*stride+n)
						for i := range in {
							in[i] = uint8(r.Intn(256))
						}
						taps := make([]int32, ntap)
						for i := range taps {
							taps[i] = int32(r.Intn(1<<(prec+1))) - 1<<prec/2
						}
						const guard = 0xA5
						got := make([]uint8, n+ElemLanes)
						for i := range got {
							got[i] = guard
						}
						want := make([]uint8, n)
						rp := prec
						if violate {
							rp = prec - 1
						}
						ref(want, in, stride, taps, rp)
						c.Call(&Args{
							Out:    (*float32)(unsafe.Pointer(&got[0])),
							AScale: (*float32)(unsafe.Pointer(&in[0])),
							W:      (*byte)(unsafe.Pointer(&taps[0])),
							Cols:   int64(ntap),
							RowStr: int64(stride),
							Scr:    (*byte)(unsafe.Pointer(&scr[0])),
							K:      int64(n / ElemLanes),
							Rows:   int64(n % ElemLanes),
						})
						for i := n; i < len(got); i++ {
							if got[i] != guard {
								t.Fatalf("%s prec %d n %d taps %d: wrote past the output at %d: % x (want % x)",
									tr.name, prec, n, ntap, i, got, want)
							}
						}
						for i := 0; i < n; i++ {
							if got[i] != want[i] {
								bad++
								if !violate {
									t.Fatalf("%s prec %d n %d taps %d: sample %d is %d, want %d",
										tr.name, prec, n, ntap, i, got[i], want[i])
								}
							}
						}
					}
				}
				c.Close()
			}
			if violate && bad == 0 {
				t.Fatalf("%s: a reference one bit off agreed everywhere -- this gate proves nothing", tr.name)
			}
		}
		t.Logf("%s: every width to %d, 1..9 taps, precisions 7/13/22 exact; the violation differs", tr.name, 3*ElemLanes+3)
	}
}
