package tier

import (
	"encoding/binary"
	"fmt"
	"testing"
)

// TestKVHistoryCrossesAGrowthBitForBit drives copyKVAcrossStride on every real
// device -- transposed and row-major K, f32 and f16 V, a growth and a shrink --
// and holds the live prefix of the new buffers to the host reshape it replaced,
// word for word. The positions past the prefix are not compared: nothing reads
// them (attention is bounded by the position count).
func TestKVHistoryCrossesAGrowthBitForBit(t *testing.T) {
	devs := realDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const kvDim = 96
	for _, d := range devs {
		g, err := New([]Slot{{Dev: d, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		dt := g.devs[0]
		for _, c := range [][2]int{{256, 512}, {512, 1024}, {1024, 256}} {
			for _, trans := range []bool{true, false} {
				for _, f16 := range []bool{false, true} {
					name := fmt.Sprintf("%s/%d->%d/trans%v/f16%v", d.API(), c[0], c[1], trans, f16)
					dt.KVF16 = f16
					kvRestrideCase(t, dt, name, kvDim, c[0], c[1], trans, f16)
				}
			}
		}
		g.Close()
	}
}

func kvRestrideCase(t *testing.T, g *devTier, name string, kvDim, oldCap, newCap int, trans, f16 bool) {
	vw := 4
	if f16 {
		vw = 2
	}
	word := func(buf, i int) uint32 { return uint32(buf*1_000_003+i) * 2654435761 }
	fill := func(buf, n int) []byte {
		b := make([]byte, n)
		for i := 0; i < n/4; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], word(buf, i))
		}
		return b
	}
	oldK, oldV := fill(0, (oldCap+1)*kvDim*4), fill(1, (oldCap+1)*kvDim*vw)
	kc, err := g.dev.Alloc(len(oldK))
	if err != nil {
		t.Fatal(err)
	}
	vc, err := g.dev.Alloc(len(oldV))
	if err != nil {
		t.Fatal(err)
	}
	kc.Write(oldK)
	vc.Write(oldV)
	defer kc.Free()
	defer vc.Free()
	nk, err := g.dev.Alloc((newCap + 1) * kvDim * 4)
	if err != nil {
		t.Fatal(err)
	}
	nv, err := g.dev.Alloc((newCap + 1) * kvDim * vw)
	if err != nil {
		t.Fatal(err)
	}
	defer nk.Free()
	defer nv.Free()
	g.mu.Lock()
	ok := g.copyKVAcrossStride(&kvPair{kc: kc, vc: vc}, nk, nv, kvDim, oldCap, newCap, trans)
	g.mu.Unlock()
	if !ok {
		t.Fatalf("%s: copy refused: %s", name, g.LastErr)
	}
	keep := min(oldCap, newCap) + 1
	gotK := make([]byte, (newCap+1)*kvDim*4)
	gotV := make([]byte, (newCap+1)*kvDim*vw)
	if nk.Read(gotK) != nil || nv.Read(gotV) != nil {
		t.Fatalf("%s: read back failed", name)
	}
	bad := 0
	check := func(what string, got []byte, at, want int) {
		if v := binary.LittleEndian.Uint32(got[4*at:]); v != binary.LittleEndian.Uint32(oldK[4*want:]) && what == "K" ||
			what == "V" && v != binary.LittleEndian.Uint32(oldV[4*want:]) {
			if bad++; bad <= 3 {
				t.Errorf("%s: %s word %d = %#x, want old word %d", name, what, at, v, want)
			}
		}
	}
	for e := 0; e < kvDim; e++ {
		for p := 0; p < keep; p++ {
			if trans {
				check("K", gotK, e*(newCap+1)+p, e*(oldCap+1)+p)
			} else {
				check("K", gotK, p*kvDim+e, p*kvDim+e)
			}
		}
	}
	for i := 0; i < keep*kvDim*vw/4; i++ {
		check("V", gotV, i, i)
	}
	if bad > 0 {
		t.Fatalf("%s: %d words of the live history wrong", name, bad)
	}
}
