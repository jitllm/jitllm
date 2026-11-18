package gpu

import (
	"fmt"
	"runtime"
	"testing"
)

// TestKVLayerZeroesOnTheCard holds a KV layer's new buffers to being zeroed
// on the device (zeroBuf): one upload of at most zeroChunk bytes a buffer,
// copies on the device for the rest, and no host allocation of the buffer's
// size. A pool's growth past a prompt's pages uploaded zeros the size of each
// buffer from a fresh host slice, and the first decode token after a long
// prompt spent a second clearing and collecting them. On a card, every
// buffer is poisoned first and must read back zero at every size, the ragged
// tails past a chunk and past a doubling among them.
func TestKVLayerZeroesOnTheCard(t *testing.T) {
	t.Run("fake", func(t *testing.T) {
		d := &fakeDev{}
		g, err := New([]Slot{{Dev: d, Bytes: 4 << 30}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(g.Close)
		dt := g.devs[0]
		dt.mu.Lock()
		defer dt.mu.Unlock()
		const p, n = 64, 256
		geom := kvGeom{kvRow: 1024}
		size := (geom.kWords(p) + geom.vWords(p)) * n * 4
		w0, c0, b0, z0 := d.writeBytes, d.copies, d.copyBytes, dt.KVZeroCopies
		var m0, m1 runtime.MemStats
		runtime.ReadMemStats(&m0)
		k, v, err := dt.allocLayerPool(geom, p, n)
		runtime.ReadMemStats(&m1)
		if err != nil {
			t.Fatal(err)
		}
		defer freeKV2(k, v)
		up, copied := d.writeBytes-w0, d.copyBytes-b0
		switch {
		case up > 2*zeroChunk:
			t.Fatalf("%d bytes uploaded zeroing %d bytes of K and V; want at most a chunk a buffer (%d)", up, size, 2*zeroChunk)
		case up+copied != size:
			t.Fatalf("%d bytes uploaded and %d copied on the device for %d bytes of K and V", up, copied, size)
		case d.copies-c0 != int(dt.KVZeroCopies-z0):
			t.Fatalf("%d copies on the device, KVZeroCopies counts %d", d.copies-c0, dt.KVZeroCopies-z0)
		case m1.TotalAlloc-m0.TotalAlloc > uint64(size)/16:
			t.Fatalf("%d bytes allocated on the host zeroing %d bytes of K and V", m1.TotalAlloc-m0.TotalAlloc, size)
		}
		t.Logf("%d MiB of K and V: %d bytes uploaded, %d copies on the device, %d bytes allocated on the host",
			size>>20, up, d.copies-c0, m1.TotalAlloc-m0.TotalAlloc)
	})
	devs := realDevices(t)
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the card arm proved nothing")
	}
	for _, dv := range devs {
		t.Run(dv.API(), func(t *testing.T) {
			g, err := New([]Slot{{Dev: dv, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
			if err != nil {
				t.Fatal(err)
			}
			defer g.Close()
			dt := g.devs[0]
			dt.mu.Lock()
			defer dt.mu.Unlock()
			for _, n := range []int{4, 28, zeroChunk - 4, zeroChunk, zeroChunk + 4, 3*zeroChunk + 12, 5<<20 + 4} {
				t.Run(fmt.Sprint(n), func(t *testing.T) {
					b, err := dt.dev.Alloc(n)
					if err != nil {
						t.Fatal(err)
					}
					defer b.Free()
					got := make([]byte, n)
					for i := range got {
						got[i] = 0xff
					}
					if err := b.Write(got); err != nil {
						t.Fatal(err)
					}
					if err := dt.zeroBuf(b, n); err != nil {
						t.Fatal(err)
					}
					if err := b.Read(got); err != nil {
						t.Fatal(err)
					}
					for i, c := range got {
						if c != 0 {
							t.Fatalf("byte %d of %d reads %#x after zeroBuf", i, n, c)
						}
					}
				})
			}
		})
	}
}
