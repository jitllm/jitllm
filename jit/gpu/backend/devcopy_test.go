package backend_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestDeviceCopyMovesExactlyTheRange: Device.Copy is how a recurrent pool's
// resize carries every session's state into its new buffers and how a step
// across sessions brings a state to the other half of its pool, so a byte
// copied short, long or from the wrong offset is another session's history.
// Every copy here is checked against a host model of BOTH buffers: the range
// holds the source's bytes, and every byte outside it keeps its poison.
//
// The offsets and lengths are ragged -- single words, odd word counts, runs
// that are not a multiple of 16 bytes and a run past a megabyte -- and two
// copies stay inside one buffer, which every driver allows when the ranges do
// not meet. What the contract refuses (a range past either end, a part-word
// offset or length, two ranges of one buffer that overlap, a negative) must
// be refused before the device sees it, with both buffers untouched.
//
// Two selection checks: backend.HostReads does not move across the copies, so
// none of them went through the host; and the source is written by a kernel
// in a Session that is not waited on (Metal commits asynchronously), so a copy
// that ran ahead of the work before it reads the poison instead.
func TestDeviceCopyMovesExactlyTheRange(t *testing.T) {
	gpuLock(t)
	devs := everyDevice(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n = 1<<20 + 4096
	word := func(seed, i int) uint32 { return uint32(seed*1_000_003+i) * 2654435761 }
	fill := func(seed int) []byte {
		b := make([]byte, n)
		for i := 0; i < n/4; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], word(seed, i))
		}
		return b
	}
	for _, d := range devs {
		t.Run(fmt.Sprintf("%s/%s", d.API(), d.Name()), func(t *testing.T) {
			alloc := func(p []byte) backend.Buf {
				t.Helper()
				b, err := d.Alloc(len(p))
				if err != nil {
					t.Fatalf("alloc: %v", err)
				}
				t.Cleanup(b.Free)
				if err := b.Write(p); err != nil {
					t.Fatalf("write: %v", err)
				}
				return b
			}
			want := map[backend.Buf][]byte{}
			// The source arrives by a kernel in an unwaited Session, from a
			// staging buffer: Restride is a straight copy at stride n/4.
			src := alloc(bytes.Repeat([]byte{0xEE}, n))
			want[src] = fill(1)
			stage := alloc(want[src])
			k, err := mainkernels.Restride(1, n/4, n/4, n/4)
			if err != nil {
				t.Fatal(err)
			}
			ck, err := d.Compile(k)
			if err != nil {
				t.Fatal(err)
			}
			defer ck.Close()
			d.Session(func(s backend.Session) {
				err = s.Launch(ck, (n/4+127)/128, 128, stage, src)
			})
			if err != nil {
				t.Fatalf("launch: %v", err)
			}
			dst := alloc(fill(2))
			want[dst] = fill(2)

			check := func(what string, bufs ...backend.Buf) {
				t.Helper()
				for i, b := range bufs {
					got := make([]byte, n)
					if err := b.Read(got); err != nil {
						t.Fatalf("%s: read: %v", what, err)
					}
					if j := firstDiff(got, want[b]); j >= 0 {
						t.Fatalf("%s: buffer %d byte %d is %#x, want %#x", what, i, j, got[j], want[b][j])
					}
				}
			}
			type cp struct {
				dst  backend.Buf
				dOff int
				src  backend.Buf
				sOff int
				n    int
			}
			ok := []cp{
				{dst, 0, src, 0, 4},
				{dst, 4, src, 1028, 12},
				{dst, 4100, src, 8, 20},
				{dst, 65540, src, 131076, 4 * 1023},
				{dst, 12, src, 4096, 1 << 20},
				{dst, n - 4, src, n - 4, 4},
				{dst, 777 * 4, src, 0, 0},
				// One buffer, two ranges that do not meet: adjacent, either
				// way round.
				{dst, 200004, dst, 200004 + 4*301, 4 * 301},
				{src, 4*9000 + 4*77, src, 4 * 9000, 4 * 77},
			}
			r0, _ := backend.HostReads()
			for _, c := range ok {
				if err := d.Copy(c.dst, c.dOff, c.src, c.sOff, c.n); err != nil {
					t.Fatalf("Copy(%d <- %d, %d bytes): %v", c.dOff, c.sOff, c.n, err)
				}
				copy(want[c.dst][c.dOff:c.dOff+c.n], want[c.src][c.sOff:c.sOff+c.n])
			}
			if r1, _ := backend.HostReads(); r1 != r0 {
				t.Fatalf("%d host reads during %d device copies: a copy went through the host", r1-r0, len(ok))
			}
			check("after the copies", src, dst)

			refused := []cp{
				{dst, 0, src, n - 4, 8},        // past the source
				{dst, n - 4, src, 0, 8},        // past the destination
				{dst, 2, src, 0, 8},            // part-word destination offset
				{dst, 0, src, 6, 8},            // part-word source offset
				{dst, 0, src, 0, 10},           // part-word length
				{dst, 0, src, 0, -4},           // negative
				{dst, 4096, dst, 4096 + 4, 64}, // one buffer, overlapping forward
				{src, 4096 + 4, src, 4096, 64}, // and backward
				{dst, 4096, dst, 4096, 4},      // and onto itself
			}
			for _, c := range refused {
				if err := d.Copy(c.dst, c.dOff, c.src, c.sOff, c.n); err == nil {
					t.Fatalf("Copy(%d <- %d, %d bytes) was taken: it is outside the contract", c.dOff, c.sOff, c.n)
				}
			}
			check("after the refusals", src, dst)
		})
	}
}

func firstDiff(a, b []byte) int {
	for i := range a {
		if a[i] != b[i] {
			return i
		}
	}
	return -1
}

// everyDevice is backend.Open and every Vulkan device it did not pick -- an
// integrated GPU beside a discrete one is a different driver and a different
// memory (host-visible, so its transfers are not staged) -- each closed at
// cleanup.
func everyDevice(t *testing.T) []backend.Device {
	devs := backend.Open()
	opened := map[int]bool{}
	for _, d := range devs {
		if o, ok := d.(backend.OrdinalDevice); ok && d.API() == "spirv" {
			opened[o.Ordinal()] = true
		}
	}
	infos, err := backend.VulkanDevices()
	if err == nil {
		for _, in := range infos {
			if opened[in.Index] || !testdevices.Has(t, "vulkan", in.Index) {
				continue
			}
			d, err := backend.OpenVulkan(strconv.Itoa(in.Index))
			if err != nil {
				t.Logf("vulkan:%d would not open: %v", in.Index, err)
				continue
			}
			devs = append(devs, d)
		}
	}
	for _, d := range devs {
		t.Cleanup(d.Close)
	}
	return devs
}
