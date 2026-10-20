package oldkernels_test

import (
	"os"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The packed arena: the host-resident home a page-in reads from.
//
// The gates here check properties no output shows: an arena that repacks on
// every page-in, or whose regions cannot be imported, produces the right bytes
// and is merely slow. So the assertions are counters, addresses and byte
// equality against the allocating packer.

// realTensor finds one quantized tensor of each format on this host, so the
// arena is exercised against the bit layouts and subnormal scales a model
// actually contains rather than against synthetic blocks (the lesson
// TestPackAgainstRealWeights records).
func realTensor(t *testing.T, want kernels.Quant) (src []byte, nrows, k int, f *meta.File) {
	t.Helper()
	models := testmodels.Glob("*.gguf")
	gt := map[kernels.Quant]quant.Type{
		kernels.Q4_0: quant.Q4_0, kernels.Q8_0: quant.Q8_0, kernels.Q4_K: quant.Q4_K,
		kernels.Q3_K: quant.Q3_K, kernels.Q5_K: quant.Q5_K, kernels.Q6_K: quant.Q6_K,
	}[want]
	for _, path := range models {
		fi, err := os.Stat(path)
		if err != nil || fi.Size() > 2<<30 {
			continue // RULE 3: a 24G cap is a deliberate act, not a default
		}
		g, err := gguf.Open(path)
		if err != nil {
			continue
		}
		for i := range g.Tensors {
			ti := &g.Tensors[i]
			if ti.Type != gt || len(ti.Dims) != 2 {
				continue
			}
			k, rows := int(ti.Dims[0]), int(ti.Dims[1])
			if k%want.Elems() != 0 || rows < 64 {
				continue
			}
			return g.Bytes(ti), rows, k, g
		}
		g.Close()
	}
	return nil, 0, 0, nil
}

// TestArenaMatchesPackWeights is the gate on the refactor underneath the arena:
// PackWeights allocates and delegates, so the two forms must be the same
// bytes for every format or every kernel in the tier is reading a different
// weight than it was.
func TestArenaMatchesPackWeights(t *testing.T) {
	seen := 0
	for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K,
		kernels.Q3_K, kernels.Q5_K, kernels.Q6_K} {
		src, nrows, k, f := realTensor(t, q)
		if src == nil {
			t.Logf("no %s tensor on this box", q)
			continue
		}
		seen++
		want0, want1, want2, err := kernels.PackWeights(q, src, nrows, k)
		if err != nil {
			f.Close()
			t.Fatalf("%s: PackWeights: %v", q, err)
		}
		a := kernels.NewArena(1 << 30)
		p, err := a.Pack(0, q, src, nrows, k)
		if err != nil || p == nil {
			f.Close()
			t.Fatalf("%s: arena declined a tensor it had budget for: %v", q, err)
		}
		for i, pair := range [][2][]uint32{{p.QS, want0}, {p.D, want1}, {p.SC, want2}} {
			got, want := pair[0], pair[1]
			if len(got) != len(want) {
				t.Fatalf("%s array %d: arena has %d words, PackWeights %d", q, i, len(got), len(want))
			}
			for j := range want {
				if got[j] != want[j] {
					t.Fatalf("%s array %d word %d: arena %#x, PackWeights %#x -- the arena is "+
						"not the same pack, so a paged block computes a different weight",
						q, i, j, got[j], want[j])
				}
			}
		}
		a.Close()
		f.Close()
	}
	if seen == 0 {
		t.Skip("no quantized tensors in any *.gguf in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	t.Logf("%d formats agree with PackWeights to the word", seen)
}

// TestArenaIsPageAligned is the gate on the half of the design that is not about
// speed.
//
// Alignment is what lets a unified device import rather than copy (the
// drivers require minImportedHostPointerAlignment); the bytes and the engine
// are correct either way, so only this test sees it fail.
func TestArenaIsPageAligned(t *testing.T) {
	a := kernels.NewArena(1 << 30)
	defer a.Close()
	n := 0
	for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K,
		kernels.Q3_K, kernels.Q5_K, kernels.Q6_K} {
		// Odd row counts on purpose: the array lengths are then not multiples of
		// anything, so only the arena's own rounding can align them.
		for _, rows := range []int{1, 7, 65, 257} {
			src := make([]byte, rows*(4096/q.Elems())*q.BlockBytes())
			p, err := a.Pack(1, q, src, rows, 4096)
			if err != nil {
				t.Fatalf("%s %dx4096: %v", q, rows, err)
			}
			if p == nil {
				t.Fatalf("%s %dx4096: the arena declined with a 1 GiB budget", q, rows)
			}
			align := uintptr(kernels.HostAlign())
			for i, v := range [][]uint32{p.QS, p.D, p.SC} {
				if len(v) == 0 {
					continue
				}
				if addr := kernels.Addr(v); addr%align != 0 {
					t.Fatalf("%s %dx4096 array %d starts at %#x, which is %d past a %d "+
						"boundary: VK_EXT_external_memory_host cannot import it",
						q, rows, i, addr, addr%align, align)
				}
				// The length too: both drivers require the imported size to be a
				// whole multiple of the alignment as well as the address, and an
				// array can start on a boundary yet end anywhere.
				if pad := kernels.PaddedBytes(len(v)); pad%uint64(align) != 0 {
					t.Fatalf("%s %dx4096 array %d is %d bytes padded, %d past a %d boundary",
						q, rows, i, pad, pad%uint64(align), align)
				}
				n++
			}
		}
	}
	t.Logf("%d packed arrays, every one starting AND ending on a %d-byte boundary",
		n, kernels.HostAlign())
}

// TestArenaPacksOnce is the property the paging design rests on, at the level
// the arena owns it: the same tensor asked for N times packs once and hands back
// the same memory every time.
//
// Address equality is the load-bearing half: equal contents would also
// hold for an arena that repacked into a fresh region each time, and the
// counter plus the identical base address tells them apart.
func TestArenaPacksOnce(t *testing.T) {
	a := kernels.NewArena(1 << 30)
	defer a.Close()
	src := make([]byte, 512*(4096/32)*18)
	first, err := a.Pack(3, kernels.Q4_0, src, 512, 4096)
	if err != nil || first == nil {
		t.Fatalf("first pack: %v", err)
	}
	base := kernels.Addr(first.QS)
	const pageIns = 8
	for i := 0; i < pageIns; i++ {
		p, err := a.Pack(3, kernels.Q4_0, src, 512, 4096)
		if err != nil || p == nil {
			t.Fatalf("page-in %d: %v", i, err)
		}
		if kernels.Addr(p.QS) != base {
			t.Fatalf("page-in %d got a DIFFERENT region (%#x, not %#x): the arena repacked, "+
				"which costs 3.1x the upload it was meant to replace",
				i, kernels.Addr(p.QS), base)
		}
	}
	st := a.Stats()
	if st.Packs != 1 {
		t.Fatalf("%d packs for 1 tensor and %d page-ins, want 1", st.Packs, pageIns)
	}
	if st.Hits != pageIns {
		t.Fatalf("%d hits for %d page-ins, want %d -- the counter has to prove the page-ins "+
			"HAPPENED, or a gate that never paged passes", st.Hits, pageIns, pageIns)
	}
}

// TestArenaRetainsNothingByDefault is RULE 6 as an assertion: a model that fits
// must not gain an arena it never uses.
//
// The structure is still there: the tier calls Pack unconditionally, so
// the path under test is the path that ships, and at a zero budget it
// declines in a map lookup and a comparison, allocating nothing.
func TestArenaRetainsNothingByDefault(t *testing.T) {
	a := kernels.NewArena(0)
	defer a.Close()
	src := make([]byte, 64*(4096/32)*18)
	for i := 0; i < 4; i++ {
		p, err := a.Pack(0, kernels.Q4_0, src, 64, 4096)
		if err != nil {
			t.Fatalf("%v", err)
		}
		if p != nil {
			t.Fatalf("a zero-budget arena retained %d bytes: a model that FITS pays for an "+
				"arena no page-in will ever read", p.Bytes())
		}
	}
	st := a.Stats()
	if st.Bytes != 0 || st.Entries != 0 || st.Packs != 0 {
		t.Fatalf("zero-budget arena holds %d bytes in %d entries after %d packs, want 0/0/0",
			st.Bytes, st.Entries, st.Packs)
	}
	if st.Declines != 4 {
		t.Fatalf("%d declines for 4 offers: the counter has to show the offers reached it",
			st.Declines)
	}
	// And the budget is a real ceiling, not advice: one entry fits, the second
	// does not, and the arena says so rather than growing.
	nq, nd, nsc, err := kernels.PackedWords(kernels.Q4_0, 64, 4096)
	if err != nil {
		t.Fatal(err)
	}
	_ = nsc
	b := kernels.NewArena(uint64(nq+nd)*4 + 2*kernels.HostAlign())
	defer b.Close()
	src2 := make([]byte, len(src))
	if p, _ := b.Pack(0, kernels.Q4_0, src, 64, 4096); p == nil {
		t.Fatalf("the first tensor did not fit a budget sized for it")
	}
	if p, _ := b.Pack(0, kernels.Q4_0, src2, 64, 4096); p != nil {
		t.Fatalf("the second tensor fitted a budget sized for one: the limit is not a limit")
	}
}

func TestArenaReleaseGivesTheMemoryBack(t *testing.T) {
	a := kernels.NewArena(1 << 30)
	defer a.Close()
	src := make([]byte, 256*(4096/32)*18)
	other := make([]byte, len(src))
	p, err := a.Pack(5, kernels.Q4_0, src, 256, 4096)
	if err != nil || p == nil {
		t.Fatalf("pack: %v", err)
	}
	if _, err := a.Pack(6, kernels.Q4_0, other, 256, 4096); err != nil {
		t.Fatalf("pack: %v", err)
	}
	before := a.Stats()
	freed := a.Release(5)
	after := a.Stats()
	if freed == 0 || after.Bytes != before.Bytes-freed {
		t.Fatalf("released block 5: freed %d, bytes %d -> %d", freed, before.Bytes, after.Bytes)
	}
	if after.Entries != 1 {
		t.Fatalf("released ONE block's entries and %d of %d remain", after.Entries, before.Entries)
	}
	// Block 6 is untouched: Release is per block, not per arena.
	if q := a.Get(kernels.Q4_0, other, 256, 4096); q == nil {
		t.Fatalf("releasing block 5 dropped block 6 as well")
	}
	// unsafe is imported for this: the released region must not still be
	// reachable through the handle, or a page-in after a release reads memory
	// that has been unmapped.
	if p.QS != nil || p.D != nil {
		t.Fatalf("a released Packed still points at %p: a page-in from it is a use after "+
			"munmap, which is a segfault rather than a panic", unsafe.Pointer(&p.QS[0]))
	}
}

// TestArenaForgetUnkeysWhatItCovers is the arena's half of placement.md 15v:
// an entry is keyed on the address it was packed from, and a host frame is one
// address for block after block. Forget over a range takes every entry packed
// from inside it out of the keys, so the next Pack of whatever lands there packs
// afresh. One nothing imports is freed then; one a device imports stays mapped
// until the device lets go (Unimport), since its buffers alias the region.
func TestArenaForgetUnkeysWhatItCovers(t *testing.T) {
	a := kernels.NewArena(1 << 30)
	defer a.Close()
	const rows, k = 64, 4096
	n := rows * (k / 32) * 18
	mem := make([]byte, 3*n)
	copied, imported, outside := mem[:n], mem[n:2*n], mem[2*n:]
	pc, _ := a.Pack(-1, kernels.Q4_0, copied, rows, k)
	pi, _ := a.Pack(-1, kernels.Q4_0, imported, rows, k)
	if pc == nil || pi == nil {
		t.Fatal("the arena declined a pack it had room for")
	}
	if _, err := a.Pack(-1, kernels.Q4_0, outside, rows, k); err != nil {
		t.Fatal(err)
	}
	a.Import(pi)
	before := a.Stats()

	lo := uintptr(unsafe.Pointer(&copied[0]))
	a.Forget(lo, lo+uintptr(2*n))
	st := a.Stats()
	if a.Get(kernels.Q4_0, copied, rows, k) != nil || a.Get(kernels.Q4_0, imported, rows, k) != nil {
		t.Fatal("an entry packed from inside the forgotten range is still found by its address")
	}
	if a.Get(kernels.Q4_0, outside, rows, k) == nil {
		t.Fatal("an entry outside the range was forgotten with it")
	}
	if pc.QS != nil || st.Bytes != before.Bytes-pc.Bytes() || st.Loose != 1 {
		t.Fatalf("after Forget: the copied entry is mapped %v, %d of %d bytes retained, %d loose; "+
			"want it freed and the imported one loose", pc.QS != nil, st.Bytes, before.Bytes, st.Loose)
	}
	if pi.QS == nil {
		t.Fatal("an imported entry was unmapped under the device buffers aliasing it")
	}
	a.Unimport(pi)
	if st := a.Stats(); pi.QS != nil || st.Loose != 0 || st.Bytes != before.Bytes-pc.Bytes()-pi.Bytes() {
		t.Fatalf("after the last Unimport: mapped %v, %d loose, %d bytes retained",
			pi.QS != nil, st.Loose, st.Bytes)
	}
}
