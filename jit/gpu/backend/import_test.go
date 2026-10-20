package backend_test

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// The zero-copy page-in, on real hardware: a device whose heap is the host's
// memory runs its kernels out of the packed arena's own bytes.
//
// The assertion is a mutation, since nothing else tells an import from a copy:
// change the host bytes with no upload, and an imported buffer's next launch
// sees the change while a copy's does not. Both arms run on every importing
// device; the copy arm is the violation and must not see the mutation.

// importCase is one device's answer.
type importCase struct {
	api, name string
	align     int
	unified   bool
}

// importDevices is every device on this host, with every Vulkan device rather
// than the discrete-first pick backend.Open returns -- the point is the
// integrated one, which backend.Open never shows.
func importDevices(t *testing.T) []backend.Device {
	t.Helper()
	var out []backend.Device
	seen := map[string]bool{}
	for _, d := range backend.Open() {
		if d.API() == "spirv" {
			d.Close() // re-opened by index below, with its siblings
			continue
		}
		out = append(out, d)
		seen[d.API()+"/"+d.Name()] = true
	}
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Logf("no Vulkan enumeration: %v", err)
		return out
	}
	for i := range infos {
		if !testdevices.Has(t, "vulkan", i) {
			continue
		}
		d, err := backend.OpenVulkan(strconv.Itoa(i))
		if err != nil {
			t.Logf("vulkan device %d: %v", i, err)
			continue
		}
		out = append(out, d)
	}
	return out
}

func unifiedDev(d backend.Device) bool {
	u, ok := d.(backend.Unified)
	return ok && u.UnifiedMemory()
}

// TestImportedWeightsAreTheHostsOwnBytes is the hardware gate for
// backend.HostImport.
func TestImportedWeightsAreTheHostsOwnBytes(t *testing.T) {
	gpuLock(t)
	devs := importDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU backend")
	}
	var ran, importers []importCase
	for _, d := range devs {
		defer d.Close()
		c := importCase{api: d.API(), name: d.Name(),
			align: backend.ImportAlign(d), unified: unifiedDev(d)}
		ran = append(ran, c)
		if c.align == 0 {
			continue
		}
		importers = append(importers, c)
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) { importCaseRun(t, d) })
	}
	for _, c := range ran {
		t.Logf("%-6s %-46s importAlign %6d  unified %v", c.api, c.name, c.align, c.unified)
	}
	// A run where nothing imported proved nothing and must say so (RULE 10).
	if len(importers) == 0 {
		t.Fatal("no device on this host offers a host-pointer import, so this gate did not " +
			"run at all -- on Vulkan check that VK_EXT_external_memory_host was enabled on " +
			"the logical device, not merely supported by the physical one; on CUDA that " +
			"the card reports CAN_MAP_HOST_MEMORY and HOST_REGISTER_SUPPORTED")
	}
}

func importCaseRun(t *testing.T, d backend.Device) {
	const rows, k = 64, 256
	q := mainkernels.Q8_0
	rng := rand.New(rand.NewSource(20260913))
	w := randWeights(q, rows, k, rng)

	// The arena is the owner, because it is the owner in the engine too: its
	// regions are mmap'd, aligned and padded for exactly this call, and a Go
	// []byte could not be handed to a driver that keeps the pointer.
	a := mainkernels.NewArena(1 << 26)
	defer a.Close()
	p, err := a.Pack(0, q, w, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	if p == nil {
		t.Fatal("the arena declined a 64x256 Q8_0 tensor with a 64 MiB budget")
	}
	if align := backend.ImportAlign(d); uint64(align) > mainkernels.HostAlign() {
		t.Skipf("the device wants %d-byte alignment and the arena lays out on %d",
			align, mainkernels.HostAlign())
	}

	x := randActs(k, rng)
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	ax := append(append([]float32{}, as...), asum...)

	ker, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: 1})
	if err != nil {
		t.Fatal(err)
	}
	kern, err := d.Compile(ker)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	keep := func(b backend.Buf) backend.Buf { bufs = append(bufs, b); return b }
	upload := func(v []uint32) backend.Buf {
		if len(v) == 0 {
			v = []uint32{0}
		}
		b, err := d.Alloc(len(v) * 4)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Write(u32bytes(v)); err != nil {
			t.Fatal(err)
		}
		return keep(b)
	}
	imp := d.(backend.HostImport)
	wrap := func(v []uint32) backend.Buf {
		if len(v) == 0 {
			return upload(v)
		}
		b, err := imp.Import(mainkernels.Ptr(v), int(mainkernels.PaddedBytes(len(v))))
		if err != nil {
			t.Fatalf("Import(%d bytes at %#x): %v", mainkernels.PaddedBytes(len(v)),
				mainkernels.Addr(v), err)
		}
		return keep(b)
	}

	bA, bAX := upload(av), keep(mustAlloc(t, d, f32bytes(ax)))
	run := func(qs, dw, sc backend.Buf) []float32 {
		out := keep(mustAlloc(t, d, make([]byte, rows*4)))
		// The kernel's own workgroup: a Metal kernel dispatched at another
		// width computes the wrong indices (backend's mtlKern.checkWidth).
		width := ker.Group[0]
		if err := kern.Launch((rows+width-1)/width, width, qs, dw, sc, bA, bAX, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, rows*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		return f32s(raw)
	}

	iQS, iD, iSC := wrap(p.QS), wrap(p.D), wrap(p.SC)
	cQS, cD, cSC := upload(p.QS), upload(p.D), upload(p.SC)

	// 1. An import computes the same thing as an upload. Bit equality, because
	//    it is the same kernel over the same bytes and anything else would be a
	//    numerical difference nobody asked for.
	wrapped, copied := run(iQS, iD, iSC), run(cQS, cD, cSC)
	for i := range wrapped {
		if wrapped[i] != copied[i] {
			t.Fatalf("row %d: imported %v, uploaded %v -- an import is supposed to be the "+
				"SAME bytes, so the two must agree to the bit", i, wrapped[i], copied[i])
		}
	}
	if allZero(wrapped) {
		t.Fatal("every output row is zero, so this kernel is not reading the weights at all " +
			"and neither arm would notice the mutation below")
	}

	// 2. The mutation, on the host, with no upload of any kind.
	for i := range p.QS {
		p.QS[i] ^= 0x0f0f0f0f
	}

	after, copyAfter := run(iQS, iD, iSC), run(cQS, cD, cSC)
	moved := 0
	for i := range after {
		if after[i] != wrapped[i] {
			moved++
		}
	}
	if moved == 0 {
		t.Fatalf("the weights changed under the imported buffer and every one of %d output "+
			"rows is identical: this device COPIED, and the machine is holding the model twice",
			len(after))
	}
	// 3. ...and the copy arm, which is the violation, must not have moved.
	for i := range copyAfter {
		if copyAfter[i] != copied[i] {
			t.Fatalf("row %d of the UPLOADED arm changed too (%v -> %v): the harness cannot "+
				"tell an import from a copy, so assertion 2 proves nothing",
				i, copied[i], copyAfter[i])
		}
	}
	t.Logf("%d of %d rows moved when the host bytes changed under the import; the uploaded "+
		"arm did not move", moved, len(after))
}

func mustAlloc(t *testing.T, d backend.Device, p []byte) backend.Buf {
	t.Helper()
	b, err := d.Alloc(len(p))
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Write(p); err != nil {
		t.Fatal(err)
	}
	return b
}

func f32s(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

func allZero(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}
