package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestRopeTableOnDeviceMatchesHost gates the rotary table on the device against
// the host kernel it replaces and against a float64 oracle, on every backend
// present.
//
// It lives in backend/ so it runs on every lowerer: spirv lowers ir.OpFma as an
// unfused multiply and add where ptx and msl emit a real fma, so results are
// reported per backend.
//
// The host table is the bar, since the device replaces its upload; the oracle
// is reported beside it to tell "agree" from "wrong together".
func TestRopeTableOnDeviceMatchesHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) { ropeTabCase(t, d) })
	}
}

// ropeTabGeoms are real geometries plus the two edges: the narrowest rotary
// (stories260K: head dimension eight, four pairs, less than one host vector) and
// a pair count that is not a multiple of anything.
var ropeTabGeoms = []struct {
	name  string
	rope  nn.Rope
	rows  int
	quirk string
}{
	{"llama-3.2-1b", nn.Rope{NRot: 64, Base: 500000, Neox: true}, 1, ""},
	{"tinyllama", nn.Rope{NRot: 64, Base: 10000}, 1, ""},
	{"qwen3", nn.Rope{NRot: 128, Base: 1000000, Neox: true}, 1, ""},
	{"gemma3-local", nn.Rope{NRot: 256, Base: 10000, Neox: true}, 1, ""},
	{"prefill-rows", nn.Rope{NRot: 128, Base: 1000000, Neox: true}, 32, "one table per position"},
	{"stories260k", nn.Rope{NRot: 8, Base: 10000}, 1, "four pairs, narrower than a host vector"},
	{"odd-pairs", nn.Rope{NRot: 14, Base: 10000}, 3, "seven pairs"},
	{"mscale", nn.Rope{NRot: 64, Base: 10000, Scale: 0.8}, 1, "attn_factor != 1"},
}

// ropeTabPositions spans the whole decomposition: zero, the digit boundaries,
// common context lengths, and the last position the four digits cover. A kernel that folds three digits instead of four is exact below 2^21
// and catastrophic above it.
func ropeTabPositions() []int {
	p := []int{0, 1, 2, 3, 5, 127, 128, 129, 1000, 4095, 16383, 16384,
		65535, 131071, 1 << 21, 1<<21 + 1, 1 << 24, kernels.RopeTabMaxPos - 1}
	return p
}

func ropeTabCase(t *testing.T, d backend.Device) {
	for _, g := range ropeTabGeoms {
		t.Run(g.name, func(t *testing.T) {
			npairs := g.rope.NRot / 2
			k, err := mainkernels.RopeTable(npairs, g.rows)
			if err != nil {
				t.Fatalf("RopeTable(%d,%d): %v", npairs, g.rows, err)
			}
			c, err := d.Compile(k)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer c.Close()

			gg := newGPU(t, d)
			defer gg.free()
			planes := g.rope.TabPlanes(npairs)
			if len(planes) != kernels.RopeTabBlock(npairs) {
				t.Fatalf("planes are %d words, the layout says %d",
					len(planes), kernels.RopeTabBlock(npairs))
			}
			bPl := gg.up(f32bytes(planes))
			bCo := gg.up(f32bytes(kernels.RopeTabConsts()))
			bPos := gg.up(make([]byte, (g.rows+1)*4))
			bOut := gg.up(make([]byte, g.rows*g.rope.NRot*4))

			var worstHost, worstOracle float64
			var atHost int
			for _, p0 := range ropeTabPositions() {
				if p0+g.rows > kernels.RopeTabMaxPos {
					continue
				}
				got := ropeTabRun(t, c, bPl, bCo, bPos, bOut, p0, g.rows, g.rope.NRot)
				for r := 0; r < g.rows; r++ {
					host := make([]float32, g.rope.NRot)
					g.rope.Table(host, p0+r)
					ref := ropeTabOracle(g.rope, p0+r, npairs)
					for i := range host {
						dh := math.Abs(float64(got[r*g.rope.NRot+i]) - float64(host[i]))
						if dh > worstHost {
							worstHost, atHost = dh, p0+r
						}
						if do := math.Abs(float64(got[r*g.rope.NRot+i]) - ref[i]); do > worstOracle {
							worstOracle = do
						}
					}
				}
			}
			// The bound is the host kernel's own (nn's violation gate uses
			// 3e-7, above the host's error on every tier), so a backend outside
			// it is structurally different rather than rounding differently.
			const bound = 3e-7
			if worstHost > bound {
				t.Fatalf("device vs host table: worst |d| %.3e at position %d, over the %.0e bound",
					worstHost, atHost, bound)
			}
			if worstOracle > bound {
				t.Fatalf("device vs float64 oracle: worst |d| %.3e, over the %.0e bound", worstOracle, bound)
			}
			t.Logf("%-14s npairs=%-3d rows=%-3d  vs host %.3e  vs oracle %.3e  %s",
				g.name, npairs, g.rows, worstHost, worstOracle, g.quirk)
		})
	}
}

// ropeTabRun writes the positions, poisons the output and launches. Zero is a
// plausible sine (position 0's whole sine plane is zero), so an unwritten lane
// must read as NaN instead (RULE 13).
func ropeTabRun(t *testing.T, c backend.Kernel, bPl, bCo, bPos, bOut backend.Buf,
	p0, rows, nrot int) []float32 {
	t.Helper()
	// Element 0 is the valid row count and the positions follow; see
	// mainkernels.RopeTable on why the row count is runtime.
	pos := make([]byte, (rows+1)*4)
	binary.LittleEndian.PutUint32(pos, uint32(rows))
	for r := 0; r < rows; r++ {
		binary.LittleEndian.PutUint32(pos[(r+1)*4:], uint32(p0+r))
	}
	if err := bPos.Write(pos); err != nil {
		t.Fatal(err)
	}
	poison := make([]byte, rows*nrot*4)
	for i := 0; i < rows*nrot; i++ {
		binary.LittleEndian.PutUint32(poison[i*4:], 0x7FC00000) // quiet NaN
	}
	if err := bOut.Write(poison); err != nil {
		t.Fatal(err)
	}
	n := rows * nrot / 2
	if err := c.Launch((n+127)/128, 128, bOut, bPl, bCo, bPos); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, rows*nrot*4)
	if err := bOut.Read(raw); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, rows*nrot)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(out[i])) {
			t.Fatalf("element %d of the table at position %d was never written "+
				"-- the poison survived", i, p0)
		}
	}
	return out
}

// ropeTabOracle is the table in float64, formed the way the float64 table the
// kernel replaced formed it: the same frequency recurrence rather than a fresh
// Pow per pair, so a difference here is about the kernel and not about two
// different sets of frequencies.
func ropeTabOracle(r nn.Rope, pos, npairs int) []float64 {
	out := make([]float64, 2*npairs)
	step := math.Pow(r.Base, -2/float64(r.NRot))
	ms := r.Scale
	if ms == 0 {
		ms = 1
	}
	f := 1.0
	for p := 0; p < npairs; p++ {
		g := f
		if p < len(r.Freqs) {
			g /= float64(r.Freqs[p])
		}
		th := float64(pos) * g
		out[2*p] = float64(float32(math.Cos(th) * ms))
		out[2*p+1] = float64(float32(math.Sin(th) * ms))
		f *= step
	}
	return out
}

// TestRopeTableDeviceGateCatchesItsViolations is engine/nn/ropetabviolation_test.go
// at the device's seam.
//
// The violations are injected into the uploaded constant block, removing one
// step each, as nn's gate does through its copy of RopeTabConsts.
//
// On hardware it also checks what the host gate cannot: the integer one and the
// sign bit read as f32 would be denormals a device may flush to zero, which is
// why the kernel declares pConst as a u32 buffer and bitcasts.
func TestRopeTableDeviceGateCatchesItsViolations(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) { ropeTabViolations(t, d) })
	}
}

func ropeTabViolations(t *testing.T, d backend.Device) {
	const bound = 3e-7
	r := nn.Rope{NRot: 128, Base: 1000000}
	npairs := r.NRot / 2
	k, err := mainkernels.RopeTable(npairs, 1)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(k)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer c.Close()

	gg := newGPU(t, d)
	defer gg.free()
	bPl := gg.up(f32bytes(r.TabPlanes(npairs)))
	bCo := gg.up(f32bytes(kernels.RopeTabConsts()))
	bPos := gg.up(make([]byte, 8))
	bOut := gg.up(make([]byte, r.NRot*4))

	worst := func() (float64, int) {
		var w float64
		var at int
		for _, pos := range ropeTabPositions() {
			got := ropeTabRun(t, c, bPl, bCo, bPos, bOut, pos, 1, r.NRot)
			ref := ropeTabOracle(r, pos, npairs)
			for i := range ref {
				if dd := math.Abs(float64(got[i]) - ref[i]); dd > w {
					w, at = dd, pos
				}
			}
		}
		return w, at
	}

	// The honest arm first, so a violation that reads large because the whole
	// kernel is broken cannot be mistaken for a violation that fired.
	if w, at := worst(); w > bound {
		t.Fatalf("the unviolated device kernel is already %.3e out at position %d", w, at)
	}

	for _, v := range []struct {
		name  string
		what  string
		word  int
		n     int
		extra int
	}{
		{"quadrant-selection", "cosine's quadrant is taken from n instead of n+1",
			kernels.RopeTabOneVecWord, 8, 0},
		{"sign-step", "bit 1 of the quadrant stops deciding the sign, so two of four quadrants are wrong",
			kernels.RopeTabSignVecWord, 8, kernels.RopeTabTwoVecWord},
		// The last two are the device's own (no host tier reads those words;
		// each stands in for an instruction the IR lacks). RopeTabMagicWord is
		// absent because the device does not read it, so zeroing it would remove
		// nothing.
		{"no-mod-4-mask", "the digit products are never folded, so the angle is formed large",
			kernels.RopeTabMod4MaskWord, 1, 0},
		{"no-round-to-integer", "n is truncated rather than rounded, so |r| leaves pi/4",
			kernels.RopeTabHalfWord, 1, 0},
	} {
		t.Run(v.name, func(t *testing.T) {
			cs := kernels.RopeTabConsts()
			for i := 0; i < v.n; i++ {
				cs[v.word+i] = 0
				if v.extra != 0 {
					cs[v.extra+i] = 0
				}
			}
			if err := bCo.Write(f32bytes(cs)); err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := bCo.Write(f32bytes(kernels.RopeTabConsts())); err != nil {
					t.Fatal(err)
				}
			}()
			w, at := worst()
			if w <= bound {
				t.Fatalf("%s: the gate did not fire -- worst |d| %.3e is still inside the "+
					"%.0e bound, so this gate certifies nothing about that step", v.what, w, bound)
			}
			t.Logf("%s -> worst |d| %.3e at position %d", v.what, w, at)
		})
	}

	// The interleave is the one step no constant can reach, so its violation
	// is made in the comparison, as nn's is.
	t.Run("swapped-sin-cos", func(t *testing.T) {
		got := ropeTabRun(t, c, bPl, bCo, bPos, bOut, 12345, 1, r.NRot)
		ref := ropeTabOracle(r, 12345, npairs)
		var w float64
		for p := 0; p < npairs; p++ {
			for _, dd := range []float64{
				math.Abs(float64(got[2*p]) - ref[2*p+1]),
				math.Abs(float64(got[2*p+1]) - ref[2*p]),
			} {
				if dd > w {
					w = dd
				}
			}
		}
		if w <= bound {
			t.Fatalf("a transposed {cos, sin} is within %.0e of the table -- the gate "+
				"cannot tell the two apart at this position", bound)
		}
		t.Logf("{cos, sin} stored the other way round -> worst |d| %.3e", w)
	})
}

// TestRopeTableLeavesThePaddedRowsAlone gates the one step that has no host
// twin and no constant to violate: the runtime row count.
//
// A prefill's last chunk is narrower than the batch width, and the host's
// staging zeroes the rest of the table. The tier clamps the kernel to the valid
// count and zeroes the buffer once; the end-to-end gate cannot see the clamp,
// because it compares only rows that are read.
//
// The violation is the clamp removed, which is spelled here as asking for the
// full row count: the tail then carries real angles instead of the poison,
// which is what the padded rows would have been given.
func TestRopeTableLeavesThePaddedRowsAlone(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, nrot = 16, 64
	r := nn.Rope{NRot: nrot, Base: 500000, Neox: true}
	npairs := nrot / 2
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			k, err := mainkernels.RopeTable(npairs, rows)
			if err != nil {
				t.Fatal(err)
			}
			c, err := d.Compile(k)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer c.Close()
			gg := newGPU(t, d)
			defer gg.free()
			bPl := gg.up(f32bytes(r.TabPlanes(npairs)))
			bCo := gg.up(f32bytes(kernels.RopeTabConsts()))
			bPos := gg.up(make([]byte, (rows+1)*4))
			bOut := gg.up(make([]byte, rows*nrot*4))

			// valid says how many rows the launch declares; the rest must keep
			// whatever the buffer held.
			run := func(valid int) []float32 {
				pos := make([]byte, (rows+1)*4)
				binary.LittleEndian.PutUint32(pos, uint32(valid))
				for i := 0; i < rows; i++ {
					binary.LittleEndian.PutUint32(pos[(i+1)*4:], uint32(1000+i))
				}
				if err := bPos.Write(pos); err != nil {
					t.Fatal(err)
				}
				poison := make([]byte, rows*nrot*4)
				for i := 0; i < rows*nrot; i++ {
					binary.LittleEndian.PutUint32(poison[i*4:], 0x7FC00000)
				}
				if err := bOut.Write(poison); err != nil {
					t.Fatal(err)
				}
				n := valid * npairs
				if err := c.Launch((n+127)/128, 128, bOut, bPl, bCo, bPos); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, rows*nrot*4)
				if err := bOut.Read(raw); err != nil {
					t.Fatal(err)
				}
				out := make([]float32, rows*nrot)
				for i := range out {
					out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
				}
				return out
			}

			// 5*32 = 160 rounds up to two groups of 128, so 96 surplus threads
			// would land in padded rows without the clamp.
			const valid = 5
			got := run(valid)
			for i := 0; i < valid*nrot; i++ {
				if math.IsNaN(float64(got[i])) {
					t.Fatalf("valid row element %d was never written", i)
				}
			}
			for i := valid * nrot; i < rows*nrot; i++ {
				if !math.IsNaN(float64(got[i])) {
					t.Fatalf("padded element %d holds %v -- the clamp let a surplus thread "+
						"write past the valid rows, so a ragged chunk's table would differ "+
						"from the host's zeros", i, got[i])
				}
			}
			// The control: the same kernel asked for every row does write the
			// tail, so the assertion above is about the clamp and not about a
			// kernel that writes nothing.
			all := run(rows)
			for i := range all {
				if math.IsNaN(float64(all[i])) {
					t.Fatalf("at valid=%d element %d was still not written -- this gate "+
						"cannot tell a clamp from a broken kernel", rows, i)
				}
			}
			for i := 0; i < valid*nrot; i++ {
				if all[i] != got[i] {
					t.Fatalf("row %d element %d moved when the row count changed: %v -> %v",
						i/nrot, i%nrot, got[i], all[i])
				}
			}
			t.Logf("%d of %d rows written, %d padded elements untouched, the valid rows identical",
				valid, rows, (rows-valid)*nrot)
		})
	}
}

// TestRopeTableWritesWhatItReads is RULE 13's first invariant, asked of the IR
// rather than of the hardware: no kernel may read a buffer it writes, because
// surplus threads are clamped rather than branched off.
func TestRopeTableWritesWhatItReads(t *testing.T) {
	k, err := mainkernels.RopeTable(64, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := k.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	var loaded, stored [8]bool
	for _, o := range k.Ops {
		if o.Kind != ir.OpLoad && o.Kind != ir.OpStore {
			continue
		}
		pd := k.Ops[o.Args[0]-1]
		if pd.Kind != ir.OpParam {
			continue
		}
		if o.Kind == ir.OpLoad {
			loaded[pd.Imm] = true
		} else {
			stored[pd.Imm] = true
		}
	}
	for i := range loaded {
		if loaded[i] && stored[i] {
			t.Fatalf("param %d is both loaded and stored", i)
		}
	}
	if !stored[0] {
		t.Fatal("pOut is never stored -- this check would pass on an empty kernel")
	}
}
