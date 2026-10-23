package lowertest

import (
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend/hip"
	"github.com/jitllm/jitllm/jit/amdgpu"
	"github.com/jitllm/jitllm/jit/ir"
)

// amdTargets are the generations the offline gate generates for: CDNA2
// (MI250, wave64, sdot4), CDNA3 (MI300X), RDNA2 (V620 and RX 6000, wave32,
// sdot4), RDNA3 (RX 7900, wave32, sudot4) and RDNA4 (RX 9070), with the target
// features those cards report.
var amdTargets = []string{
	"gfx90a:sramecc+:xnack-",
	"gfx942:sramecc+:xnack-",
	"gfx1030",
	"gfx1100",
	"gfx1201",
}

// comgrOrSkip loads comgr from JITLLM_ROCM, or the default places. Without it
// the gate cannot run, and says so by name rather than passing.
func comgrOrSkip(t *testing.T) *hip.Comgr {
	t.Helper()
	c, err := hip.LoadComgr(hip.Config{Path: os.Getenv("JITLLM_ROCM")})
	if err != nil {
		t.Skipf("NO COMGR: %v -- the AMDGPU code object gate did not run; set JITLLM_ROCM to a "+
			"directory holding libamd_comgr.so.3 (docs/design/amd-hip.md)", err)
	}
	return c
}

// TestAMDGPULowersForEveryTarget is the AMD lowering's offline gate: every
// kernel the other lowerings are held to becomes a code object for each
// target, and the code object's own metadata says what the lowering promised
// -- no scratch and no spill (the lowering never asks for private memory, so
// any is the code generator running out of registers), the wave size the
// target runs, and exactly the shared memory the kernel declared.
func TestAMDGPULowersForEveryTarget(t *testing.T) {
	c := comgrOrSkip(t)
	ks := all(t)
	names := make([]string, 0, len(ks))
	for n := range ks {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, arch := range amdTargets {
		tg, err := amdgpu.TargetFor(arch)
		if err != nil {
			t.Fatal(err)
		}
		t.Run(tg.Base(), func(t *testing.T) {
			maxV, maxS, compiled := 0, 0, 0
			for _, n := range names {
				k := ks[n]
				src, err := amdgpu.Lower(k, tg)
				if err != nil {
					t.Errorf("%s: lower: %v", n, err)
					continue
				}
				co, err := c.Compile(src, tg.Arch)
				if err != nil {
					t.Errorf("%s: comgr: %v", n, err)
					continue
				}
				if len(co.Kernels) != 1 || co.Kernels[0].Name != k.Name {
					t.Errorf("%s: the code object holds %+v, want the one kernel %s", n, co.Kernels, k.Name)
					continue
				}
				m := co.Kernels[0]
				if m.Scratch != 0 || m.VGPRSpills != 0 {
					t.Errorf("%s: %d bytes of scratch, %d VGPR spills", n, m.Scratch, m.VGPRSpills)
				}
				if m.SGPRSpills != 0 {
					// With no scratch, a spilled SGPR went to a VGPR lane
					// (v_writelane), not to memory: a register move, reported.
					t.Logf("%s: %d SGPRs spilled to VGPR lanes, no scratch", n, m.SGPRSpills)
				}
				if m.Wave != tg.Wave {
					t.Errorf("%s: generated at wave %d, the target runs %d", n, m.Wave, tg.Wave)
				}
				if want := sharedBytes(k); m.LDS != want {
					t.Errorf("%s: %d bytes of LDS, the kernel declares %d", n, m.LDS, want)
				}
				if g := k.Group[0]; m.MaxGroup != g {
					t.Errorf("%s: max workgroup %d, the kernel declares %d", n, m.MaxGroup, g)
				}
				// The hardware's limits, from each ISA's reference guide: 256
				// VGPRs a wave (512 with the AGPRs on CDNA, which share the
				// file), 108 SGPRs counting VCC as the metadata does, 64 KiB of LDS a workgroup. The code
				// generator should never exceed them; a code object that did
				// would fail to launch, not merely run slowly.
				vmax := 256
				if m.AGPRs > 0 || tg.Wave == 64 && (tg.Base() == "gfx90a" || tg.Base() == "gfx942") {
					vmax = 512
				}
				if m.VGPRs+max(m.AGPRs, 0) > vmax || m.SGPRs > 108 || m.LDS > 64<<10 {
					t.Errorf("%s: %d VGPRs + %d AGPRs, %d SGPRs, %d bytes of LDS: past %s's limits",
						n, m.VGPRs, m.AGPRs, m.SGPRs, m.LDS, tg.Base())
				}
				maxV, maxS = max(maxV, m.VGPRs), max(maxS, m.SGPRs)
				compiled++
			}
			// A gate that compiled nothing would pass: the count is the claim.
			if compiled != len(names) || compiled < 100 {
				t.Errorf("%d of %d kernels became code objects; the gate expects every one, and at least 100",
					compiled, len(names))
			}
			t.Logf("%s (wave%d, dot4 %s): %d kernels compiled, at most %d VGPRs and %d SGPRs",
				tg.Arch, tg.Wave, tg.Dot4, compiled, maxV, maxS)
		})
	}
}

// sharedBytes is the LDS a kernel declares.
func sharedBytes(k *ir.Kernel) int {
	n := 0
	for _, o := range k.Ops {
		if o.Kind == ir.OpShared {
			n += 4 * int(o.Imm)
		}
	}
	return n
}

// TestAMDGPURefusesWhatItCannotLower: the fragment-shaped MMA kernels are
// PTX's, and the AMD lowering must refuse them by name, not quietly emit
// something.
func TestAMDGPURefusesWhatItCannotLower(t *testing.T) {
	tg, err := amdgpu.TargetFor("gfx1100")
	if err != nil {
		t.Fatal(err)
	}
	mk := mmaKernels(t)
	if len(mk) == 0 {
		t.Fatal("no MMA kernels; this test would pass and prove nothing")
	}
	for n, k := range mk {
		if _, err := amdgpu.Lower(k, tg); err == nil || !strings.Contains(err.Error(), "mma") {
			t.Errorf("%s: lowered, or refused for the wrong reason: %v", n, err)
		}
	}
}
