package model

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend/hip"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/amdgpu"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
)

// amdTargets are the generations the offline gate generates for: CDNA2
// (MI250, wave64, sdot4), CDNA3 (MI300X), RDNA2 (V620 and RX 6000, wave32,
// sdot4), RDNA3 (RX 7900, wave32, sudot4) and RDNA4 (RX 9070), with the target
// features those cards report.
var amdTargets = []string{"gfx90a:sramecc+:xnack-", "gfx942:sramecc+:xnack-", "gfx1030", "gfx1100", "gfx1201"}

// TestComposedFormsLowerForAMDGPU is the AMD lowering's offline gate: every
// block of each served synthetic fixture, formed by the composer at decode
// and at a prompt chunk's rows, with the IR's subgroup (ir.SubgroupLanes,
// which a wave of 64 runs as two) and with none, becomes a code object for
// each AMD generation through comgr, and the code object's own metadata says
// what the lowering promised -- no scratch and no spill (the lowering never
// asks for private memory, so any is the code generator out of registers),
// the wave the target runs, exactly the workgroup memory the kernel declares,
// and each ISA's register and LDS limits.
func TestComposedFormsLowerForAMDGPU(t *testing.T) {
	c, err := hip.LoadComgr(hip.Config{Path: os.Getenv("JITLLM_ROCM")})
	if err != nil {
		t.Skipf("NO COMGR: %v -- the AMDGPU code object gate did not run; set JITLLM_ROCM to a "+
			"directory holding libamd_comgr.so.3 (docs/design/amd-hip.md)", err)
	}
	var files []string
	for _, f := range slices.Concat(servedModels, ssmServed, stepMixtures()) {
		if strings.HasPrefix(f, "synth-") && !slices.Contains(files, f) {
			files = append(files, f)
		}
	}
	type kern struct {
		k    *ir.Kernel
		from string
	}
	targets := make([]amdgpu.Target, len(amdTargets))
	for i, a := range amdTargets {
		if targets[i], err = amdgpu.TargetFor(a); err != nil {
			t.Fatal(err)
		}
	}
	var kernels []kern
	for _, file := range files {
		m := frontModel(t, file)
		f := m.Composing()
		for li := range m.Cfg.NLayer {
			g := f.BlockGraph(li)
			if g == nil {
				m.Close()
				t.Fatalf("%s block %d has no graph: %s", file, li, m.Uncovered(li))
			}
			formats := map[jlm.Role]jlm.Type{}
			for _, o := range g.Ops {
				for _, w := range o.Weight {
					formats[w.Role] = jlm.TypeF32
					if e, ok := f.Find(w.Role, int32(li), -1); ok && w.Rows > 0 && w.K > 4 {
						formats[w.Role] = e.Type
					}
				}
			}
			gr := *g
			gr.Block = 0
			for _, lanes := range []int{ir.SubgroupLanes, 1} {
				for _, rows := range []int{1, 64} {
					_, codes, err := composer.Default().Form(composer.Request{Graph: &gr, Formats: formats,
						Step:   composer.Step{Rows: rows, Page: 64, Keys: 2048, ActWindow: 32},
						Device: composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: lanes, Shared: 64 << 10}})
					if err != nil {
						m.Close()
						t.Fatalf("%s block %d at %d rows, %d lanes: %v", file, li, rows, lanes, err)
					}
					for _, cd := range codes {
						if cd.GPU != nil {
							kernels = append(kernels, kern{cd.GPU, file})
						}
					}
				}
			}
		}
		m.Close()
	}
	for _, tg := range targets {
		t.Run(tg.Base(), func(t *testing.T) {
			seen := map[string]bool{}
			maxV, maxS, compiled := 0, 0, 0
			for _, x := range kernels {
				k := x.k
				src, err := amdgpu.Lower(k, tg)
				if err != nil {
					t.Errorf("%s %s: lower: %v", x.from, k.Name, err)
					continue
				}
				if seen[src] {
					continue
				}
				seen[src] = true
				co, err := c.Compile(src, tg.Arch)
				if err != nil {
					t.Errorf("%s %s: comgr: %v", x.from, k.Name, err)
					continue
				}
				if len(co.Kernels) != 1 || co.Kernels[0].Name != k.Name {
					t.Errorf("%s %s: the code object holds %+v, want the one kernel", x.from, k.Name, co.Kernels)
					continue
				}
				m := co.Kernels[0]
				if m.Scratch != 0 || m.VGPRSpills != 0 {
					t.Errorf("%s %s: %d bytes of scratch, %d VGPR spills", x.from, k.Name, m.Scratch, m.VGPRSpills)
				}
				if m.SGPRSpills != 0 {
					// With no scratch, a spilled SGPR went to a VGPR lane
					// (v_writelane), not to memory: a register move, reported.
					t.Logf("%s %s: %d SGPRs spilled to VGPR lanes, no scratch", x.from, k.Name, m.SGPRSpills)
				}
				if m.Wave != tg.Wave {
					t.Errorf("%s %s: generated at wave %d, the target runs %d", x.from, k.Name, m.Wave, tg.Wave)
				}
				if want := sharedBytes(k); m.LDS != want {
					t.Errorf("%s %s: %d bytes of LDS, the kernel declares %d", x.from, k.Name, m.LDS, want)
				}
				if g := k.Group[0]; m.MaxGroup != g {
					t.Errorf("%s %s: max workgroup %d, the kernel declares %d", x.from, k.Name, m.MaxGroup, g)
				}
				// The hardware's limits, from each ISA's reference guide: 256
				// VGPRs a wave (512 with the AGPRs on CDNA, which share the
				// file), 108 SGPRs counting VCC as the metadata does, 64 KiB
				// of LDS a workgroup. A code object past them fails to
				// launch, not merely runs slowly.
				vmax := 256
				if m.AGPRs > 0 || tg.Wave == 64 && (tg.Base() == "gfx90a" || tg.Base() == "gfx942") {
					vmax = 512
				}
				if m.VGPRs+max(m.AGPRs, 0) > vmax || m.SGPRs > 108 || m.LDS > 64<<10 {
					t.Errorf("%s %s: %d VGPRs + %d AGPRs, %d SGPRs, %d bytes of LDS: past %s's limits",
						x.from, k.Name, m.VGPRs, m.AGPRs, m.SGPRs, m.LDS, tg.Base())
				}
				maxV, maxS = max(maxV, m.VGPRs), max(maxS, m.SGPRs)
				compiled++
			}
			// A gate that compiled nothing would pass: the count is the claim.
			if compiled < 100 {
				t.Errorf("%d distinct kernels became code objects, from %d fixtures; the gate expects at least 100", compiled, len(files))
			}
			t.Logf("%s (wave%d, dot4 %v): %d distinct kernels compiled from %d fixtures, at most %d VGPRs and %d SGPRs",
				tg.Arch, tg.Wave, tg.Dot4, compiled, len(files), maxV, maxS)
		})
	}
}

// sharedBytes is the workgroup memory a kernel declares.
func sharedBytes(k *ir.Kernel) int {
	n := 0
	for _, o := range k.Ops {
		if o.Kind == ir.OpShared {
			n += 4 * int(o.Imm)
		}
	}
	return n
}
