//go:build amd64 && jitllmtest

package composer

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/internal/ssegate"
)

// TestComposedCPUGatesOnSSE runs every composer CPU gate on a CPU at the SSE
// tier (SSE4.1 and SSSE3, no AVX; cpu.AtTier): each forms by the narrow
// profile and runs in the legacy encodings, held to the same oracles, and
// every kernel it mapped is read back through objdump for a VEX leak. On an
// SSE-only machine the tier changes nothing and the kernels run natively.
func TestComposedCPUGatesOnSSE(t *testing.T) {
	sse := cpu.New(cpu.AtTier(cpu.TierSSE))
	if sse.Tier() != cpu.TierSSE || sse.ISA()&cpu.ISAAVX2 != 0 {
		t.Fatalf("a CPU at the SSE tier is at %v, ISA %v", sse.Tier(), sse.ISA())
	}
	gates := []struct {
		name string
		run  func(*testing.T, cpuArm)
	}{
		{"MatVec", gateComposedCPUMatVec},
		{"HonoursTheDevicesISA", gateComposedCPUHonoursTheDevicesISA},
		{"RowsMatVec", gateComposedCPURowsMatVec},
		{"Equal", gateComposedCPUEqual},
		{"Halves", gateComposedCPUHalves},
		{"Q8Rows", gateComposedCPUQ8Rows},
		{"BlockRows", gateComposedCPUBlockRows},
		{"SelectBroadcastsAGeneralMask", gateCPUSelectBroadcastsAGeneralMask},
		{"IndexTakesARemainder", gateCPUIndexTakesARemainder},
		{"Attend", gateComposedCPUAttend},
		{"AttendRows", gateComposedCPUAttendRows},
		{"CrossAttend", gateComposedCPUCrossAttend},
		{"Clamp", gateComposedCPUClamp},
		{"Rope", gateComposedCPURope},
		{"ExpertPages", gateComposedCPUExpertPages},
		{"Rank", gateComposedCPURank},
		{"TopK", gateComposedCPUTopK},
		{"SampleTopK", gateComposedCPUSampleTopK},
		{"SampleGreedy", gateComposedCPUSampleGreedy},
		{"DeltaRule", gateComposedCPUDeltaRule},
		{"StateConv", gateComposedCPUStateConv},
		{"LayerNorm", gateComposedCPULayerNorm},
		{"Head", gateComposedCPUHead},
		{"HeadStoresItsOutputs", gateCPUHeadStoresItsOutputs},
		{"AZeroInEveryLoop", gateAZeroIsMadeInEveryLoopThatReadsIt},
		{"JamStoresAtAFlatIndex", gateCPUJamStoresAtAFlatIndex},
		{"JamLooksUpByAWeightWord", gateCPUJamLooksUpByAWeightWord},
	}
	for _, g := range gates {
		t.Run(g.name, func(t *testing.T) {
			dir := t.TempDir()
			g.run(t, cpuArm{tg: sse, kept: dir})
			files, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			n := 0
			for _, f := range files {
				c, err := os.ReadFile(filepath.Join(dir, f.Name()))
				if err != nil {
					t.Fatal(err)
				}
				if need, _ := x86.KernelISA(c); need&x86.ISATierAVX2 != 0 {
					t.Errorf("a kernel declared %v on the SSE tier", need)
				}
				ssegate.Require(t, fmt.Sprintf("%s-%d", g.name, n), c)
				n++
			}
			t.Logf("%d SSE kernels read back", n)
		})
	}
}
