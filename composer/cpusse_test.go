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

// TestComposedCPUGatesOnSSE runs every composer CPU gate with the host taken
// as the SSE tier (SSE4.1 and SSSE3, no AVX): each forms by the narrow
// profile and runs in the legacy encodings, held to the same oracles, and
// every kernel it mapped is read back through objdump for a VEX leak. On an
// SSE-only host the force changes nothing and the kernels run natively.
func TestComposedCPUGatesOnSSE(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE || cpu.CPU().ISA()&cpu.ISAAVX2 != 0 {
		t.Fatalf("forcing the SSE tier left the host at %v, ISA %v", cpu.HostTier(), cpu.CPU().ISA())
	}
	gates := []struct {
		name string
		run  func(*testing.T)
	}{
		{"MatVec", TestComposedCPUMatVec},
		{"HonoursTheDevicesISA", TestComposedCPUHonoursTheDevicesISA},
		{"RowsMatVec", TestComposedCPURowsMatVec},
		{"Equal", TestComposedCPUEqual},
		{"Halves", TestComposedCPUHalves},
		{"Q8Rows", TestComposedCPUQ8Rows},
		{"SelectBroadcastsAGeneralMask", TestCPUSelectBroadcastsAGeneralMask},
		{"Attend", TestComposedCPUAttend},
		{"AttendRows", TestComposedCPUAttendRows},
		{"CrossAttend", TestComposedCPUCrossAttend},
		{"Clamp", TestComposedCPUClamp},
		{"Rope", TestComposedCPURope},
		{"ExpertPages", TestComposedCPUExpertPages},
		{"Rank", TestComposedCPURank},
		{"TopK", TestComposedCPUTopK},
		{"SampleTopK", TestComposedCPUSampleTopK},
		{"SampleGreedy", TestComposedCPUSampleGreedy},
		{"DeltaRule", TestComposedCPUDeltaRule},
		{"StateConv", TestComposedCPUStateConv},
		{"LayerNorm", TestComposedCPULayerNorm},
		{"Head", TestComposedCPUHead},
		{"HeadStoresItsOutputs", TestCPUHeadStoresItsOutputs},
		{"AZeroInEveryLoop", TestAZeroIsMadeInEveryLoopThatReadsIt},
		{"JamStoresAtAFlatIndex", TestCPUJamStoresAtAFlatIndex},
		{"JamLooksUpByAWeightWord", TestCPUJamLooksUpByAWeightWord},
	}
	for _, g := range gates {
		t.Run(g.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv(sseKernelDir, dir)
			g.run(t)
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
