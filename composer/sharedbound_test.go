package composer_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// declaredShared is the workgroup memory k declares, in bytes: every
// element of the IR's arrays is a 32-bit word.
func declaredShared(k *ir.Kernel) int {
	b := 0
	for _, o := range k.Ops {
		if o.Kind == ir.OpShared {
			b += 4 * int(o.Imm)
		}
	}
	return b
}

// TestFormsHoldTheDevicesWorkgroupMemory forms attentions at decode and at
// a prompt chunk's rows, with and without a matrix unit, on a device stating
// 32 KiB of workgroup memory (an Apple GPU's maxThreadgroupMemoryLength) and
// on one stating 16 KiB: no kernel may declare more than the device holds,
// and at 32 KiB every case forms; below it a form that cannot fit is
// refused by name.
// The violation, the same forms on a device stating none, must exceed 32
// KiB somewhere, or the gate bounds nothing.
func TestFormsHoldTheDevicesWorkgroupMemory(t *testing.T) {
	cases := []attendRowsCase{
		{8, 2, 64, 0, false, false},
		{16, 8, 128, 0, false, false},
		{32, 8, 64, 0, false, false},
		{64, 8, 64, 16, true, false},
		{64, 8, 64, 0, true, false},
		{16, 8, 256, 0, false, false},
		{32, 4, 128, 0, false, false},
		{16, 8, 128, 0, false, true},
		{16, 8, 256, 0, false, true},
		{8, 1, 512, 0, false, false},
		{8, 2, 512, 0, false, true},
	}
	unbounded, refused := 0, 0
	for _, shared := range []int{0, 32 << 10, 16 << 10} {
		for _, matrix := range []bool{false, true} {
			for _, rows := range []int{1, 64, 512} {
				for _, c := range cases {
					name := fmt.Sprintf("shared%d/matrix%v/rows%d/%dq%dkv%dd/window%d/f32values%v", shared, matrix, rows, c.heads, c.kvh, c.dim, c.window, c.f32v)
					b := graph.New()
					n := b.KVAppend(b.In("k", c.kvh*c.dim), b.In("v", c.kvh*c.dim), 0)
					var sinks []jlm.Role
					if c.sinks {
						sinks = []jlm.Role{jlm.RoleAttnSinks}
					}
					b.Out(b.Attend(b.In("q", c.heads*c.dim), n, graph.Attrs{NHead: c.heads, NKVHead: c.kvh, HeadDim: c.dim,
						Scale: 1 / math.Sqrt(float64(c.dim)), Window: c.window}, sinks...))
					dev := composer.Device{Kind: composer.DeviceGPU, Lanes: 32, Units: 10, Shared: shared}
					if matrix {
						dev.Matrix = [3]int{16, 16, 16}
					}
					_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Step: composer.Step{Rows: rows, Page: 64, Keys: 4096, Seqs: 1},
						Device: dev, History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: !c.f32v}}})
					if err != nil {
						// Below Apple's 32 KiB a form may hold more than the
						// device: refused by name, never launched.
						if shared >= 32<<10 || !strings.Contains(err.Error(), "of workgroup memory, past the device's") {
							t.Errorf("%s: %v", name, err)
						}
						refused++
						continue
					}
					for _, cd := range codes {
						if cd.GPU == nil {
							continue
						}
						s := declaredShared(cd.GPU)
						switch {
						case shared == 0:
							unbounded = max(unbounded, s)
						case s > shared:
							t.Errorf("%s: %s declares %d bytes of workgroup memory, past the device's %d", name, cd.Name, s, shared)
						}
					}
				}
			}
		}
	}
	if unbounded <= 32<<10 {
		t.Errorf("the forms on a device stating no bound declare at most %d bytes: the gate bounds nothing", unbounded)
	}
	t.Logf("the largest workgroup memory unbounded: %d bytes; %d forms refused at 16 KiB", unbounded, refused)
}
