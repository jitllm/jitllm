package composer_test

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestCapacityFormLaunchesFewRowsAsOneRow forms a Llama-3.1-8B-shaped block
// at a joint step's capacity of 512 rows on a device of a V100's CUDA
// statement, and counts the kernels a call of two live rows launches and a
// call of the whole capacity: each norm hosts its rows' int8 form as one
// workgroup a row (Coarsen), as decode's form of one row does, so the few
// rows launch 11 kernels where they launched 18 (each norm's window maxima
// and its codes two launches of their own; q, k and v three and gate and up
// two, which Siblings runs as one each), and the capacity's call launches no
// more than its 24.
func TestCapacityFormLaunchesFewRowsAsOneRow(t *testing.T) {
	d := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32,
		Matrix: [3]int{16, 16, 16}, Fragment: ir.MMAVolta(), HalfPairs: true, VectorLoads: true, Shared: 96 << 10, Buffers: 32}
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp} {
		formats[r] = jlm.TypeQ4S
	}
	formats[jlm.RoleFFNDown] = jlm.TypeQ6S
	formats[jlm.RoleAttnNorm], formats[jlm.RoleFFNNorm] = jlm.TypeF32, jlm.TypeF32
	_, codes, err := composer.Default().Form(composer.Request{Graph: blockOf(4096, 32, 8, 128, 14336), Formats: formats,
		Step: composer.Step{Rows: 512, Page: 64, Keys: 1024, ActWindow: 32, Seqs: 2}, Device: d,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}})
	if err != nil {
		t.Fatal(err)
	}
	launches := func(rows int) (n int, names []string) {
		for _, c := range codes {
			if c.Runs(rows, 0) {
				n++
				names = append(names, c.Name)
			}
		}
		return n, names
	}
	few, fewNames := launches(2)
	all, allNames := launches(512)
	t.Logf("two rows: %d %v", few, fewNames)
	t.Logf("512 rows: %d %v", all, allNames)
	if few != 11 {
		t.Errorf("a call of two rows launches %d kernels, want 11", few)
	}
	if all > 24 {
		t.Errorf("a call of the capacity launches %d kernels, more than 24", all)
	}
	hosted := 0
	for _, c := range codes {
		if c.Name == "reduce+map+quantized" && c.Runs(2, 0) && c.Live == 512 {
			hosted++
		}
	}
	if hosted != 2 {
		t.Errorf("%d norms host their rows' int8 form at two rows, want 2", hosted)
	}
}
