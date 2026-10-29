package jit_test

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestDecodeSplitFollowsTheDevice forms a Llama-3.1-8B block's decode on a
// V100's statement and on a card of a fifth its resident threads: q, k and
// v, split into different segments, take one workgroup width and run as one
// launch on both; the projections' workgroups follow the device's resident
// threads (Split.Fill), so the smaller card's are narrower.
func TestDecodeSplitFollowsTheDevice(t *testing.T) {
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp} {
		formats[r] = jlm.TypeQ4S
	}
	formats[jlm.RoleFFNDown] = jlm.TypeQ6S
	formats[jlm.RoleAttnNorm], formats[jlm.RoleFFNNorm] = jlm.TypeF32, jlm.TypeF32
	type launch struct{ groups, group int }
	form := func(d jit.Device) []launch {
		t.Helper()
		_, codes, err := jit.Default().Form(jit.Request{Graph: blockOf(4096, 32, 8, 128, 14336), Formats: formats,
			Step: jit.Step{Rows: 1, Page: 64, Keys: 1024, ActWindow: 32, Seqs: 1}, Device: d,
			History: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: true}}})
		if err != nil {
			t.Fatal(err)
		}
		var out []launch
		for _, c := range codes {
			if slices.Contains(c.Passes, "split") {
				out = append(out, launch{c.Groups, c.GPU.Group[0]})
			}
		}
		return out
	}
	v100 := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, ExactArith: true, Regs: 32, Shared: 96 << 10}
	small := v100
	small.Slots, small.Units = 20*1536, 20
	// q's 4096 rows in 16 segments, k's and v's 1024 each in 32, at 512
	// threads: 128 + 32 + 32 workgroups.
	want := []launch{{256, 512}, {128, 512}, {448, 256}, {128, 896}}
	big, little := form(v100), form(small)
	if !slices.Equal(big, want) {
		t.Fatalf("V100: projections launch %v, want %v", big, want)
	}
	if len(little) != len(big) {
		t.Fatalf("small card: %d split launches, V100 %d", len(little), len(big))
	}
	for i := range big {
		if little[i].group*little[i].groups > big[i].group*big[i].groups {
			t.Fatalf("launch %d runs %v on the small card, %v on the V100: not the device's", i, little[i], big[i])
		}
	}
	if slices.Equal(little, big) {
		t.Fatalf("both cards launch %v: the split does not follow the device", big)
	}
	t.Logf("V100 %v, small card %v", big, little)
}
