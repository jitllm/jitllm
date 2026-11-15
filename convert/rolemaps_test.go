package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestARenameOfOneLayoutNeedsItsMarker: Falcon-40B's backwards norm names
// are read as its layout only where the file stores the tensor that marks it
// (block 0's attn_norm_2); a 7B file, or a reader with no file to ask, keeps
// the names as they are.
func TestARenameOfOneLayoutNeedsItsMarker(t *testing.T) {
	two, err := retargets(ArchFalcon, func(n string) bool { return n == "blk.0.attn_norm_2.weight" })
	if err != nil {
		t.Fatal(err)
	}
	for from, to := range map[jlm.Role]jlm.Role{
		jlm.RoleAttnNorm: jlm.RoleFFNNorm, jlm.RoleAttnNormBias: jlm.RoleFFNNormBias,
		jlm.RoleAttnNorm2: jlm.RoleAttnNorm, jlm.RoleAttnNorm2Bias: jlm.RoleAttnNormBias,
	} {
		if got := two[from]; got != to {
			t.Errorf("a two-norm file's %v is %v, want %v", from, got, to)
		}
	}
	for _, stored := range []func(string) bool{nil, func(string) bool { return false }} {
		one, err := retargets(ArchFalcon, stored)
		if err != nil {
			t.Fatal(err)
		}
		if len(one) != 0 {
			t.Errorf("a one-norm file renames %v", one)
		}
	}
}
