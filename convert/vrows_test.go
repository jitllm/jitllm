package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestValueRowsAreTheValueHeads: v_proj is a value head per kv head, as wide
// as the value head where it differs from the key's (HeadDimV), as the output
// projection beside it already reads.
func TestValueRowsAreTheValueHeads(t *testing.T) {
	c := &Config{Config: &jlm.Config{NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, HeadDimV: 8}}
	at := func(role jlm.Role, rows uint64) error {
		e := jlm.Tensor{Name: "v", Role: role, NDim: 2, Dims: [4]uint64{64, rows}}
		return hfCheckShape(&e, c)
	}
	if err := at(jlm.RoleAttnV, 16); err != nil {
		t.Errorf("two value heads of 8: %v", err)
	}
	if err := at(jlm.RoleAttnV, 32); err == nil {
		t.Error("value rows sized by the key head were accepted")
	}
	if err := at(jlm.RoleAttnK, 32); err != nil {
		t.Errorf("two key heads of 16: %v", err)
	}
}
