package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAttentionQKVIsSplitByItsRows: an attending block's attn_qkv of the
// query heads' and two key-value heads' rows is split into q, k and v, in
// that order and those widths, whatever the architecture; a recurrent
// block's tensor of that name, or one beside the block's own attn_q, is
// left whole, and one of other rows is refused; likewise an ffn_up of twice
// the width is gate|up only where the block stores no ffn_gate.
func TestAttentionQKVIsSplitByItsRows(t *testing.T) {
	c := &Config{Config: &jlm.Config{NLayer: 2, NEmbd: 16, NHead: 4, NKVHead: 2, HeadDim: 4, NFFN: 24}}
	matrix := func(role jlm.Role, name string, rows int, block int32) *jlm.Tensor {
		vals := make([]float32, rows*16)
		for i := range vals {
			vals[i] = float32(i)
		}
		e := &jlm.Tensor{Role: role, Block: block, Index: -1, Name: name}
		setF32Dims(e, vals, 16, uint64(rows))
		return e
	}
	qkv := func(rows int, block int32) *jlm.Tensor {
		return matrix(jlm.RoleAttnQKV, "blk.0.attn_qkv.weight", rows, block)
	}
	none := func(string) bool { return false }
	has := func(name string) func(string) bool { return func(n string) bool { return n == name } }
	parts, ok, err := unfuse(qkv(32, 0), c, none)
	if err != nil || !ok || len(parts) != 3 {
		t.Fatalf("an attending block's q|k|v: %d parts, %v, %v", len(parts), ok, err)
	}
	for i, want := range []struct {
		role jlm.Role
		rows uint64
	}{{jlm.RoleAttnQ, 16}, {jlm.RoleAttnK, 8}, {jlm.RoleAttnV, 8}} {
		if parts[i].Role != want.role || parts[i].Dims[1] != want.rows {
			t.Errorf("part %d is %v of %d rows, want %v of %d", i, parts[i].Role, parts[i].Dims[1], want.role, want.rows)
		}
	}
	c.LayerKinds = []jlm.LayerKind{jlm.LayerFullAttn, jlm.LayerLinearAttn}
	if _, ok, err := unfuse(qkv(32, 1), c, none); ok || err != nil {
		t.Errorf("a recurrent block's attn_qkv was split (%v, %v)", ok, err)
	}
	if _, ok, err := unfuse(qkv(32, 0), c, none); !ok || err != nil {
		t.Errorf("the attending block beside it was not split (%v, %v)", ok, err)
	}
	if _, _, err := unfuse(qkv(40, 0), c, none); err == nil {
		t.Error("an attn_qkv of other rows was not refused")
	}
	if _, ok, err := unfuse(qkv(32, 0), c, has("blk.0.attn_q.weight")); ok || err != nil {
		t.Errorf("an attn_qkv beside the block's own attn_q was split (%v, %v)", ok, err)
	}
	up := matrix(jlm.RoleFFNUp, "blk.0.ffn_up.weight", 48, 0)
	if parts, ok, err := unfuse(up, c, none); !ok || err != nil || len(parts) != 2 || parts[0].Role != jlm.RoleFFNGate {
		t.Errorf("an ffn_up of twice the width with no gate beside it: %d parts, %v, %v", len(parts), ok, err)
	}
	if _, ok, err := unfuse(up, c, has("blk.0.ffn_gate.weight")); ok || err != nil {
		t.Errorf("an ffn_up of twice the width beside its own gate was split (%v, %v)", ok, err)
	}
}
