package convert

import (
	"encoding/json"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAFilledOperandIsItsValueAtItsShape: a gather's filled operand is its
// value at its shape, counts read off the keys, on every block an every step
// runs on; one that reads a source, states no shape or is sized by what is
// not a count is refused.
func TestAFilledOperandIsItsValueAtItsShape(t *testing.T) {
	var from []mjOperand
	if err := json.Unmarshal([]byte(`[{"fill": 1, "shape": ["ssm.state_size"]}]`), &from); err != nil {
		t.Fatal(err)
	}
	step := hfGatherStep{role: jlm.RoleSSMBNorm, from: from, every: true}
	env := func(p string) any { return map[string]any{"ssm.state_size": 3.0}[p] }
	read := func(string) ([]float32, error) { return nil, nil }
	out, _, err := gatherBlocks(env, "blk.", []hfGatherStep{step}, nil, read, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("%d tensors for two blocks", len(out))
	}
	for b, e := range out {
		v, err := vecOf(&e)
		if err != nil {
			t.Fatal(err)
		}
		if e.Block != int32(b) || e.Role != jlm.RoleSSMBNorm || len(v) != 3 || v[0] != 1 || v[1] != 1 || v[2] != 1 {
			t.Errorf("block %d: %v of block %d is %v, want three ones", b, e.Role, e.Block, v)
		}
	}
	for _, bad := range []string{
		`[{"fill": 1, "shape": ["n"], "from": ["x"]}]`,
		`[{"fill": 1}]`,
		`[{"from": ["x"], "do": [{"map": "x"}], "shape": ["n"]}]`,
	} {
		if err := json.Unmarshal([]byte(bad), &from); err == nil {
			t.Errorf("%s was read", bad)
		}
	}
	if err := json.Unmarshal([]byte(`[{"fill": 1, "shape": ["ssm.state_size - 3.5"]}]`), &from); err != nil {
		t.Fatal(err)
	}
	step.from = from
	if _, _, err := gatherBlocks(env, "blk.", []hfGatherStep{step}, nil, read, nil, 2); err == nil {
		t.Error("a shape of -0.5 was filled")
	}
}
