package convert

import (
	"encoding/json"
	"fmt"
	"strings"
)

// hfUint is a count config.json writes as a number or as a list of one value
// per layer (DeepSeek's moe_intermediate_size). A list whose
// entries differ is refused: the container has one value per model.
type hfUint uint32

func (u *hfUint) UnmarshalJSON(b []byte) error {
	var n uint32
	if err := json.Unmarshal(b, &n); err == nil {
		*u = hfUint(n)
		return nil
	}
	var l []uint32
	if err := json.Unmarshal(b, &l); err != nil {
		return err
	}
	for i := range l {
		if l[i] != l[0] {
			return fmt.Errorf("a per-layer list %v whose entries differ, and the container "+
				"carries one value: %w", l, ErrNotImplemented)
		}
	}
	if len(l) > 0 {
		*u = hfUint(l[0])
	}
	return nil
}

// hfGateAgrees refuses a scoring_func that contradicts the class's own gate.
func hfGateAgrees(c *hfConfig) error {
	want := ""
	if hardcodesSigmoid(c.Architectures) {
		want = "sigmoid"
	}
	if want != "" && c.ScoringFunc != "" && c.ScoringFunc != want {
		return fmt.Errorf("convert: %s: scoring_func %q, and %s gates with a %s whatever "+
			"the file says: %w", c.path, c.ScoringFunc, strings.Join(c.Architectures, ", "), want,
			ErrNotImplemented)
	}
	return nil
}
