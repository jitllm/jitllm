package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// expertsFit refuses a mixture no reference states: experts weighted at
// their input (Llama 4's, which has no expert biases) over a biased gate or
// up bank, routed experts wider than the residual, or latent experts --
// narrower than it -- weighted at their input.
func expertsFit(s *jlm.Source) error {
	c := s.Config
	in := c.Flags.Has(jlm.FlagExpertWeightIn)
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch {
		case in && (t.Role == jlm.RoleExpGateBias || t.Role == jlm.RoleExpUpBias):
			return fmt.Errorf("convert: %v weights its experts at their input over a biased bank (%s): %w",
				c.Arch, t.Name, ErrNotImplemented)
		case t.Role != jlm.RoleExpUpBank || t.NDim < 1 || t.Dims[0] == uint64(c.NEmbd):
		case t.Dims[0] > uint64(c.NEmbd) || in:
			return fmt.Errorf("convert: %v's routed experts are %d wide on a residual of %d (weighted at their input %v): %w",
				c.Arch, t.Dims[0], c.NEmbd, in, ErrNotImplemented)
		}
	}
	return nil
}
