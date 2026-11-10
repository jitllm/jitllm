package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAMixtureWithNoReferenceIsRefused: experts weighted at their input
// over a biased gate or up bank, routed experts wider than the residual and
// latent experts weighted at their input have no reference to state them
// from, so the converter refuses each with ErrNotImplemented; Llama 4's
// unbiased input weighting, a biased mixture weighted at its output and
// latent experts weighted at their output convert.
func TestAMixtureWithNoReferenceIsRefused(t *testing.T) {
	bank := func(w uint64) jlm.Tensor {
		return jlm.Tensor{Role: jlm.RoleExpUpBank, Block: 0, Index: -1, NDim: 3, Dims: [4]uint64{w, 128, 4}, Name: "up"}
	}
	bias := func(r jlm.Role) jlm.Tensor {
		return jlm.Tensor{Role: r, Block: 0, Index: -1, NDim: 2, Dims: [4]uint64{128, 4}, Name: r.String()}
	}
	for _, tc := range []struct {
		name   string
		in     bool
		ts     []jlm.Tensor
		refuse bool
	}{
		{"weighted at the input, unbiased", true, []jlm.Tensor{bank(64)}, false},
		{"weighted at the output, biased", false, []jlm.Tensor{bank(64), bias(jlm.RoleExpGateBias), bias(jlm.RoleExpUpBias)}, false},
		{"latent, weighted at the output", false, []jlm.Tensor{bank(32)}, false},
		{"weighted at the input over a gate bias", true, []jlm.Tensor{bank(64), bias(jlm.RoleExpGateBias)}, true},
		{"weighted at the input over an up bias", true, []jlm.Tensor{bank(64), bias(jlm.RoleExpUpBias)}, true},
		{"latent, weighted at the input", true, []jlm.Tensor{bank(32)}, true},
		{"wider than the residual", false, []jlm.Tensor{bank(128)}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &jlm.Config{NLayer: 1, NEmbd: 64}
			if tc.in {
				c.Flags |= jlm.FlagExpertWeightIn
			}
			err := expertsFit(&Config{Config: c}, &jlm.Source{Config: c, Tensors: tc.ts})
			switch {
			case tc.refuse && !errors.Is(err, ErrNotImplemented):
				t.Fatalf("refused with %v, want ErrNotImplemented", err)
			case !tc.refuse && err != nil:
				t.Fatal(err)
			}
		})
	}
}
