package tok

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSPMPartitionsItsSpecialTokens holds SentencePiece Encode's special-token
// partition:
//
//   - user-defined tokens are matched in the raw text before the score merge,
//     always: gemma's runs of spaces and tabs are such tokens, so "*   **" is
//     "*" "   " "**".
//   - control tokens are matched too when the caller parses specials
//     (EncodeSpecial): "<start_of_turn>" is one id.
//
// The goldens are llama-tokenize on the GGUFs these containers came from, with
// --no-parse-special for Encode and without it for EncodeSpecial.
func TestSPMPartitionsItsSpecialTokens(t *testing.T) {
	const text = "*   **Iconic\tx\t\ty<start_of_turn>"
	const chat = "<start_of_turn>user hi<end_of_turn>"
	cases := []struct {
		file             string
		plain, withSpecs []int32
	}{
		{"gemma-3-4b-it-Q4_K_M.jlm",
			[]int32{2, 236829, 139, 1018, 11405, 525, 255968, 236781, 255969, 236762, 236820, 3041, 236779, 1340, 236779, 887, 236813},
			[]int32{2, 105, 2364, 5631, 106}},
		{"gemma-2-2b-it-Q4_K_M.jlm",
			[]int32{2, 235287, 140, 688, 233912, 226, 235297, 255969, 235267, 235322, 2997, 235298, 559, 235298, 15508, 235313},
			[]int32{2, 106, 1645, 5827, 107}},
	}
	for _, c := range cases {
		t.Run(c.file, func(t *testing.T) {
			f, err := jlm.Open(testmodels.Path(c.file))
			if err != nil {
				testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- RULE 11, fetch it", err)
			}
			vc := f.Vocab()
			f.Close()
			v, err := New(vc)
			if err != nil {
				t.Fatal(err)
			}
			if got := v.Encode(text, true); !slices.Equal(got, c.plain) {
				t.Errorf("Encode(%q)\n got %v\nwant %v", text, got, c.plain)
			}
			if got := v.EncodeSpecial(chat, true); !slices.Equal(got, c.withSpecs) {
				t.Errorf("EncodeSpecial(%q)\n got %v\nwant %v", chat, got, c.withSpecs)
			}
		})
	}
}
