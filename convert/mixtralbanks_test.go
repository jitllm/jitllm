package convert

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPerExpertGGUFBecomesBanks converts a Mixtral-era GGUF (one 2-D matrix
// per expert, blk.N.ffn_gate.E.weight) and demands the container carry the
// same three banks a modern mixture does. The device reads a mixture only
// through its banks, so per-expert tensors kept every block on the host. The
// bank is checked byte for byte against its experts: the wrong expert order
// has the right shape and routes every token to another expert's weights.
func TestPerExpertGGUFBecomesBanks(t *testing.T) {
	path := testmodels.Path("Mixtral-8x7B-Instruct-Q3_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
	}
	s := ggufSource(t, path)
	c := s.Config
	if c.NExpert != 8 || c.NLayer != 32 {
		t.Fatalf("not the Mixtral this gate is about: %d experts, %d layers", c.NExpert, c.NLayer)
	}
	// feed_forward_length is one expert's width on Mixtral, and a bank is
	// sized from this number.
	if c.NFFNExp != 14336 {
		t.Fatalf("NFFNExp = %d, want 14336 (one expert's width)", c.NFFNExp)
	}
	banks := map[jlm.Role]int{}
	var gate0 *jlm.Tensor
	for i := range s.Tensors {
		x := &s.Tensors[i]
		switch x.Role {
		case jlm.RoleExpGate, jlm.RoleExpUp, jlm.RoleExpDown:
			t.Fatalf("%s reached the container as a per-expert %v; a mixture's "+
				"experts are a bank", x.Name, x.Role)
		case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
			banks[x.Role]++
			want := [3]uint64{uint64(c.NEmbd), uint64(c.NFFNExp), uint64(c.NExpert)}
			if x.Role == jlm.RoleExpDownBank {
				want = [3]uint64{uint64(c.NFFNExp), uint64(c.NEmbd), uint64(c.NExpert)}
			}
			if x.NDim != 3 || [3]uint64(x.Dims[:3]) != want || x.Index != -1 {
				t.Fatalf("%s: %d-D %v index %d, want 3-D %v index -1",
					x.Name, x.NDim, x.Dims[:x.NDim], x.Index, want)
			}
			if x.Role == jlm.RoleExpGateBank && x.Block == 0 {
				gate0 = x
			}
		}
	}
	for _, r := range []jlm.Role{jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank} {
		if banks[r] != int(c.NLayer) {
			t.Fatalf("%v: %d bank(s), want one per block (%d)", r, banks[r], c.NLayer)
		}
	}
	if gate0 == nil || gate0.Load == nil {
		t.Fatal("block 0's gate bank is missing or has no bytes to load")
	}
	got, err := gate0.Load()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var want []byte
	for e := 0; e < int(c.NExpert); e++ {
		x, ok := f.Get(fmt.Sprintf("blk.0.ffn_gate.%d.weight", e))
		if !ok {
			t.Fatalf("expert %d's gate is not in the source", e)
		}
		want = append(want, f.Bytes(x)...)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("block 0's gate bank (%d bytes) is not its eight experts back to back "+
			"in id order (%d bytes)", len(got), len(want))
	}
	t.Logf("32 blocks x 3 banks of %v, NFFNExp %d; block 0's gate bank is its "+
		"experts in id order, %d bytes", gate0.Type, c.NFFNExp, len(got))
}
