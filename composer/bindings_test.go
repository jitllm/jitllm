package composer_test

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestBindingsAreTheCodesAccesses forms a Llama-shaped block at one row and
// at a capacity of rows, on a device of CUDA's statement and one of
// Vulkan's, and holds every binding to what its lowered kernel does with it:
// a binding that writes is a parameter the code stores to, and one that
// neither reads nor writes is one the code never touches -- the caller binds
// it to a word, so a code reaching it would read past it. And no form of a
// step of several rows binds a buffer of a value for every row, head and
// key, at any key bound (a decode's partials outweigh its one row's): the
// attention's range is split into partitions of a workgroup's chunk, one
// where the bound fits one (composer.SplitRange), and each partition's scores
// live in workgroup memory (composer.Stage). Before, the temporaries declared over
// every key were bound as written however staged -- a 256-row prompt over
// 3328 keys asked about 0.5 GB for them -- and a bound of one chunk or less
// kept them in device memory.
func TestBindingsAreTheCodesAccesses(t *testing.T) {
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp} {
		formats[r] = jlm.TypeQ4S
	}
	formats[jlm.RoleFFNDown] = jlm.TypeQ6S
	formats[jlm.RoleAttnNorm], formats[jlm.RoleFFNNorm] = jlm.TypeF32, jlm.TypeF32
	form := func(d composer.Device, R, keys int) []composer.Code {
		t.Helper()
		_, codes, err := composer.Default().Form(composer.Request{Graph: llamaBlock(), Formats: formats,
			Step: composer.Step{Rows: R, Page: 64, Keys: keys, ActWindow: 32, Seqs: 1}, Device: d,
			History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}})
		if err != nil {
			t.Fatalf("at %d rows over %d keys: %v", R, keys, err)
		}
		return codes
	}
	// plane is the scratch binding the codes read or write that holds a
	// value for every row, query head and key, and its bytes.
	plane := func(codes []composer.Code, R, keys int) (string, int64) {
		for _, c := range codes {
			for _, b := range c.Binds {
				if b.Operand.Kind == composer.OperandScratch && (b.Read || b.Writes) && b.Bytes >= int64(R*32*keys*4) {
					return c.Name + " " + b.Operand.Name, b.Bytes
				}
			}
		}
		return "", 0
	}
	cuda := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32}
	for _, d := range []composer.Device{cuda, {Kind: composer.DeviceGPU, Lanes: 32}} {
		checked := 0
		for _, R := range []int{1, 64, 256} {
			for _, keys := range []int{256, 1024, 2048, 3328, 8192} {
				codes := form(d, R, keys)
				for _, c := range codes {
					if err := accesses(c); err != nil {
						t.Fatalf("units %d, %d rows over %d keys: %v", d.Units, R, keys, err)
					}
					checked++
				}
				if name, n := plane(codes, R, keys); R > 1 && n > 0 {
					t.Fatalf("units %d, %d rows over %d keys: %s binds %d bytes, a value for every key", d.Units, R, keys, name, n)
				}
			}
		}
		if checked == 0 {
			t.Fatal("no code was formed: the gate checked nothing")
		}
	}
}

// accesses reports a binding of c that claims a write its code does not
// store, or one bound as untouched that the code loads or stores. A code's
// parameters are its bindings, in order.
func accesses(c composer.Code) error {
	k := c.GPU
	if k == nil {
		return fmt.Errorf("%s has no GPU code", c.Name)
	}
	if len(k.Params) != len(c.Binds) {
		return fmt.Errorf("%s has %d parameters and %d bindings", c.Name, len(k.Params), len(c.Binds))
	}
	param := func(v ir.Value) int {
		for v > 0 {
			op := k.Ops[v-1]
			switch {
			case op.Kind == ir.OpParam:
				return int(op.Imm)
			case op.Type == ir.Ptr && op.Kind == ir.OpAdd:
				v = op.Args[0]
			default:
				return -1
			}
		}
		return -1
	}
	loads, stores := make([]int, len(k.Params)), make([]int, len(k.Params))
	for _, op := range k.Ops {
		switch op.Kind {
		case ir.OpLoad, ir.OpLoadV:
			if p := param(op.Args[0]); p >= 0 {
				loads[p]++
			}
		case ir.OpStore, ir.OpStoreV:
			if p := param(op.Args[0]); p >= 0 {
				stores[p]++
			}
		}
	}
	for i, b := range c.Binds {
		switch {
		case b.Writes && stores[i] == 0:
			return fmt.Errorf("%s binds %s as written and stores nothing to it", c.Name, b.Operand.Name)
		case !b.Read && !b.Writes && loads[i]+stores[i] > 0:
			return fmt.Errorf("%s binds %s as untouched and reaches it %d times", c.Name, b.Operand.Name, loads[i]+stores[i])
		}
	}
	return nil
}
