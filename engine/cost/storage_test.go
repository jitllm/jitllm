package cost

import (
	"testing"

	"github.com/jitllm/jitllm/jit"
)

// TestEveryStorageStatesWhatItsKernelsReach holds each storage the
// composer's plan sizes on the GPU to the bytes the estimate finds its
// kernels reach in it: a buffer the plan states too small is one the engine
// would allocate too small. Storages the engine sizes (its history, page
// table and rotary planes) are not the plan's to state. The host is not held:
// the estimate's interval over a host kernel's flat index loses the
// correlation of its terms (a 256-byte row read to 26 MB), so its reach is a
// bound, not a measure.
func TestEveryStorageStatesWhatItsKernelsReach(t *testing.T) {
	f := composing(t, *composeModel+".jlm")
	s := jit.Step{Rows: 1, Page: 64, Keys: 9}
	for _, d := range []Device{{Device: jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}}} {
		plan, err := jit.Default().Compose(f, s, func(jit.Part) jit.Device { return d.Device })
		if err != nil {
			t.Fatal(err)
		}
		ccs, err := EstimateCodes(jit.Default(), f, s, func(jit.Part) Device { return d }, true)
		if err != nil {
			t.Fatal(err)
		}
		var codes []jit.Code
		for _, pp := range plan.Parts {
			codes = append(codes, pp.Codes...)
		}
		if len(codes) != len(ccs) {
			t.Fatalf("%v: the plan forms %d kernels, the estimate %d", d.Kind, len(codes), len(ccs))
		}
		checked := 0
		for i, cc := range ccs {
			for j, b := range codes[i].Binds {
				st := plan.Storages[b.Storage]
				if st.Source.Engine() {
					continue
				}
				checked++
				if reach := cc.Binds[j].Reach; st.Bytes < reach {
					t.Errorf("%v %s: %s storage %q of part %d states %d bytes, its kernel reaches %d", d.Kind, codes[i].Name, st.Source, b.Operand.Name, st.Key.Of, st.Bytes, reach)
				}
			}
		}
		if checked == 0 {
			t.Fatalf("%v: no binding checked", d.Kind)
		}
		t.Logf("%v: %d bindings of %d storages", d.Kind, checked, len(plan.Storages))
	}
}
