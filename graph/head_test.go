package graph

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// headFacts is what a container stores for a head of the given kind.
func headFacts(h Head) Facts {
	inv := map[jlm.Role]Shape{
		jlm.RoleOutputNorm: {K: h.NEmbd},
		h.Output:           {K: h.NEmbd, Rows: h.NVocab},
	}
	if h.NormBias {
		inv[jlm.RoleOutputNormBias] = Shape{K: h.NEmbd}
	}
	if h.Bias {
		inv[jlm.RoleOutputBias] = Shape{K: h.NVocab}
	}
	return Facts{Inventory: inv}
}

// TestHeadBindsWhatTheContainerStores builds the head under every one of its
// ops, verifies it against the container it states, and demands that it bind
// every tensor that container stores for it. Then a head built without an op
// the container stores leaves its tensor unbound, and one that binds a tensor
// the container does not store is refused.
func TestHeadBindsWhatTheContainerStores(t *testing.T) {
	for _, layerNorm := range []bool{false, true} {
		for _, normBias := range []bool{false, true} {
			for _, bias := range []bool{false, true} {
				for _, softcap := range []float64{0, 30} {
					for _, scale := range []float64{0, 1, 8} {
						if normBias && !layerNorm {
							continue
						}
						h := Head{NEmbd: 8, NVocab: 24, Eps: 1e-5, LayerNorm: layerNorm, NormBias: normBias,
							Output: jlm.RoleTokenEmbd, Bias: bias, Softcap: softcap, Scale: scale}
						name := fmt.Sprintf("layernorm %v, norm bias %v, bias %v, softcap %v, scale %v", layerNorm, normBias, bias, softcap, scale)
						g, f := h.Build(), headFacts(h)
						if err := Verify(g, f); err != nil {
							t.Errorf("%s: %v", name, err)
						}
						if u := Unbound(g, f); len(u) != 0 {
							t.Errorf("%s: leaves %v unbound", name, u)
						}
						if kind := g.Ops[0].Kind; layerNorm != (kind == OpLayerNorm) {
							t.Errorf("%s: the norm is %v", name, kind)
						}
						caps := 0
						for _, o := range g.Ops {
							if o.Kind == OpMap && o.Attr.Fn == MapSigmoid {
								caps++
							}
						}
						if want := softcap != 0; (caps == 1) != want || caps > 1 {
							t.Errorf("%s: %d softcaps", name, caps)
						}
						last := g.Ops[len(g.Ops)-1]
						scaled := last.Kind == OpMap && last.Attr.Fn == MapMul && last.Attr.Const == 1/scale && scale != 0
						if want := scale != 0 && scale != 1; scaled != want {
							t.Errorf("%s: ends in %v %v by %v, scaled %v want %v", name, last.Kind, last.Attr.Fn, last.Attr.Const, scaled, want)
						}
					}
				}
			}
		}
	}
	full := Head{NEmbd: 8, NVocab: 24, Eps: 1e-5, LayerNorm: true, NormBias: true, Output: jlm.RoleOutput, Bias: true, Softcap: 30, Scale: 8}
	for _, c := range []struct {
		name   string
		build  func(h *Head)
		unbind jlm.Role
	}{
		{"no output bias", func(h *Head) { h.Bias = false }, jlm.RoleOutputBias},
		{"no norm bias", func(h *Head) { h.NormBias = false }, jlm.RoleOutputNormBias},
	} {
		h := full
		f := headFacts(full)
		c.build(&h)
		if u := Unbound(h.Build(), f); !slices.Equal(u, []jlm.Role{c.unbind}) {
			t.Errorf("%s: %v unbound, want %v", c.name, u, c.unbind)
		}
	}
	h := full
	f := headFacts(full)
	delete(f.Inventory, jlm.RoleOutputBias)
	if err := Verify(h.Build(), f); err == nil || !strings.Contains(err.Error(), "does not store") {
		t.Errorf("a head binding a bias the container does not store: Verify said %v", err)
	}
}
