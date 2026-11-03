package model

import (
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestUnsparsifiedIsThePlainActivation: the activation a per-layer input's
// gate reads, taken from a sparse block's stated one with its gaussian
// top-k read as the gate, is the plain block's term for term -- so a model
// whose every block is sparse states it too -- and a plain activation comes
// back as it is.
func TestUnsparsifiedIsThePlainActivation(t *testing.T) {
	c := &jlm.Config{NLayer: 4, Flags: jlm.FlagGELU, NSparse: 2, SparseStd: 1.6448536}
	sparse, err := convert.Activation(c, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := convert.Activation(c, 3, true)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(sparse.Terms, plain.Terms) {
		t.Fatal("the sparse and plain activations are one expression: the gate cannot see the sparsity")
	}
	got := unsparsified(&sparse)
	if err := got.Valid(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Terms, plain.Terms) {
		t.Fatalf("unsparsified:\n  %v\nthe plain activation:\n  %v", got.Terms, plain.Terms)
	}
	if p := unsparsified(&plain); p != &plain {
		t.Fatalf("a plain activation rewritten: %v", p.Terms)
	}
}
