package lowertest

import "testing"

// gpuElementwiseInventory is the device half of "every elementwise op", the
// same shape as jit_old/cpu's elementwiseInventory. It holds names, because `all`
// already builds and lowers them; the list makes an op registered nowhere a
// red line.
//
// LayerNorm needs one more kernel than RMSNorm for numerics: it needs the mean
// before the variance, since E[x^2]-E[x]^2 cancels in f32 when the mean dwarfs
// the variance.
var gpuElementwiseInventory = []string{
	"normpart", "normapply", "normapply1", // RMSNorm, and gemma's (w-1) form
	"layernormpart", "layernormvar", "layernormapply", "layernormapply-bias",
	"silumul", "actmul-gelu", // the gated activations
	// sigmoidmul is sigma(g)*v of two vectors, not SiLU's g*sigma(g) of one;
	// expressing it through silumul would be finite and wrong.
	"sigmoidmul",
	"act-silu", "act-gelu", // the ungated ones, for a vision tower's FFN
	"add",      // the residual
	"scale",    // dst *= alpha: the embedding scale and every attention score row
	"quantize", // f32 -> int8, the seam every matvec reads through
	// The rotary table: one thread per output pair, and a transcription of a
	// host kernel that must stay lowered.
	"ropetable", "ropetable-rows",
}

// TestEveryElementwiseOpLowersEverywhere asserts the device inventory is
// registered, which is what makes TestLowersEverywhere cover it.
//
// It is a coverage check, not a correctness one: assembling is not computing.
// Correctness for these lives in jit_old/gpu/backend, against a reference on every
// device present.
func TestEveryElementwiseOpLowersEverywhere(t *testing.T) {
	ks := all(t)
	if len(gpuElementwiseInventory) == 0 {
		t.Fatal("the inventory is empty; this test would pass and prove nothing")
	}
	for _, name := range gpuElementwiseInventory {
		if _, ok := ks[name]; !ok {
			t.Errorf("%s is in the elementwise inventory and is not registered in "+
				"lower_test.go's `all`, so no backend ever assembles it", name)
		}
	}
	t.Logf("%d elementwise kernels registered for every backend", len(gpuElementwiseInventory))
}
