//go:build amd64 && jitllmtest

package server

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestHostBatchSampledRowsEqualAlonePreVNNI is the sampled gate on the host
// tier it failed on: an x86 without AVX-VNNI (the Xeon E5-2680 v4, the
// D-2123IT), forced here so a VNNI host runs it too. There the prefill GEMM
// folds a k-quant's super-block in integers (cpu.StationaryIntAcc) and the
// decode matvec does not, so a row stepped beside others and the same row
// decoded alone part unless the loaded model sums them in one order. The
// violation loads the model without that order and must part some row.
func TestHostBatchSampledRowsEqualAlonePreVNNI(t *testing.T) {
	old := cpu.ForceNoVNNIForTest(true)
	defer cpu.ForceNoVNNIForTest(old)
	if cpu.HostDotKind() != cpu.DotVEX {
		t.Fatal("ForceNoVNNIForTest did not make HostDotKind report DotVEX: this gate would run the VNNI arithmetic")
	}
	const name = "Llama-3.2-1B-Instruct-Q4_K_M.jlm"
	if i, got, want := sampledBatchParts(t, name); i >= 0 {
		t.Fatalf("sampled row %d differs from its run alone\nbatched %v\nalone   %v", i, got, want)
	}
	t.Run("violation-inexact", func(t *testing.T) {
		hostInexact = true
		defer func() { hostInexact = false }()
		i, got, want := sampledBatchParts(t, name)
		if i < 0 {
			t.Fatal("a model loaded without the one summation order still drew every row's tokens: this gate cannot see the arithmetic")
		}
		t.Logf("row %d parted, as it must\nbatched %v\nalone   %v", i, got, want)
	})
}
