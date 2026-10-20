package paritytest

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// Scratch: main's kernel text at a shape, into $ZZ.
func TestZZDump(t *testing.T) {
	k, err := oldkernels.MatVec(oldkernels.MatVecShape{T: kernels.Q8_0, K: 64, Rows: 64, Split: 1})
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(os.Getenv("ZZ")+"/main_q8_decode.txt", []byte(recorder.Text(k)), 0o644)
}
