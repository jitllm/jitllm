// Command ptxdump lowers named kernels to PTX so ptxas and cuobjdump can be
// pointed at them: the SASS is the only thing that says what a kernel actually
// issues (the MMA matvec turned out to be mostly address arithmetic).
package main

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

func main() {
	dir := os.Args[1]
	out := func(name string, mk func() (*ir.Kernel, error)) {
		k, err := mk()
		if err != nil {
			panic(err)
		}
		src, err := ptx.Lower(k, "sm_86")
		if err != nil {
			panic(err)
		}
		if err := os.WriteFile(dir+"/"+name+".ptx", []byte(src), 0o600); err != nil {
			panic(err)
		}
		fmt.Println(name)
	}
	// tinyllama's real prefill shapes at the shipping batch width.
	const R, H, HD, KV, GQA, MS = 128, 32, 64, 256, 8, 640
	out("mma-q3k", func() (*ir.Kernel, error) {
		return kernels.MatVecMMA(kernels.MatVecShape{
			T: kernels.Q3_K, K: 2048, Rows: 5632, NTok: 128, MT: 4, NT: 4})
	})
	out("dp4a-q3k", func() (*ir.Kernel, error) {
		return kernels.MatVec(kernels.MatVecShape{
			T: kernels.Q3_K, K: 2048, Rows: 5632, NTok: 128, Tok: 16, Rowt: 4})
	})
	out("scores", func() (*ir.Kernel, error) { return kernels.AttnScoresTiled(H, HD, KV, GQA, MS, 0.125, R, 4, 2, MS) })
	out("scores-mma", func() (*ir.Kernel, error) { return kernels.AttnScoresMMA(H, HD, KV, GQA, MS, 0.125, R, MS, 1) })
	out("attnacc", func() (*ir.Kernel, error) { return kernels.AttnAccTiled(H, HD, KV, GQA, MS, R, 4) })
	out("softmax", func() (*ir.Kernel, error) { return kernels.SoftmaxRows(H, MS, 32, R, 4) })
	out("normapply", func() (*ir.Kernel, error) { return kernels.NormApplyRows(2048, 64, 1e-5, false, R) })
	out("quantize", func() (*ir.Kernel, error) { return kernels.Quantize(R*2048, 256) })
}
