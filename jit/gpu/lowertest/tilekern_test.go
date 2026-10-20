package lowertest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/msl"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestTileKernelsAssemble lowers the shipped collective-tile kernels
// (mainkernels.GemmTile, its grouped form, mainkernels.FlashPrefillTile) to PTX at
// sm_75 and sm_86 and assembles them with ptxas, and to MSL. They are not in
// all() because SPIR-V does not lower a tile carried across a loop or staged
// in workgroup memory, which both need; the backend gates run them on the
// devices that do.
func TestTileKernelsAssemble(t *testing.T) {
	ptxas, _ := exec.LookPath("ptxas")
	dir := t.TempDir()
	ks := map[string]*ir.Kernel{}
	for _, q := range []mainkernels.Quant{mainkernels.Q4_K, mainkernels.Q6_K, mainkernels.Q8_0, mainkernels.Q4_0} {
		sub, _, _, _ := mainkernels.Layout(q)
		tl := mainkernels.TileGemm{MT: 4, NT: 4, WM: 2, WN: 2, KB: max(32/sub, 1)}
		k, err := mainkernels.GemmTile(mainkernels.MatVecShape{T: q, K: 1024, Rows: 128, NTok: 64, Bias: true}, tl)
		if err != nil {
			t.Fatal(err)
		}
		ks["gemmtile/"+q.String()] = k
		k, err = mainkernels.GemmTile(mainkernels.MatVecShape{T: q, K: 1024, Rows: 128, NTok: 64, Experts: 4}, tl)
		if err != nil {
			t.Fatal(err)
		}
		ks["gemmtile-grouped/"+q.String()] = k
	}
	for _, hd := range []int{64, 128} {
		ft, _ := mainkernels.FlashTileFor(hd)
		k, err := mainkernels.FlashPrefillTile(mainkernels.FlashPrefill70Shape{Heads: 8, KVHeads: 2, Dim: hd,
			Rows: 64, KStride: 4096, Scale: .125}, ft)
		if err != nil {
			t.Fatal(err)
		}
		ks[fmt.Sprintf("flashtile/d%d", hd)] = k
	}
	for name, k := range ks {
		t.Run(name, func(t *testing.T) {
			if _, err := msl.Emit(k); err != nil {
				t.Fatalf("msl: %v", err)
			}
			for _, sm := range []string{"sm_75", "sm_86"} {
				src, err := ptx.Lower(k, sm)
				if strings.HasPrefix(name, "flashtile") && sm == "sm_75" {
					// float32 operands are three tf32 products, sm_80 on.
					if err == nil {
						t.Fatalf("ptx %s lowered float32 operands", sm)
					}
					continue
				}
				if err != nil {
					t.Fatalf("ptx %s: %v", sm, err)
				}
				if ptxas == "" {
					continue
				}
				f := filepath.Join(dir, strings.ReplaceAll(name, "/", "_")+sm+".ptx")
				if err := os.WriteFile(f, []byte(src), 0o600); err != nil {
					t.Fatal(err)
				}
				if out, err := exec.Command(ptxas, "--gpu-name", sm, f,
					"-o", f+".cubin").CombinedOutput(); err != nil {
					t.Fatalf("ptxas %s: %v\n%s", sm, err, out)
				}
			}
		})
	}
	if ptxas == "" {
		t.Log("ptxas not installed: PTX is lowered but not assembled")
	}
}
