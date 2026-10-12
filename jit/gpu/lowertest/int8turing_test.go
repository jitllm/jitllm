package lowertest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
)

// TestInt8MMALowersOnTuring lowers the int8 matrix kernels (GemmInt8 and
// MatVecMMA) for sm_75, where the integer instruction is m8n8k16 only, and
// assembles them with ptxas. An m16n8 integer mnemonic assembles only from
// sm_80, so ptxas is the violation check, and the text is checked too in case
// ptxas is missing. sm_86 keeps the m16n8 shapes.
func TestInt8MMALowersOnTuring(t *testing.T) {
	ptxas, _ := exec.LookPath("ptxas")
	dir := t.TempDir()
	for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0} {
		gi, err := kernels.GemmInt8(kernels.MatVecShape{T: q, K: 1024, Rows: 128, NTok: 64},
			kernels.Int8Tile{MT: 2, NT: 2, WM: 2, WN: 2})
		if err != nil {
			t.Fatal(err)
		}
		mm, err := kernels.MatVecMMA(kernels.MatVecShape{T: q, K: 1024, Rows: 128, NTok: 64, MT: 2, NT: 2, ActWin: 32})
		if err != nil {
			t.Fatal(err)
		}
		for name, k := range map[string]func(string) (string, error){
			"gemmint8_" + q.String():  func(sm string) (string, error) { return ptx.Lower(gi, sm) },
			"matvecmma_" + q.String(): func(sm string) (string, error) { return ptx.Lower(mm, sm) },
		} {
			for _, sm := range []string{"sm_75", "sm_86"} {
				src, err := k(sm)
				if err != nil {
					t.Fatalf("%s %s: %v", name, sm, err)
				}
				turing := strings.Contains(src, "m8n8k16.row.col.s32.s8")
				m16 := strings.Contains(src, "m16n8k32.row.col.s32.s8") || strings.Contains(src, "m16n8k16.row.col.s32.s8")
				if sm == "sm_75" && (!turing || m16) || sm == "sm_86" && (turing || !m16) {
					t.Fatalf("%s %s: m8n8k16 %v, m16n8 %v", name, sm, turing, m16)
				}
				if ptxas == "" {
					continue
				}
				f := filepath.Join(dir, name+sm+".ptx")
				if err := os.WriteFile(f, []byte(src), 0o600); err != nil {
					t.Fatal(err)
				}
				if out, err := exec.Command(ptxas, "--gpu-name", sm, f, "-o", f+".cubin").CombinedOutput(); err != nil {
					t.Fatalf("ptxas %s %s: %v\n%s", name, sm, err, out)
				}
			}
		}
	}
	if ptxas == "" {
		t.Log("ptxas not installed: PTX is lowered but not assembled")
	}
}
