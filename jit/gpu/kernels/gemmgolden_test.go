package kernels_test

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

var updateGemm = flag.Bool("update-gemm", false, "rewrite testdata/gemm_ir.sha")

// gemmSweep is every chunk GEMM kernel at the shapes that reach a different
// code path: the staged binary16 GEMM at each instruction shape (m8n8k4,
// m16n8k8, m16n8k16), single and double staging, a k split, a bias, an
// expert bank, an epilogue and the segment launch; the tiled GEMM; the int8
// GEMM; the int8 matrix-instruction matvec; and the operand conversions. A
// refused shape is recorded as refused.
func gemmSweep() map[string]string {
	out := map[string]string{}
	add := func(name string, k *ir.Kernel, err error) {
		if err != nil {
			out[name] = "refused"
			return
		}
		out[name] = irHash(k)
	}
	g := &compose.Row{Width: 128, Rows: 1}
	in := g.Add(compose.Node{Kind: compose.NInput})
	epi := &compose.Epi{G: g, Out: g.Bin(compose.OpAdd, in, g.Load(g.Buf(compose.BufRow)))}
	tiles := []struct {
		name string
		t    kernels.VoltaTile
	}{
		{"m4n4", kernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1}},
		{"m2n2", kernels.VoltaTile{MT: 2, NT: 2, WM: 2, WN: 2, KB: 1}},
		{"m4n4k16", kernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1, F16K: 16}},
		{"m2n4k8", kernels.VoltaTile{MT: 2, NT: 4, WM: 2, WN: 2, KB: 1, F16K: 8}},
		{"deep", kernels.VoltaTile{MT: 4, NT: 4, WM: 2, WN: 4, KB: 2, F16K: 16, Single: true}},
	}
	for q := kernels.Q4_0; q <= kernels.TQ2F; q++ {
		for _, tl := range tiles {
			for _, v := range []struct {
				name string
				s    kernels.MatVecShape
			}{
				{"plain", kernels.MatVecShape{K: 1024, Rows: 256, NTok: 128}},
				{"split2", kernels.MatVecShape{K: 1024, Rows: 256, NTok: 128, Split: 2}},
				{"bias", kernels.MatVecShape{K: 1024, Rows: 256, NTok: 128, Bias: true}},
				{"experts", kernels.MatVecShape{K: 1024, Rows: 256, NTok: 128, Experts: 4}},
			} {
				s := v.s
				s.T = q
				k, err := kernels.GemmVolta(s, tl.t)
				add(fmt.Sprintf("volta/%v/%s/%s", q, tl.name, v.name), k, err)
			}
			s := kernels.MatVecShape{T: q, K: 1024, Rows: 256, NTok: 128}
			k, err := kernels.GemmVoltaEpi(s, tl.t, epi)
			add(fmt.Sprintf("volta/%v/%s/epi", q, tl.name), k, err)
		}
		segs := []kernels.MatVecShape{{T: q, K: 1024, Rows: 256, NTok: 128}, {T: q, K: 1024, Rows: 128, NTok: 128}}
		k, err := kernels.GemmVoltaSegments(segs, []kernels.VoltaTile{tiles[2].t, tiles[2].t})
		add(fmt.Sprintf("voltasegs/%v", q), k, err)
		for _, tt := range []kernels.TileGemm{{MT: 2, NT: 2, WM: 2, WN: 2, KB: 1}, {MT: 4, NT: 2, WM: 2, WN: 2, KB: 2, Tall: true}} {
			k, err := kernels.GemmTile(kernels.MatVecShape{T: q, K: 1024, Rows: 256, NTok: 128}, tt)
			add(fmt.Sprintf("tile/%v/%+v", q, tt), k, err)
		}
		for _, it := range []kernels.Int8Tile{{MT: 2, NT: 2, WM: 2, WN: 2}, {MT: 4, NT: 4, WM: 2, WN: 2, Single: true}} {
			k, err := kernels.GemmInt8(kernels.MatVecShape{T: q, K: 1024, Rows: 256, NTok: 128}, it)
			add(fmt.Sprintf("int8/%v/%+v", q, it), k, err)
			k, err = kernels.GemmInt8Epi(kernels.MatVecShape{T: q, K: 1024, Rows: 256, NTok: 128}, it, epi)
			add(fmt.Sprintf("int8/%v/%+v/epi", q, it), k, err)
		}
		for _, mn := range [][2]int{{2, 2}, {4, 4}} {
			k, err := kernels.MatVecMMA(kernels.MatVecShape{T: q, K: 1024, Rows: 256, NTok: 128, MT: mn[0], NT: mn[1], ActWin: 32})
			add(fmt.Sprintf("mma/%v/%dx%d", q, mn[0], mn[1]), k, err)
		}
		k, err = kernels.MatVecMMA70(kernels.MatVecShape{T: q, K: 1024, Rows: 256, NTok: 128})
		add(fmt.Sprintf("mma70/%v", q), k, err)
		k, err = kernels.ActF16T(q, 128, 1024)
		add(fmt.Sprintf("actf16t/%v", q), k, err)
		k, err = kernels.ActF16(q, 128, 1024)
		add(fmt.Sprintf("actf16/%v", q), k, err)
		k, err = kernels.ActF16TGather(q, 128, 1024, 64)
		add(fmt.Sprintf("actf16tgather/%v", q), k, err)
	}
	return out
}

// TestGemmIRUnchanged holds every chunk GEMM kernel's IR to
// testdata/gemm_ir.sha, recorded before the GEMMs' per-format arithmetic was
// lifted into fragments (-update-gemm rewrites it).
func TestGemmIRUnchanged(t *testing.T) {
	got := gemmSweep()
	var keys []string
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s %s\n", k, got[k])
	}
	path := filepath.Join("testdata", "gemm_ir.sha")
	if *updateGemm {
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (rerun with -update-gemm to create it)", err)
	}
	wl := strings.Split(strings.TrimSpace(string(want)), "\n")
	gl := strings.Split(strings.TrimSpace(b.String()), "\n")
	if len(wl) != len(gl) {
		t.Fatalf("%d kernels, the golden has %d", len(gl), len(wl))
	}
	built := 0
	for i := range wl {
		if wl[i] != gl[i] {
			t.Errorf("moved: %s (was %s)", gl[i], wl[i])
		}
		if !strings.HasSuffix(gl[i], " refused") {
			built++
		}
	}
	if built < 150 {
		t.Fatalf("only %d kernels built: the sweep does not reach the emitters", built)
	}
	t.Logf("%d GEMM kernels, %d built, unchanged", len(gl), built)
}
