package oldkernels_test

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

var updateMatVec = flag.Bool("update-matvec", false, "rewrite testdata/matvec_ir.sha")

// matVecSweep is every matvec kernel at the shapes that reach a different
// code path of the emitter: each weight format against decode, a k split
// (global and in-group), rows a thread, a chunk of tokens (one and several a
// thread), a bias, a gate, the gate's second stream, an expert bank and its
// slots, the in-kernel operand prologue, an epilogue, and the segment
// launches. A refused shape is recorded as refused, so a shape that starts
// or stops building moves the golden too.
func matVecSweep() map[string]string {
	out := map[string]string{}
	add := func(name string, k *ir.Kernel, err error) {
		if err != nil {
			out[name] = "refused"
			return
		}
		out[name] = irHash(k)
	}
	g := &compose.Row{Width: 64, Rows: 1}
	in := g.Add(compose.Node{Kind: compose.NInput})
	epi := &compose.Epi{G: g, Out: g.Bin(compose.OpAdd, in, g.Load(g.Buf(compose.BufRow)))}
	shapes := []struct {
		name string
		s    kernels.MatVecShape
	}{
		{"decode", kernels.MatVecShape{K: 512, Rows: 64}},
		{"split4", kernels.MatVecShape{K: 512, Rows: 64, Split: 4}},
		{"group4", kernels.MatVecShape{K: 512, Rows: 64, Split: 4, GroupSplit: true}},
		{"rowt2", kernels.MatVecShape{K: 512, Rows: 64, Rowt: 2}},
		{"ntok8", kernels.MatVecShape{K: 512, Rows: 64, NTok: 8, Tok: 1}},
		{"ntok8tok4", kernels.MatVecShape{K: 512, Rows: 64, NTok: 8, Tok: 4}},
		{"ntok8tok8", kernels.MatVecShape{K: 512, Rows: 64, NTok: 8, Tok: 8}},
		{"bias", kernels.MatVecShape{K: 512, Rows: 64, Bias: true}},
		{"gate", kernels.MatVecShape{K: 512, Rows: 64, Gate: true, GateAct: kernels.ActSiLU}},
		{"gate2", kernels.MatVecShape{K: 512, Rows: 64, Gate2: true, GateAct: kernels.ActSiLU}},
		{"experts", kernels.MatVecShape{K: 512, Rows: 64, Experts: 4, Slots: 2}},
		{"slotact", kernels.MatVecShape{K: 512, Rows: 64, Experts: 4, Slots: 2, SlotAct: true}},
		{"actin", kernels.MatVecShape{K: 512, Rows: 64, In: kernels.ActIn{On: true, Window: 32}}},
		{"actinnorm", kernels.MatVecShape{K: 512, Rows: 64, In: kernels.ActIn{On: true, Norm: true, Eps: 1e-6, Window: 32}}},
		{"sub16", kernels.MatVecShape{K: 48, Rows: 64, Sub: 16}},
	}
	for q := kernels.Q4_0; q <= kernels.TQ2F; q++ {
		for _, sh := range shapes {
			s := sh.s
			s.T = q
			k, err := kernels.MatVec(s)
			add(fmt.Sprintf("%v/%s", q, sh.name), k, err)
		}
		k, err := kernels.MatVecEpi(kernels.MatVecShape{T: q, K: 512, Rows: 64}, epi)
		add(fmt.Sprintf("%v/epi", q), k, err)
		segs := []kernels.MatVecShape{{T: q, K: 512, Rows: 64}, {T: q, K: 512, Rows: 32}}
		k, err = kernels.MatVecSegments(segs, 1, false)
		add(fmt.Sprintf("%v/segments", q), k, err)
		k, err = kernels.MatVecSegments(segs, 4, true)
		add(fmt.Sprintf("%v/segments-group4", q), k, err)
	}
	return out
}

// TestMatVecIRUnchanged holds every matvec kernel's IR to
// testdata/matvec_ir.sha, recorded before the emitter's per-format
// arithmetic was lifted into fragments (-update-matvec rewrites it; a moved
// hash is a changed kernel).
func TestMatVecIRUnchanged(t *testing.T) {
	got := matVecSweep()
	var keys []string
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s %s\n", k, got[k])
	}
	path := filepath.Join("testdata", "matvec_ir.sha")
	if *updateMatVec {
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (rerun with -update-matvec to create it)", err)
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
	if built < 100 {
		t.Fatalf("only %d kernels built: the sweep does not reach the emitter", built)
	}
	t.Logf("%d matvec kernels, %d built, unchanged", len(gl), built)
}
