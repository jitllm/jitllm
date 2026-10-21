package jit

import (
	"flag"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

var (
	composeModel = flag.String("compose-model", "synth-ernie45", "the fixture TestComposeIR forms")
	composeShow  = flag.String("compose-show", "", "TestComposeIR writes out the IR of the kernels with this name prefix, stopping after the pass of this name (after:NAME)")
)

// TestComposeIR forms the fixture's first block on a GPU, stopping where
// asked, and writes out the IR of the kernels asked for.
func TestComposeIR(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	r := Default()
	parts, err := front(f)
	if err != nil {
		t.Fatal(err)
	}
	d := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	prims, err := r.decompose(f, parts[0], Step{Rows: 1, Page: 64, Keys: 5}, d)
	if err != nil {
		t.Fatal(err)
	}
	var ks []*Kernel
	for _, p := range prims {
		k, err := generate(p)
		if err != nil {
			t.Fatal(err)
		}
		ks = append(ks, k)
	}
	name, stop, _ := strings.Cut(*composeShow, "@")
	for _, ps := range r.Profiles[DeviceGPU].Passes {
		ks, err = ps.Apply(ks, d)
		if err != nil {
			t.Fatal(err)
		}
		if ps.Name() == stop {
			break
		}
	}
	for _, k := range ks {
		t.Logf("%s %v loops %d passes %v", k.Name, k.Space, len(k.Loops), k.Passes)
		if name != "" && strings.HasPrefix(k.Name, name) {
			t.Log(k.String())
			for i, l := range k.Loops {
				t.Logf("loop %d %+v", i, l)
			}
			t.Logf("order %v", k.topOrder())

		}
	}
}

func (k *Kernel) explainFuse(t *testing.T, a, b int) {
	ia, ib := k.iterOf(a), k.iterOf(b)
	t.Logf("fuse %d %d: range %v %v trips %d %d inner %v %v storesAt(b) %v iters %d %d folds(b) %v",
		a, b, k.Loops[a].Range, k.Loops[b].Range, k.Loops[a].Trip, k.Loops[b].Trip, k.hasInner(a), k.hasInner(b), k.storesAt(b), ia, ib, k.foldsOf(b))
	for _, s := range k.Stores {
		if s.At == a && !s.Barrier && !s.Run {
			t.Logf("  store buf %d %v ax %v", s.Buf, k.Bufs[s.Buf].Operand.Name, s.Ax)
		}
	}
	for _, f := range k.foldsOf(b) {
		for i, nd := range k.Nodes {
			if nd.Op == OpLoad && k.reachesNode(k.Nodes[f].A, i) {
				t.Logf("  fold %d load %d buf %d %s ax %v", f, i, nd.Buf, k.Bufs[nd.Buf].Operand.Name, nd.Ax)
			}
		}
	}
}

func (k *Kernel) reachesNode(root, x int) bool {
	seen := map[int]bool{}
	var walk func(i int) bool
	walk = func(i int) bool {
		if i == x {
			return true
		}
		if seen[i] {
			return false
		}
		seen[i] = true
		nd := k.Nodes[i]
		for _, a := range k.args(&nd) {
			if walk(*a) {
				return true
			}
		}
		return false
	}
	return walk(root)
}
