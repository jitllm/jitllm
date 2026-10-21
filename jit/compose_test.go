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
	prims, err := r.decompose(f, parts[1], 1, Step{Rows: 1, Page: 64, Keys: 5}, d)
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

// TestSuggestPlacesFromTheIR estimates placements from final IR and holds
// plan mode's suggestion to costing no more than any of them, and to the
// estimate of the placement it returns.
func TestSuggestPlacesFromTheIR(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	r := Default()
	gpu := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32, Bandwidth: 800e9, Launch: 4e-6, Link: 12e9}
	cpu := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, Bandwidth: 40e9, Launch: 3e-6, Link: 12e9}
	s := Step{Rows: 1, Page: 64, Keys: 5}
	forced := map[string]Placement{
		"card":          func(Part) Device { return gpu },
		"host":          func(Part) Device { return cpu },
		"head on host":  func(p Part) Device { return map[bool]Device{true: cpu, false: gpu}[p.Kind == PartHead] },
		"table on host": func(p Part) Device { return map[bool]Device{true: cpu, false: gpu}[p.Kind == PartTable] },
	}
	p, c, err := r.Suggest(f, s, []Device{gpu, cpu})
	if err != nil {
		t.Fatal(err)
	}
	again, err := r.Estimate(f, s, p)
	if err != nil {
		t.Fatal(err)
	}
	if d := again.Seconds - c.Seconds; d > 1e-12 || d < -1e-12 {
		t.Errorf("suggested %+v, its estimate %+v", c, again)
	}
	for name, q := range forced {
		e, err := r.Estimate(f, s, q)
		if err != nil {
			t.Fatal(err)
		}
		if e.Bytes == 0 || e.Launches == 0 {
			t.Errorf("%s: an empty estimate %+v", name, e)
		}
		if e.Seconds < c.Seconds-1e-12 {
			t.Errorf("%s at %+v beats the suggestion %+v", name, e, c)
		}
		t.Logf("%-13s %+v", name, e)
	}
	parts, _ := front(f)
	for _, part := range parts {
		t.Logf("suggested: %s on %d", part, p(part).Kind)
	}
}
