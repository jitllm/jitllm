package jit

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

var (
	composeModel = flag.String("compose-model", "synth-ernie45", "the fixture TestComposeIR forms")
	composeShow  = flag.String("compose-show", "", "TestComposeIR writes out the IR of the kernels with this name prefix, stopping after the pass of this name (after:NAME)")
	composeHost  = flag.Bool("compose-host", false, "TestComposeIR forms the block on the host")
	composePart  = flag.Int("compose-part", 1, "the part TestComposeIR forms")
	composeIndex = flag.Int("compose-index", -1, "TestComposeIR writes out the IR of the kernel at this index")
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
	parts, err := Front(f)
	if err != nil {
		t.Fatal(err)
	}
	d := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	if *composeHost {
		d = Device{Kind: DeviceCPU, Slots: 6, Lanes: 8}
	}
	prims, err := r.decompose(f, parts[*composePart], *composePart, Step{Rows: 1, Page: 64, Keys: 5}, d)
	if err != nil {
		t.Fatal(err)
	}
	var ks []*Kernel
	for _, p := range prims {
		k, err := generate(p)
		if err != nil {
			t.Fatal(err)
		}
		k.Part, k.Of = parts[*composePart].Kind, *composePart
		ks = append(ks, k)
	}
	name, stop, _ := strings.Cut(*composeShow, "@")
	for _, ps := range r.Profiles[d.Kind].Passes {
		ks, err = ps.Apply(ks, d)
		if err != nil {
			t.Fatal(err)
		}
		if ps.Name() == stop {
			break
		}
	}
	for i, k := range ks {
		t.Logf("%d %s %v loops %d passes %v", i, k.Name, k.Space, len(k.Loops), k.Passes)
		if name != "" && strings.HasPrefix(k.Name, name) || i == *composeIndex {
			t.Log(k.String())
			for i, l := range k.Loops {
				t.Logf("loop %d %+v", i, l)
			}
			t.Logf("order %v", k.topOrder())
			if *composeHost {
				t.Logf("seq %v", k.Seq)
				l, err := r.Lowerings[DeviceCPU].Lower(k, d)
				if err != nil {
					t.Log(err)
				} else {
					t.Log(objdump(t, l.CPU))
				}
			}

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
		for _, a := range k.Operands(&nd) {
			if walk(*a) {
				return true
			}
		}
		return false
	}
	return walk(root)
}

func objdump(t *testing.T, code []byte) string {
	f := filepath.Join(t.TempDir(), "k.bin")
	if err := os.WriteFile(f, code, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-mi386:x86-64", "--no-show-raw-insn", f).Output()
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

// TestEveryKernelRecordsItsPasses composes the fixture on each device and
// holds every kernel formed to naming the passes that formed it.
func TestEveryKernelRecordsItsPasses(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	r := Default()
	s := Step{Rows: 1, Page: 64, Keys: 5}
	for _, d := range []Device{{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: DeviceCPU, Slots: 6, Lanes: 8}} {
		plan, err := r.Compose(f, s, func(Part) Device { return d })
		if plan == nil {
			t.Fatal(err)
		}
		n := 0
		for _, pp := range plan.Parts {
			for _, c := range pp.Codes {
				n++
				if len(c.Passes) == 0 {
					t.Errorf("device %d: %s %s records no pass", d.Kind, pp.Part, c.Name)
				}
			}
		}
		if n == 0 {
			t.Errorf("device %d formed no kernel", d.Kind)
		}
		t.Logf("device %d: %d kernels, each naming its passes", d.Kind, n)
	}
}

// runProbe records, for each call, the parts whose kernels it was given.
type runProbe struct{ calls *[][]int }

func (runProbe) Name() string { return "runprobe" }

func (p runProbe) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	seen := map[int]bool{}
	var of []int
	for _, k := range ks {
		if !seen[k.Of] {
			seen[k.Of] = true
			of = append(of, k.Of)
		}
	}
	*p.calls = append(*p.calls, of)
	return ks, nil
}

// TestPassesSeeRuns holds the passes to seeing each run of consecutive
// parts placed on one device in one call, and a part placed elsewhere
// cutting the run in two.
func TestPassesSeeRuns(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gpu := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	cpu := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8}
	for _, tc := range []struct {
		name string
		host func(Part) bool
		want [][]int
	}{
		{"all on the card", func(p Part) bool { return false }, [][]int{{0, 1, 2, 3, 4, 5, 6}}},
		{"block 1 on the host", func(p Part) bool { return p.Kind == PartBlock && p.Block == 1 }, [][]int{{0, 1, 2}, {3}, {4, 5, 6}}},
	} {
		var calls [][]int
		r := Default()
		for kind, prof := range r.Profiles {
			prof.Passes = append([]Pass{runProbe{&calls}}, prof.Passes...)
			r.Profiles[kind] = prof
		}
		place := func(p Part) Device {
			if tc.host(p) {
				return cpu
			}
			return gpu
		}
		if _, err := r.Compose(f, Step{Rows: 1, Page: 64, Keys: 5}, place); err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(calls) != fmt.Sprint(tc.want) {
			t.Errorf("%s: the passes saw parts %v, want %v", tc.name, calls, tc.want)
		}
		t.Logf("%s: runs %v", tc.name, calls)
	}
}
