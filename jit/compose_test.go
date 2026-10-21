package jit

import (
	"flag"
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
	parts, err := front(f)
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
				l, err := cpuLowering{}.Lower(k, d)
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

// TestEngineTakesOrForcesThePlacement composes the fixture on plan mode's
// suggestion and on a placement the engine forces, each part landing where
// its placement put it.
func TestEngineTakesOrForcesThePlacement(t *testing.T) {
	f, err := jlm.Open(testmodels.Path(*composeModel + ".jlm"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	r := Default()
	gpu := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: 32, Bandwidth: 800e9, Launch: 4e-6, Link: 12e9}
	cpu := Device{Kind: DeviceCPU, Slots: 6, Lanes: 8, Bandwidth: 40e9, Launch: 3e-6, Link: 12e9}
	s := Step{Rows: 1, Page: 64, Keys: 5}
	suggested, _, err := r.Suggest(f, s, []Device{gpu, cpu})
	if err != nil {
		t.Fatal(err)
	}
	forced := func(p Part) Device {
		if p.Kind == PartSample {
			return cpu
		}
		return gpu
	}
	for name, p := range map[string]Placement{"suggested": suggested, "forced": forced} {
		plan, err := r.Compose(f, s, p)
		if plan == nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i, pp := range plan.Parts {
			if pp.Device != p(pp.Part) {
				t.Errorf("%s: %s on %d, placed on %d", name, pp.Part, pp.Device.Kind, p(pp.Part).Kind)
			}
			if i > 0 && plan.Parts[i-1].Device != pp.Device {
				seen := false
				for _, tr := range plan.Transfers {
					seen = seen || tr.From == i-1 && tr.To == i
				}
				if !seen {
					t.Errorf("%s: no transfer between %s and %s", name, plan.Parts[i-1].Part, pp.Part)
				}
			}
		}
		t.Logf("%s: %d parts, %d transfers", name, len(plan.Parts), len(plan.Transfers))
	}
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
