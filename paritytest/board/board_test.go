package board

import (
	"flag"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit_old/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest"
	"github.com/jitllm/jitllm/paritytest/maintier"
)

const prompt = "Hello there"

var (
	boardModel   = flag.String("board-model", "synth-ernie45", "the fixture the board runs")
	boardProfile = flag.String("board-profile", "cuda-sm70", "the recorder profile main's kernels are taken on")
)

// mainDecode is every kernel main's tier launched, live, for one decode step
// after a short prompt.
func mainDecode(t *testing.T, name, profile string) []*ir.Kernel {
	t.Helper()
	path := testmodels.Path(name + ".jlm")
	prof, ok := recorder.Profiles[profile]
	if !ok {
		t.Fatalf("no recorder profile %q", profile)
	}
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rec := recorder.New(prof, 1<<40)
	dev, err := maintier.New([]maintier.Slot{{Dev: rec, Bytes: 1 << 40}}, maintier.WithDeviceTune(maintier.TuneOff),
		maintier.WithConfig(func(c *maintier.Config) { c.NoGraph = true }))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(dev); err != nil {
		t.Fatal(err)
	}
	lg, err := s.Prefill(m.Vocab.Encode(prompt, true))
	if err != nil {
		t.Fatal(err)
	}
	mark := rec.Mark()
	if _, err := s.Forward(model.Greedy(lg)); err != nil {
		t.Fatal(err)
	}
	var ks []*ir.Kernel
	for _, e := range rec.LiveLaunchedFrom(mark) {
		ks = append(ks, e.K)
	}
	if len(ks) == 0 {
		t.Fatalf("%s on %s: main launched nothing for the step", name, profile)
	}
	return ks
}

func TestMainDecodeKernels(t *testing.T) {
	for _, k := range mainDecode(t, *boardModel, *boardProfile) {
		t.Log(describe(k))
		if *boardDump != "" && strings.HasPrefix(k.Name, *boardDump) {
			dump(t, k)
		}
	}
}

func describe(k *ir.Kernel) string {
	loops := 0
	for _, o := range k.Ops {
		if o.Kind == ir.OpLoop {
			loops++
		}
	}
	return fmt.Sprintf("%-28s %4d ops %2d loops group %v params %d lanes %d", k.Name, len(k.Ops), loops, k.Group, len(k.Params), k.Lanes)
}

func TestBlockGraph(t *testing.T) {
	m, err := model.Open(testmodels.Path(*boardModel+".jlm"), model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Log(m.BlockGraph(0).String())
}

var (
	boardDump    = flag.String("board-dump", "", "TestMainDecodeKernels writes out every op of main's kernels with this name prefix")
	boardExplain = flag.String("board-explain", "", "TestBoard writes out what main's kernels with this name prefix miss")
	boardMine    = flag.String("board-mine", "", "TestBoard writes out every op of the composed kernels with this name prefix")
	boardDepth   = flag.Int("board-depth", 3, "how deep -board-explain and -board-mine write an instruction's operands")
)

func dump(t *testing.T, k *ir.Kernel) {
	for i, o := range k.Ops {
		t.Logf("  %%%d = %s.%s %v imm %d", i+1, o.Kind, o.Type, o.Args, o.Imm)
	}
}

// keys is the history the board's decode step attends: the prompt and the
// token decoded.
func keys(t *testing.T, name string) int {
	t.Helper()
	m, err := model.Open(testmodels.Path(name+".jlm"), model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	return len(m.Vocab.Encode(prompt, true)) + 1
}

// composed is the composer's plan for the model with every part on d, and
// what it could not form.
func composed(t *testing.T, name string, d jit.Device) (*jit.Plan, error) {
	t.Helper()
	f, err := jlm.Open(testmodels.Path(name + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	step := jit.Step{Rows: 1, Page: 64, Keys: keys(t, name)}
	return jit.Default().Compose(f, step, func(jit.Part) jit.Device { return d })
}

var sm70 = jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}

func TestBoard(t *testing.T) {
	main := mainDecode(t, *boardModel, *boardProfile)
	plan, err := composed(t, *boardModel, sm70)
	if err != nil {
		t.Logf("not formed: %v", err)
	}
	var mine []*ir.Kernel
	for _, pp := range plan.Parts {
		for _, c := range pp.Codes {
			if c.GPU == nil {
				continue
			}
			old := paritytest.Old(c.GPU)
			mine = append(mine, old)
			if *boardMine != "" && strings.HasPrefix(c.Name, *boardMine) {
				in := paritytest.NewInterner()
				for i := range old.Ops {
					t.Logf("  mine %%%d %s", i+1, paritytest.Explain(in, old, i, *boardDepth))
				}
			}
			t.Logf("%-8s %-24s %4d ops passes %v", pp.Part, c.Name, len(c.GPU.Ops), c.Passes)
		}
	}
	total, same, miss := paritytest.Compare(main, mine)
	byKernel := map[string]int{}
	in := paritytest.NewInterner()
	for _, m := range miss {
		byKernel[m.K.Name]++
		if *boardExplain != "" && strings.HasPrefix(m.K.Name, *boardExplain) {
			t.Logf("%s misses %%%d %s", m.K.Name, m.Op+1, paritytest.Explain(in, m.K, m.Op, *boardDepth))
		}
	}
	if *boardExplain != "" {
		fin := paritytest.NewInterner()
		for _, m := range paritytest.Frontier(main, mine) {
			if strings.HasPrefix(m.K.Name, *boardExplain) {
				t.Logf("%s parts at %%%d %s", m.K.Name, m.Op+1, paritytest.Explain(fin, m.K, m.Op, *boardDepth))
			}
		}
	}
	identical := 0
	type row struct {
		name string
		n    int
	}
	var rows []row
	seen := map[string]bool{}
	for _, k := range main {
		if seen[k.Name] {
			continue
		}
		seen[k.Name] = true
		if n := byKernel[k.Name]; n > 0 {
			rows = append(rows, row{k.Name, n})
		} else {
			identical++
		}
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].n > rows[j].n })
	for i, r := range rows {
		if i == 10 {
			break
		}
		t.Logf("missing %4d of %s", r.n, r.name)
	}
	whole := 0
	for _, k := range main {
		if !seen[k.Name] {
			continue
		}
		seen[k.Name] = false
		kt, _, _ := paritytest.Compare([]*ir.Kernel{k}, nil)
		found := false
		for _, m := range mine {
			if _, s, _ := paritytest.Compare([]*ir.Kernel{k}, []*ir.Kernel{m}); s == kt {
				found = true
				break
			}
		}
		if found {
			whole++
		} else {
			t.Logf("no one kernel holds all of %s", k.Name)
		}
	}
	din := paritytest.NewInterner()
	keyOf := func(k *ir.Kernel) string { return fmt.Sprint(din.Dataflow(paritytest.Simplified([]*ir.Kernel{k})[0])) }
	distinct := func(ks []*ir.Kernel) int {
		keys := map[string]bool{}
		for _, k := range ks {
			keys[keyOf(k)] = true
		}
		return len(keys)
	}
	t.Logf("%s on %s: %d of main's %d instructions (%.1f%%), %d of main's %d kernels identical, %d held whole by one kernel; %d distinct kernels against main's %d",
		*boardModel, *boardProfile, same, total, 100*float64(same)/float64(max(total, 1)), identical, len(seen), whole, distinct(mine), distinct(main))
}

func TestAttendAttrs(t *testing.T) {
	m, err := model.Open(testmodels.Path(*boardModel+".jlm"), model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, o := range m.BlockGraph(0).Ops {
		t.Logf("%s in %v out %v weights %v attr %+v", o.Kind, o.In, o.Out, o.Weight, o.Attr)
	}
}
