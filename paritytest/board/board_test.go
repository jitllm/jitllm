package board

import (
	"flag"
	"fmt"
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
	lg, err := s.Prefill(m.Vocab.Encode("Hello there", true))
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

var boardDump = flag.String("board-dump", "", "TestMainDecodeKernels writes out every op of main's kernels with this name prefix")

func dump(t *testing.T, k *ir.Kernel) {
	for i, o := range k.Ops {
		t.Logf("  %%%d = %s.%s %v imm %d", i+1, o.Kind, o.Type, o.Args, o.Imm)
	}
}

// composed is every kernel the composer forms for the model's blocks at one
// row, and each block's error.
func composed(t *testing.T, name string, tg jit.Target) ([]jit.Kernel, []error) {
	t.Helper()
	path := testmodels.Path(name + ".jlm")
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, err := jlm.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var ks []jit.Kernel
	var errs []error
	for li := 0; li < m.Blocks(); li++ {
		g := m.BlockGraph(li)
		if g == nil {
			continue
		}
		formats := map[jlm.Role]jlm.Type{}
		for _, o := range g.Ops {
			for _, w := range o.Weight {
				if e, ok := f.Find(w.Role, int32(li), -1); ok {
					formats[w.Role] = e.Type
				}
			}
		}
		k, err := jit.Compose(jit.Block{Graph: g, Rows: 1, Formats: formats}, tg, jit.ProfileFor(tg, 1))
		ks = append(ks, k...)
		if err != nil {
			errs = append(errs, fmt.Errorf("block %d: %w", li, err))
		}
	}
	return ks, errs
}

func TestBoard(t *testing.T) {
	tg := jit.Target{Device: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	main := mainDecode(t, *boardModel, *boardProfile)
	ks, errs := composed(t, *boardModel, tg)
	for _, err := range errs {
		t.Log(err)
	}
	var mine []*ir.Kernel
	for _, k := range ks {
		mine = append(mine, paritytest.Old(k.IR))
		t.Logf("composed %-24s %4d ops passes %v", k.IR.Name, len(k.IR.Ops), k.Passes)
	}
	total, same, miss := paritytest.Compare(main, mine)
	byKernel := map[string]int{}
	for _, m := range miss {
		byKernel[m.K.Name]++
	}
	for _, k := range main {
		if n := byKernel[k.Name]; n > 0 {
			t.Logf("missing %4d of %-28s", n, k.Name)
			byKernel[k.Name] = 0
		}
	}
	t.Logf("%s on %s: %d of main's %d instructions (%.1f%%), %d blocks with ops not formed",
		*boardModel, *boardProfile, same, total, 100*float64(same)/float64(max(total, 1)), len(errs))
}
