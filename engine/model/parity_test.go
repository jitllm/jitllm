package model

import (
	"cmp"
	"flag"
	"fmt"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/paritytest"
	"github.com/jitllm/jitllm/paritytest/maintier"
)

var (
	parityProfiles = flag.String("parity-profiles", "cuda-sm70",
		"the recorder profiles TestComposerParity compares on")
	parityModel    = flag.String("parity-model", "", "TestComposerParity on this fixture alone, printing what it misses")
	parityByShape  = flag.Bool("parity-by-shape", false, "TestComposerParity groups what is missing by main's kernel name with its sizes elided")
	parityExplain  = flag.Bool("parity-explain", false, "with -parity-kernel, every composed instruction written out")
	parityPerModel = flag.Bool("parity-per-model", false, "TestComposerParity logs each model's instructions matched")
	parityKernel   = flag.String("parity-kernel", "", "the main kernel name prefix to print: with -parity-model what it misses, else every model's kernels of that name")
)

// TestComposerParity holds what the composer emits to what main emits, with
// no GPU: every fixture opened twice on a device that runs nothing and
// records what is compiled for it (recorder) -- once on main's tier
// (paritytest/maintier, main's own code), once on the composer's -- a prompt
// chunk and a decode step asked of each, and every instruction of main's
// kernels looked for among the composer's formed kernels by its dataflow
// (paritytest.Compare): what it reads, computes and writes, so the order of
// independent instructions and the form of index arithmetic do not count. It reports per profile the share
// of main's instructions matched, the models whose every instruction is, and
// main's kernels by instructions missing. A measurement, not a gate.
func TestComposerParity(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	if len(paths) == 0 {
		t.Fatalf("no synth-*.jlm under %s (set JITLLM_MODELS)", testmodels.Dir())
	}
	for _, pn := range strings.Split(*parityProfiles, ",") {
		prof, ok := recorder.Profiles[pn]
		if !ok {
			t.Fatalf("no recorder profile %q", pn)
		}
		var total, same, models, identical int
		missing := map[string]int{}
		// Each model's pair of tiers is its own: they run in parallel, a
		// worker a core, and their results are summed in model order.
		type result struct {
			name     string
			mt, ms   int
			miss     []paritytest.Miss
			main     []*ir.Kernel
			composed []*ir.Kernel
			mainErr  error
			compErr  error
			hasMain  bool
		}
		var todo []string
		for _, p := range paths {
			name := strings.TrimSuffix(filepath.Base(p), ".jlm")
			if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
				continue
			}
			if *parityModel != "" && name != *parityModel {
				continue
			}
			todo = append(todo, p)
		}
		results := make([]result, len(todo))
		work := make(chan int)
		var wg sync.WaitGroup
		for w := 0; w < runtime.GOMAXPROCS(0); w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range work {
					p := todo[i]
					r := &results[i]
					r.name = strings.TrimSuffix(filepath.Base(p), ".jlm")
					mk, err := parityKernels(t, prof, p, true)
					r.mainErr = err
					if len(mk) == 0 {
						continue
					}
					r.hasMain = true
					ck, err := parityKernels(t, prof, p, false)
					r.compErr = err
					r.main, r.composed = mk, composedOnly(ck)
					r.mt, r.ms, r.miss = paritytest.Compare(r.main, r.composed)
				}
			}()
		}
		for i := range todo {
			work <- i
		}
		close(work)
		wg.Wait()
		ex := paritytest.NewInterner()
		for _, r := range results {
			if r.mainErr != nil {
				t.Logf("%s on %s, main: %v", r.name, pn, r.mainErr)
			}
			if !r.hasMain {
				continue
			}
			if r.compErr != nil && *parityModel != "" {
				t.Logf("%s on %s, composer: %v", r.name, pn, r.compErr)
			}
			if *parityPerModel {
				t.Logf("model %s %d %d", r.name, r.ms, r.mt)
			}
			if *parityModel == "" && *parityKernel != "" {
				for _, k := range r.main {
					if strings.HasPrefix(k.Name, *parityKernel) {
						t.Logf("%s main %s", r.name, shapeOf(k))
					}
				}
			}
			if *parityModel != "" && *parityKernel != "" {
				for _, k := range r.main {
					if strings.HasPrefix(k.Name, *parityKernel) {
						t.Logf("main     %s", shapeOf(k))
					}
				}
				for _, k := range r.composed {
					t.Logf("composed %s", shapeOf(k))
					if *parityExplain {
						for i := range k.Ops {
							t.Logf("    %%%d %s", i, paritytest.Explain(ex, k, i, 4))
						}
					}
				}
			}
			for _, m := range r.miss {
				prefix, _, _ := strings.Cut(m.K.Name, "_")
				if *parityByShape {
					prefix = digits.ReplaceAllString(m.K.Name, "N")
				}
				missing[prefix]++
				if *parityModel != "" && *parityKernel != "" && strings.HasPrefix(m.K.Name, *parityKernel) {
					t.Logf("%s missing %s\n        %s", m.K.Name, opLine(m.K, m.Op), paritytest.Explain(ex, m.K, m.Op, 4))
				}
			}
			total, same, models = total+r.mt, same+r.ms, models+1
			if r.ms == r.mt {
				identical++
			}
		}
		t.Logf("%-16s main's instructions %d, the composer's dataflow-identical %d (%.1f%%), models identical %d/%d",
			pn, total, same, 100*float64(same)/float64(max(total, 1)), identical, models)
		var names []string
		for n := range missing {
			names = append(names, n)
		}
		slices.SortFunc(names, func(a, b string) int { return cmp.Compare(missing[b], missing[a]) })
		for _, n := range names[:min(10, len(names))] {
			t.Logf("    %-22s %9d missing", n, missing[n])
		}
	}
}

// parityKernels is every kernel compiled for the model at path on a
// recording device standing in for prof, on main's tier or the composer's,
// over a prompt chunk and a decode step: main's launched, the composer's
// formed. Launches run nothing.
func parityKernels(t *testing.T, prof recorder.Profile, path string, main bool) ([]*ir.Kernel, error) {
	m, err := Open(path, noTune)
	if err != nil {
		return nil, err
	}
	defer m.Close()
	if m.IsEncoder() {
		return nil, nil
	}
	rec := recorder.New(prof, 1<<40)
	var g nn.Device
	if main {
		// Tuning off on both sides: a tuner on a device that runs nothing
		// times no-ops, and launches every candidate it times.
		mg, err := maintier.New([]maintier.Slot{{Dev: rec, Bytes: 1 << 40}}, maintier.WithDeviceTune(maintier.TuneOff),
			maintier.WithConfig(func(c *maintier.Config) { c.NoGraph = true }))
		if err != nil {
			return nil, err
		}
		defer mg.Close()
		g = mg
	} else {
		cg, err := tier.New([]tier.Slot{{Dev: rec, Bytes: 1 << 40}}, append(testTierOpts(t), tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))...)
		if err != nil {
			return nil, err
		}
		defer cg.Close()
		g = cg
	}
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		return nil, err
	}
	if !main && *parityPerModel {
		defer func() {
			if st := g.(*tier.GPU).Stats(); st.LastErr != "" || st.DeclineWhy != "" {
				t.Logf("%s composer: last %q, declined %q", filepath.Base(path), st.LastErr, st.DeclineWhy)
			}
		}()
	}
	lg, err := s.Prefill(m.Vocab.Encode("Hello there", true))
	if !main && *parityPerModel {
		if st := g.(*tier.GPU).Stats(); st.LastErr != "" {
			t.Logf("%s composer after the prefill: %q", filepath.Base(path), st.LastErr)
		}
	}
	tok := int32(1)
	if err == nil {
		tok = Greedy(lg)
	}
	if _, ferr := s.Forward(tok); err == nil {
		err = ferr
	}
	// Main's kernels are what it launched (its tuners compile alternatives
	// they never run); the composer's what it formed, launched or not (a
	// step stops at the first op it cannot form).
	var ks []*ir.Kernel
	if main {
		for _, e := range rec.Launched() {
			ks = append(ks, e.K)
		}
		return ks, err
	}
	for _, e := range rec.Compiled(func(k *ir.Kernel) bool { return strings.HasPrefix(k.Comment, "composed") }) {
		ks = append(ks, e.K)
	}
	return ks, err
}

var digits = regexp.MustCompile(`[0-9]+`)

// shapeOf is a kernel's schedule at a glance: its op count, its f32
// accumulators (the f32 phis: rows times tokens a thread), its matrix
// instructions, its loops' trips and its grid clamp (the first min's
// constant).
func shapeOf(k *ir.Kernel) string {
	acc, mma, clamp := 0, 0, int64(-1)
	var trips []int64
	for _, o := range k.Ops {
		switch {
		case o.Kind == ir.OpMMA:
			mma++
		case o.Kind == ir.OpPhi && o.Type == ir.F32:
			acc++
		case o.Kind == ir.OpLoop:
			trips = append(trips, o.Imm)
		case o.Kind == ir.OpMin && clamp < 0:
			for _, a := range o.Args[:2] {
				if a != 0 && k.Ops[a-1].Kind == ir.OpConst {
					clamp = k.Ops[a-1].Imm
				}
			}
		}
	}
	return fmt.Sprintf("%s: %d ops, %d f32 accumulators, %d mma, loops %v, grid clamp %d", k.Name, len(k.Ops), acc, mma, trips, clamp)
}

// composedOnly is the kernels the composer formed (genKern marks them).
func composedOnly(ks []*ir.Kernel) []*ir.Kernel {
	var out []*ir.Kernel
	for _, k := range ks {
		if strings.HasPrefix(k.Comment, "composed") {
			out = append(out, k)
		}
	}
	return out
}

// opLine is instruction i of k as recorder.Text prints it.
func opLine(k *ir.Kernel, i int) string {
	lines := strings.Split(recorder.Text(k), "\n")
	pre := fmt.Sprintf("%%%d = ", i)
	for _, l := range lines {
		if strings.HasPrefix(l, pre) {
			return l
		}
	}
	return "?"
}
