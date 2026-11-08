package engine

import (
	"errors"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestEveryModelRunsGenerated drives every model available through all three
// transcriptions of the graph -- Forward, Prefill and ForwardBatch -- and an
// encoder through its embedder. The assertion is that they run: a shape with
// no kernel is an error, so a model that completes them ran generated code end
// to end. Kernel gaps are keyed on shapes, which is why every model is swept.
//
// "Every model" is the model directory's GGUFs, the GGUFs one directory down
// (granite/, qwen35/, moefam/, embed/ ...), and the safetensors fixtures the
// transformers goldens are taken on -- the only models available for deepseek2,
// kimilinear and the HF mixtures.
//
// Each model is a subtest of its own, so one file cannot decide the sweep's
// fate: an architecture outside RULE 7's list (a fixture another tree is adding,
// say) is a refusal NAMED in its own subtest, and every other model still runs.
// A model the converter accepts and the engine then cannot open is a FAILURE,
// not a silent pass. The sweep itself never reports SKIP: it fails when no model
// ran.
func TestEveryModelRunsGenerated(t *testing.T) { everyModelRunsGenerated(t) }

// TestEveryModelRunsOnACard is the card half of every model running: each
// model the sweep converts places every block on the card the test's
// devices select and decodes there; a model whose blocks the card does not
// all take fails by name, with what the card declined.
func TestEveryModelRunsOnACard(t *testing.T) {
	g, err := openGPU(t)
	if err != nil || g == nil {
		t.Fatalf("no card to run the models on: %v", err)
	}
	defer g.Close()
	slow, only := os.Getenv("JITLLM_SLOW") != "", os.Getenv("JITLLM_SLOW") == "only"
	pick := os.Getenv("JITLLM_AUDIT_MODEL")
	ran, last, held := 0, "", [2]uint64{}
	defer func() { heldAfter(t, g, last, &held) }()
	for _, src := range auditModels(t) {
		if pick != "" && !strings.Contains(src.path, pick) {
			continue
		}
		if pick == "" && (!slow && src.size > 2<<30 || only && src.size <= 2<<30) {
			continue
		}
		ran++
		heldAfter(t, g, last, &held)
		last = src.name
		t.Run(src.name, func(t *testing.T) {
			path := src.path
			if src.hf {
				path = jlmOfSafetensors(t, src.path)
			} else {
				p, err := convertPair(src.path, src.mmproj)
				if errors.Is(err, convert.ErrNotImplemented) {
					t.Skipf("NOT IMPLEMENTED (RULE 7 scope): %v", err)
				}
				if errors.Is(err, convert.ErrPart) {
					t.Skipf("PART OF A MODEL: %v", err)
				}
				if err != nil {
					t.Fatalf("convert: %v", err)
				}
				path = p
			}
			m, err := model.Open(path)
			if err != nil {
				t.Fatalf("the converter accepted it and Open refused it: %v", err)
			}
			defer m.Close()
			if m.IsEmbedding() {
				// An embedding model runs through its embedder, every block on
				// the card.
				e, err := NewEmbedder(m, WithKVF16(false))
				if err != nil {
					t.Fatal(err)
				}
				defer e.Close()
				if err := e.SetDeviceLayers(g, -1); err != nil {
					t.Fatal(err)
				}
				if n := e.DeviceBlocks(); n != m.Cfg.NLayer {
					t.Fatalf("the card took %d of %d blocks: %v (%s)", n, m.Cfg.NLayer, e.DeviceDeclines(), g.Err())
				}
				formed, before := g.Stats().UnformedAtStep, cardDeclines(g)
				for i := range 3 {
					v, err := e.Embed([]int32{1, int32(2 + i), 3})
					if err != nil {
						t.Fatalf("Embed on the card: %v (%s)", err, g.Err())
					}
					finite(t, "Embed on the card", v)
				}
				if n := e.DeviceBlocks(); n != m.Cfg.NLayer {
					t.Fatalf("after the steps the card holds %d of %d blocks: %s", n, m.Cfg.NLayer, g.Err())
				}
				if s := g.Stats(); s.UnformedAtStep != formed {
					t.Fatalf("the card formed %d part(s) at a step after placement (by part %v): %s",
						s.UnformedAtStep-formed, s.UnformedAtStepParts, g.Err())
				}
				refusedAtStep(t, g, before)
				return
			}
			st := cardInstance(t, m, g)
			defer st.Close()
			if m.Cfg.Head != nil {
				formed, before := g.Stats().UnformedAtStep, cardDeclines(g)
				for i := range 3 {
					finite(t, "MarkScores on the card", markerScores(t, st, i))
				}
				stayedOnCard(t, st, m, g)
				if s := g.Stats(); s.UnformedAtStep != formed {
					t.Fatalf("the card formed %d part(s) at a step after placement (by part %v): %s",
						s.UnformedAtStep-formed, s.UnformedAtStepParts, g.Err())
				}
				refusedAtStep(t, g, before)
				return
			}
			var img image.Image
			if m.Tower != nil {
				img = auditPicture(t)
				// A chain's blocks run as parts over their own grids, each on
				// the device chainOn names (nil: the CPU device).
				n := st.tower.s.GPULayers()
				if m.Tower.Chain {
					n = 0
					for _, d := range st.tower.chainOn {
						if d != nil {
							n++
						}
					}
				}
				if n != len(m.Tower.Blocks) {
					t.Fatalf("the card took %d of the tower's %d blocks: %v (%s)", n, len(m.Tower.Blocks), st.tower.s.DeviceDeclines(), g.Err())
				}
			}
			formed, before := g.Stats().UnformedAtStep, cardDeclines(g)
			for i := range 3 {
				lg, err := st.Forward(int32(1 + i))
				if err != nil {
					t.Fatalf("Forward on the card: %v (%s)", err, g.Err())
				}
				finite(t, "Forward on the card", lg)
			}
			if img != nil {
				rows, err := st.EncodePicture(img)
				if err != nil {
					t.Fatalf("EncodePicture on the card: %v (%s)", err, g.Err())
				}
				finite(t, "EncodePicture on the card", rows)
			}
			stayedOnCard(t, st, m, g)
			if s := g.Stats(); s.UnformedAtStep != formed {
				t.Fatalf("the card formed %d part(s) at a step after placement (by part %v): %s",
					s.UnformedAtStep-formed, s.UnformedAtStepParts, g.Err())
			}
			refusedAtStep(t, g, before)
		})
	}
	if ran == 0 {
		t.Fatalf("no model ran on the card (JITLLM_AUDIT_MODEL %q) -- this gate proved nothing", pick)
	}
}

// heldAfter fails t where card g holds more after model name's subtest closed
// its instances and the model than it held before it, the bytes charged or
// allocated: one card serves every model in turn, and what one leaves the
// next finds (an input table once stayed resident under the entry the next
// model's took). held is what the card held before.
func heldAfter(t *testing.T, g *gpu.GPU, name string, held *[2]uint64) {
	t.Helper()
	s := g.Stats()
	now := [2]uint64{uint64(s.BudgetUsed), uint64(s.Allocated)}
	if name != "" && (now[0] > held[0] || now[1] > held[1]) {
		t.Errorf("after %s the card holds %d bytes charged and %d allocated, %d and %d before it", name, now[0], now[1], held[0], held[1])
	}
	*held = now
}

// stayedOnCard fails t where the card no longer holds every block after the
// steps: a device failure moves them to the CPU (demoteAll), and the steps'
// answers then say nothing about the card.
func stayedOnCard(t *testing.T, st *Instance, m *model.Model, g *gpu.GPU) {
	t.Helper()
	if n := st.GPULayers(); n != m.Cfg.NLayer {
		t.Fatalf("after the steps the card holds %d of %d blocks: %s", n, m.Cfg.NLayer, g.Err())
	}
}

// cardDeclines is the steps card g declined so far, its blocks' and its
// table, sample and input rows', by reason.
func cardDeclines(g *gpu.GPU) [nn.NumDeclines]int {
	s := g.Stats()
	b, o := s.BlockDeclines(), s.StepDeclines()
	for i := range b {
		b[i] += o[i]
	}
	return b
}

// refusedAtStep fails t on every step card g declined since before for a
// reason that is not a hardware limit: a part placed on the card runs there.
func refusedAtStep(t *testing.T, g *gpu.GPU, before [nn.NumDeclines]int) {
	t.Helper()
	for _, d := range stepDeclines(before, cardDeclines(g)) {
		t.Errorf("the card declined %s", d)
	}
}

// stepDeclines names the steps declined between two counts for a reason that
// is not a hardware limit: the device not serving the part, holding no room
// for it, or lacking the ISA.
func stepDeclines(before, after [nn.NumDeclines]int) []string {
	var out []string
	for i := range after {
		switch nn.Decline(i) {
		case nn.DeclineServes, nn.DeclineNoRoom, nn.DeclineISA:
			continue
		}
		if n := after[i] - before[i]; n > 0 {
			out = append(out, fmt.Sprintf("%d step(s): %s", n, nn.DeclineName(i)))
		}
	}
	return out
}

// TestTheCensusSeesADeclinedStep runs the census's decline check against its
// violations: a step declined as not formed, or for a table it lacked, is
// named; the hardware limits, and declines from before the decode, are not.
func TestTheCensusSeesADeclinedStep(t *testing.T) {
	var before, after [nn.NumDeclines]int
	before[nn.DeclineNotFormed] = 4
	after[nn.DeclineNotFormed] = 4
	after[nn.DeclineServes], after[nn.DeclineNoRoom], after[nn.DeclineISA] = 2, 3, 1
	if got := stepDeclines(before, after); len(got) != 0 {
		t.Fatalf("hardware limits and earlier declines were named: %v", got)
	}
	after[nn.DeclineNotFormed]++
	after[nn.DeclineNoTable] = 2
	if got := stepDeclines(before, after); len(got) != 2 {
		t.Fatalf("a step declined as not formed and two for a table gave %v, want two findings", got)
	}
}

// cardInstance is an instance of m with every block on card g: resident
// where the card holds them all, and where it has room for only some, an
// instance listing the card alone, on which the model pages and no block
// runs on the CPU. Any block off the card fails the test, with what the card
// declined.
func cardInstance(t *testing.T, m *model.Model, g *gpu.GPU) *Instance {
	t.Helper()
	st := testInstance(t, m, 16)
	if err := st.SetDeviceLayers(g, -1); err != nil {
		st.Close()
		t.Fatalf("placement: %v (%s)", err, g.Err())
	}
	if st.GPULayers() == m.Cfg.NLayer {
		return st
	}
	declined := st.DeviceDeclines()
	st.Close()
	spec := testdevices.One("cuda:0")
	card := nn.DeviceName(spec)
	st = testInstance(t, m, 16, WithDevices(spec))
	if err := st.SetDevice(g); err != nil {
		st.Close()
		t.Fatalf("the card listed alone: %v (%s); declined with the CPU: %v", err, g.Err(), declined)
	}
	// A paged block not faulted in yet has no holder to name; none is the
	// CPU's.
	for li := range m.Cfg.NLayer {
		if st.WhereIs(li) == nn.CPU {
			st.Close()
			t.Fatalf("the card %s listed alone: block %d runs on the CPU; declined with the CPU: %v", card, li, declined)
		}
	}
	t.Logf("%d blocks paged on %s, which had room for some; declined with the CPU: %v", m.Cfg.NLayer, card, declined)
	return st
}

// everyModelRunsGenerated is TestEveryModelRunsGenerated with opts on every model it opens.
func everyModelRunsGenerated(t *testing.T, opts ...Option) {
	// JITLLM_AUDIT_MODEL=<substring> audits one file whatever its size (run it
	// under `scripts/cap 24G` for anything over ~16 GB); JITLLM_SLOW=1 admits
	// every model over 2 GiB, and JITLLM_SLOW=only sweeps those alone.
	pick := os.Getenv("JITLLM_AUDIT_MODEL")
	slow, only := os.Getenv("JITLLM_SLOW") != "", os.Getenv("JITLLM_SLOW") == "only"
	var ran, refused, large []string
	for _, src := range auditModels(t) {
		name := src.name
		if pick != "" && !strings.Contains(src.path, pick) {
			continue
		}
		if pick == "" && only && src.size <= 2<<30 {
			continue
		}
		if pick == "" && !slow && src.size > 2<<30 {
			large = append(large, name)
			continue
		}
		var why error
		done := false
		t.Run(name, func(t *testing.T) {
			path := src.path
			if src.hf {
				path = jlmOfSafetensors(t, src.path)
			} else {
				p, err := convertPair(src.path, src.mmproj)
				if errors.Is(err, convert.ErrNotImplemented) {
					why = err
					t.Skipf("NOT IMPLEMENTED (RULE 7 scope): %v", err)
				}
				if errors.Is(err, convert.ErrPart) {
					t.Skipf("PART OF A MODEL: %v", err)
				}
				if err != nil {
					t.Fatalf("convert: %v", err)
				}
				path = p
			}
			instOpts := opts
			m, err := model.Open(path)
			if err != nil {
				t.Fatalf("the converter accepted it and Open refused it: %v", err)
			}
			defer m.Close()
			auditRun(t, m, instOpts...)
			done = true
			t.Run("composed-cpu", func(t *testing.T) { auditComposedCPU(t, name, path, opts...) })
		})
		switch {
		case done:
			ran = append(ran, name)
		case why != nil:
			refused = append(refused, name)
		}
	}
	if len(large) > 0 {
		t.Logf("over 2 GiB, not swept (JITLLM_SLOW=1 admits them): %s", strings.Join(large, ", "))
	}
	if len(refused) > 0 {
		t.Logf("outside RULE 7's list, refused by the converter: %s", strings.Join(refused, ", "))
	}
	if len(ran) == 0 {
		t.Fatal("no model ran on generated code -- this gate proved nothing")
	}
	t.Logf("%d models ran on generated code", len(ran))
}

// markerScores is a marker head's scores for question i of three (its type
// i, its markers rows 1, 3 and 5 of eight).
func markerScores(t *testing.T, s *Instance, i int) []float32 {
	t.Helper()
	out := make([]float32, 3)
	if err := s.MarkScores([]int32{1, int32(5 + i), 9, 13, 17, 21, 25, 29}, i, []int{1, 3, 5}, out); err != nil {
		t.Fatalf("MarkScores: %v", err)
	}
	return out
}

// auditRun is the gate's body for one opened model: the three decoder entry
// points, and its embedder too when the container carries pooling.
func auditRun(t *testing.T, m *model.Model, io ...Option) {
	t.Helper()
	ids := []int32{1, 2, 3, 4, 5}
	if m.IsEmbedding() {
		e, err := NewEmbedder(m, io...)
		if err != nil {
			t.Fatalf("NewEmbedder: %v", err)
		}
		v, err := e.Embed(ids)
		e.Close()
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		finite(t, "Embed", v)
		// An embedding model has no head to decode with: it runs as it is
		// read, through its embedder.
		return
	}
	if m.Cfg.Head != nil {
		// A marker head reads out through its scorer: it answers questions.
		s := testInstance(t, m, 16, io...)
		finite(t, "MarkScores", markerScores(t, s, 0))
		s.Close()
		return
	}
	s := testInstance(t, m, 16, io...)
	logits, err := s.Forward(1)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	finite(t, "Forward", logits)
	if _, err := s.Prefill(ids); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	s.Close()
	if m.Tower != nil {
		ps := testInstance(t, m, 256, io...)
		rows, err := ps.EncodePicture(auditPicture(t))
		if err != nil {
			t.Fatalf("EncodePicture: %v", err)
		}
		finite(t, "EncodePicture", rows)
		ps.Close()
	}
	bs := newSessBatch(m, 2, 16, io...)
	if _, err := bs.ForwardBatch([]int32{1, 2}); err != nil {
		t.Fatalf("ForwardBatch: %v", err)
	}
	bs.Close()
}

// auditComposedCPU is the gate's composed CPU arm: every entry point again
// with the CPU's blocks on what the composer forms for it, and a decode whose
// every block step is counted, served or declined by name -- all of them
// served on the models the composer serves on the CPU (cpuServedModels).
func auditComposedCPU(t *testing.T, name, path string, opts ...Option) {
	instOpts := opts
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	auditRun(t, m, instOpts...)
	if m.Cfg.Head != nil {
		return
	}
	const steps = 3
	if m.IsEmbedding() {
		e, err := NewEmbedder(m, instOpts...)
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close()
		asked := e.st.Account(nn.CPU).Asks
		for i := range steps {
			v, err := e.Embed([]int32{1, int32(2 + i), 3})
			if err != nil {
				t.Fatalf("Embed: %v", err)
			}
			finite(t, "Embed", v)
		}
		if n := e.st.Account(nn.CPU).Asks - asked; n != 0 {
			t.Fatalf("the embeddings asked the composer for %d form(s) after the instance was made", n)
		}
		served, declined := e.st.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
		for i, n := range declined {
			if n > 0 {
				t.Fatalf("%d block steps declined on the composed CPU: %s (%v)", n, nn.DeclineName(i), e.st.Account(nn.CPU).FormErr)
			}
		}
		if served == 0 {
			t.Fatal("the embeddings served no block step on the composed CPU")
		}
		return
	}
	s := testInstance(t, m, 16, instOpts...)
	defer s.Close()
	asked := s.Account(nn.CPU).Asks
	for i := range steps {
		lg, err := s.Forward(int32(1 + i))
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		finite(t, "Forward", lg)
	}
	if n := s.Account(nn.CPU).Asks - asked; n != 0 {
		t.Fatalf("the decode asked the composer for %d form(s) after the instance was made", n)
	}
	served, declined := s.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
	total := served
	var named []string
	for i, n := range declined {
		total += n
		if n > 0 {
			named = append(named, fmt.Sprintf("%d %s", n, nn.DeclineName(i)))
		}
	}
	if want := steps * m.Cfg.NLayer; total != want {
		t.Fatalf("the census counts %d block steps on the CPU, where the decode ran %d", total, want)
	}
	t.Logf("composed CPU: %d of %d block steps served; declined: %s (%v)", served, total, strings.Join(named, ", "), s.Account(nn.CPU).FormErr)
	if slices.Contains(cpuServedModels(), name) && served != total {
		t.Fatalf("a model the composer serves on the CPU declined %d block steps", total-served)
	}
}

func finite(t *testing.T, what string, v []float32) {
	t.Helper()
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("%s: element %d is %v", what, i, x)
		}
	}
}

type auditModel struct {
	name, path string
	// mmproj is the catalog's projector for a vision model, converted with
	// it into one container.
	mmproj string
	size   int64
	hf     bool // a safetensors directory
}

// libraryPath is where the model directory holds a catalog file: at its top,
// or a directory or two down; "" where it holds none.
func libraryPath(file string) string {
	if p := testmodels.Path(file); fileExists(p) {
		return p
	}
	for _, pat := range []string{filepath.Join("*", file), filepath.Join("*", "*", file)} {
		if ps, _ := filepath.Glob(filepath.Join(testmodels.Dir(), pat)); len(ps) > 0 {
			sort.Strings(ps)
			return ps[0]
		}
	}
	return ""
}

// laterShard is the second and later file of a split GGUF; the first names the
// whole model.
func laterShard() *regexp.Regexp {
	return regexp.MustCompile(`-0*([2-9]|[1-9][0-9]+)-of-[0-9]+\.gguf$`)
}

// auditModels is everything TestEveryModelRunsGenerated sweeps. On a host
// holding only containers it is the containers, as languageModels' fallback
// is.
func auditModels(t *testing.T) []auditModel {
	t.Helper()
	var out []auditModel
	add := func(name, p string, hf bool) {
		var size int64
		if hf {
			fs, _ := filepath.Glob(filepath.Join(p, "*.safetensors"))
			for _, f := range fs {
				if fi, err := os.Stat(f); err == nil {
					size += fi.Size()
				}
			}
		} else if fi, err := os.Stat(p); err == nil {
			size = fi.Size()
		} else {
			t.Errorf("%s: %v", name, err)
			return
		}
		out = append(out, auditModel{name: name, path: p, size: size, hf: hf})
	}
	// Every catalog model, its projector with it: a file the box does not
	// hold fails here rather than leaving the list.
	listed := map[string]bool{}
	missing := t.Errorf
	if os.Getenv("JITLLM_AUDIT_MODEL") != "" {
		missing = t.Logf
	}
	for _, lm := range library.Models() {
		p := libraryPath(lm.File)
		if p == "" {
			missing("%s: MODEL MISSING: %s is not in %s (https://huggingface.co/%s)", lm.Name, lm.File, testmodels.Dir(), lm.Repo)
			continue
		}
		listed[p] = true
		mp := towerOf(p)
		if lm.MMProj != "" {
			if mp = libraryPath(lm.MMProj); mp == "" {
				missing("%s: MODEL MISSING: its projector %s is not in %s", lm.Name, lm.MMProj, testmodels.Dir())
				continue
			}
		}
		if mp != "" {
			listed[mp] = true
		}
		n := len(out)
		add(lm.Name, p, false)
		if len(out) > n {
			out[n].mmproj = mp
		}
	}
	withTower := func(name, p string) {
		n := len(out)
		add(name, p, false)
		if len(out) > n {
			out[n].mmproj = towerOf(p)
		}
	}
	for _, p := range languageModels(t) {
		if !listed[p] {
			withTower(filepath.Base(p), p)
		}
	}
	sub, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "*", "*.gguf"))
	sort.Strings(sub)
	for _, p := range sub {
		if laterShard().MatchString(p) || listed[p] || isProjector(p) {
			continue
		}
		withTower(filepath.Join(filepath.Base(filepath.Dir(p)), filepath.Base(p)), p)
	}
	golds, _ := filepath.Glob(filepath.Join("..", "testdata", "golden", "hf", "*.json"))
	for _, g := range golds {
		name := strings.TrimSuffix(filepath.Base(g), ".json")
		dir := testmodels.Path(name)
		if fileExists(filepath.Join(dir, "config.json")) {
			add(name+" (safetensors)", dir, true)
		} else {
			missing("%s: MODEL MISSING: its safetensors fixture %s is not on this box", name, dir)
		}
	}
	// The decision models' files, which sit deeper than one directory down
	// (laya/gguf/, laya/fixture/) where the glob above does not reach.
	for _, rel := range auditDecision() {
		if p := testmodels.Path(rel); fileExists(p) && !listed[p] {
			add(rel, p, false)
		}
	}
	for _, rel := range auditHFDirs() {
		if dir := testmodels.Path(rel); fileExists(filepath.Join(dir, "config.json")) {
			add(rel+" (safetensors)", dir, true)
		} else {
			missing("%s: MODEL MISSING: its safetensors fixture %s is not on this box", rel, dir)
		}
	}
	return out
}

// towerOf is the projector GGUF converted with text GGUF p: the one the tower
// gates pair it with, else mmproj-<its file> beside it, else none.
func towerOf(p string) string {
	rel := filepath.ToSlash(strings.TrimPrefix(p, testmodels.Dir()+string(filepath.Separator)))
	if notTowers()[rel] {
		return ""
	}
	if rel, ok := towerPairs()[rel]; ok {
		if q := testmodels.Path(rel); fileExists(q) {
			return q
		}
	}
	if q := filepath.Join(filepath.Dir(p), "mmproj-"+filepath.Base(p)); fileExists(q) {
		return q
	}
	return ""
}

// towerPairs are the text GGUFs, relative to the model directory, whose
// tower's file is not named mmproj-<file> beside them, as the tower gates pair
// them.
func towerPairs() map[string]string {
	return map[string]string{
		"gemma-3-4b-it-Q4_K_M.gguf":          "vlm/mmproj-gemma-3-4b-it-f16.gguf",
		"llava-phi-3-mini-int4.gguf":         "llava-phi-3-mini-mmproj-f16.gguf",
		"minicpmv/MiniCPM-V-4-Q4_K_M.gguf":   "minicpmv/mmproj-MiniCPM-V-4-f16.gguf",
		"minicpmv/Janus-Pro-1B-Q4_K_M.gguf":  "minicpmv/mmproj-Janus-Pro-1B-Q8_0.gguf",
		"Qwen2-VL-2B-Instruct-Q4_K_M.gguf":   "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
		"Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
		"qwen35/Qwen3.5-0.8B-Q4_K_M.gguf":    "mmproj-Qwen3.5-0.8B-f16.gguf",
	}
}

// notTowers are text GGUFs whose mmproj-<file> beside them is no tower: the
// converter refuses it as input. tinyopenjev-for-testing's is a loader test
// file whose block MLP reads 160 wide and writes 144 (fc1 160x288, fc2
// 288x144), which no residual width fits.
func notTowers() map[string]bool {
	return map[string]bool{"openjev/tinyopenjev-for-testing-Q8_0.gguf": true}
}

// auditPicture is the picture a towered model encodes in the census: the
// tower gates' quadrants and disc.
func auditPicture(t *testing.T) image.Image {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// auditHFDirs are safetensors inputs held to their reference outside
// testdata/golden/hf, relative to the model directory: a class whose
// fixtures carry their own goldens appends them (kimik3_test.go).
func auditHFDirs() []string { return []string{"hf/synth-kimik3", "kimik3/Kimi-K3-0.40B-MXFP4-hf"} }

// auditDecision is the decision models' GGUFs two directories down; d1's and
// lev's one directory down are found by the glob.
func auditDecision() []string {
	return []string{"laya/fixture/laya-fixture.gguf", "laya/gguf/Laya-BF16.gguf"}
}

// isProjector reads general.architecture, as languageModels does: a vision
// projector is the wrong input, not missing work.
func isProjector(p string) bool {
	f, err := gguf.Open(p)
	if err != nil {
		return false
	}
	arch, _ := f.KV["general.architecture"].String()
	f.Close()
	return arch == "clip"
}
