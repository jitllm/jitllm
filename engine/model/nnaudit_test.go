package model

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
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
func TestEveryModelRunsGenerated(t *testing.T) {
	// JITLLM_AUDIT_MODEL=<substring> audits one file whatever its size (run it
	// under `scripts/cap 24G` for anything over ~16 GB); JITLLM_SLOW=1 admits
	// every model over 2 GiB.
	pick := os.Getenv("JITLLM_AUDIT_MODEL")
	slow := os.Getenv("JITLLM_SLOW") != ""
	var ran, refused, large []string
	for _, src := range auditModels(t) {
		name := src.name
		if pick != "" && !strings.Contains(src.path, pick) {
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
				p, err := jlmOfErr(src.path)
				if errors.Is(err, convert.ErrNotImplemented) {
					why = err
					t.Skipf("NOT IMPLEMENTED (RULE 7 scope): %v", err)
				}
				if err != nil {
					t.Fatalf("convert: %v", err)
				}
				path = p
			}
			m, err := Open(path)
			if err != nil {
				t.Fatalf("the converter accepted it and Open refused it: %v", err)
			}
			defer m.Close()
			auditRun(t, m)
			done = true
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

// auditRun is the gate's body for one opened model: an encoder runs its
// embedder, everything else the three decoder entry points (and its embedder
// too when the container carries pooling); a decision model answers a request
// through its readout first (an encoder's, Laya's, has nothing else to run).
func auditRun(t *testing.T, m *Model) {
	t.Helper()
	if m.Decision() != jlm.DecisionNone {
		d, err := m.NewDecider(4096)
		if err != nil {
			t.Fatalf("NewDecider: %v", err)
		}
		state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
		ans, err := d.Decide(state, qs, nil)
		d.Close()
		if err != nil {
			t.Fatalf("Decide: %v", err)
		}
		for i, a := range ans {
			for _, p := range append([]float64{a.Noul, a.Score, a.Confidence}, a.Probs...) {
				if math.IsNaN(p) || math.IsInf(p, 0) {
					t.Fatalf("Decide: question %s answered %v", qs[i].ID, p)
				}
			}
		}
		if m.enc != nil {
			return
		}
	}
	ids := []int32{1, 2, 3, 4, 5}
	if m.IsEmbedding() {
		e, err := m.NewEmbedder()
		if err != nil {
			t.Fatalf("NewEmbedder: %v", err)
		}
		v, err := e.Embed(ids)
		e.Close()
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		finite(t, "Embed", v)
		if m.enc != nil {
			return
		}
	}
	s := m.NewState(16)
	logits, err := s.Forward(1)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	finite(t, "Forward", logits)
	if _, err := s.Prefill(ids); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	s.Close()
	bs := m.NewBatch(2, 16)
	if _, err := bs.ForwardBatch([]int32{1, 2}); err != nil {
		t.Fatalf("ForwardBatch: %v", err)
	}
	bs.Close()
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
	size       int64
	hf         bool // a safetensors directory
}

// laterShard is the second and later file of a split GGUF; the first names the
// whole model.
var laterShard = regexp.MustCompile(`-0*([2-9]|[1-9][0-9]+)-of-[0-9]+\.gguf$`)

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
	for _, p := range languageModels(t) {
		add(filepath.Base(p), p, false)
	}
	sub, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "*", "*.gguf"))
	sort.Strings(sub)
	for _, p := range sub {
		if laterShard.MatchString(p) || isProjector(p) {
			continue
		}
		add(filepath.Join(filepath.Base(filepath.Dir(p)), filepath.Base(p)), p, false)
	}
	golds, _ := filepath.Glob(filepath.Join("..", "..", "testdata", "golden", "hf", "*.json"))
	for _, g := range golds {
		name := strings.TrimSuffix(filepath.Base(g), ".json")
		dir := testmodels.Path(name)
		if fileExists(filepath.Join(dir, "config.json")) {
			add(name+" (safetensors)", dir, true)
		}
	}
	// The decision models' files, which sit deeper than one directory down
	// (laya/gguf/, laya/fixture/) where the glob above does not reach.
	for _, rel := range auditDecision {
		if p := testmodels.Path(rel); fileExists(p) {
			add(rel, p, false)
		}
	}
	for _, rel := range auditHFDirs {
		if dir := testmodels.Path(rel); fileExists(filepath.Join(dir, "config.json")) {
			add(rel+" (safetensors)", dir, true)
		}
	}
	return out
}

// auditHFDirs are safetensors inputs held to their reference outside
// testdata/golden/hf, relative to the model directory: a class whose
// fixtures carry their own goldens appends them (kimik3_test.go).
var auditHFDirs []string

// auditDecision is the decision models' GGUFs two directories down; d1's and
// lev's one directory down are found by the glob.
var auditDecision = []string{"laya/fixture/laya-fixture.gguf", "laya/gguf/Laya-BF16.gguf"}

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
