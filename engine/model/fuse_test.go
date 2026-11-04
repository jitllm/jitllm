//go:build amd64 || arm64

package model

import (
	"math"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// fuseModels is every model the fused-run gate decodes: the dense, mixture
// and hybrid real models the allocation gate runs, and one fixture per
// architecture. JITLLM_FUSE_MODELS adds more.
func fuseModels() []string {
	names := []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf",
		"qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", "gemma-3-1b-it-Q4_K_M.gguf",
		"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf"}
	names = append(names, principleFixtures()...)
	names = append(names, principleMixtures()...)
	if v := os.Getenv("JITLLM_FUSE_MODELS"); v != "" {
		names = append(names, strings.Split(v, ",")...)
	}
	return names
}

// fusedDecode greedy-decodes n tokens from a fixed prompt on a fresh instance,
// fused or per op, and returns every step's token and logits, with the pool
// regions and fused phases the decode tokens ran.
type fuseArm struct {
	toks          []int32
	logits        [][]float32
	regions       int64 // parallel pool regions over the decoded tokens
	phases        int64 // phases fused regions ran over them
	runs          int64 // fused runs opened
	grouped       int64 // matvec groups run
	decoded, nlyr int
}

func fusedDecode(t *testing.T, m *Model, fused bool, n int, resid bool) fuseArm {
	t.Helper()
	s := m.NewInstance(128)
	defer s.Close()
	s.SetBlockFusion(fused)
	s.SetResidualFold(resid)
	var a fuseArm
	prompt := []int32{1, 7, 42, 3, 99, 5}
	tok := prompt[0]
	for i := 0; i < len(prompt)+n; i++ {
		if i == len(prompt) {
			// The counters cover the generated tokens, after the prompt warmed
			// the CPU device's forms.
			s.Pool().ResetRegions()
			a.phases = s.Pool().Phases()
			a.runs = s.FusedRuns
			a.grouped = s.Grouped
		}
		lg, err := s.Forward(tok)
		if err != nil {
			t.Fatal(err)
		}
		if i+1 < len(prompt) {
			tok = prompt[i+1]
			continue
		}
		next := Greedy(lg)
		a.toks = append(a.toks, next)
		a.logits = append(a.logits, append([]float32(nil), lg...))
		tok = next
	}
	a.regions, _ = s.Pool().Regions()
	a.phases = s.Pool().Phases() - a.phases
	a.runs = s.FusedRuns - a.runs
	a.grouped = s.Grouped - a.grouped
	a.decoded = n
	a.nlyr = m.Cfg.NLayer
	return a
}

// TestFusedBlocksMatchPerOp: a decode token whose host blocks run as fused
// runs (fuse.go) produces the per-op path's tokens and its logits to the bit,
// on every architecture, resident and paged. The fused arm must have RUN:
// runs opened and phases counted, and fewer pool regions than the per-op arm.
func TestFusedBlocksMatchPerOp(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes real models")
	}
	const n = 12
	for _, name := range fuseModels() {
		t.Run(name, func(t *testing.T) {
			p := testmodels.Path(name)
			for _, paged := range []bool{false, true} {
				m, err := Open(jlmOf(t, p))
				if err != nil {
					t.Fatal(err)
				}
				if paged {
					// Two frames: every block is evicted and read back within a
					// token, so a run ends at each read and reopens after it.
					if m.container == nil || m.container.NPages() < 3 {
						m.Close()
						continue
					}
					m.SetPageBudget(2 * m.PageSize())
				}
				per := fusedDecode(t, m, false, n, false)
				fu := fusedDecode(t, m, true, n, false)
				// The residual fold is off by default; it must be the same
				// bits on.
				fr := fusedDecode(t, m, true, n, true)
				m.Close()
				for i := range per.toks {
					for j, v := range per.logits[i] {
						if math.Float32bits(v) != math.Float32bits(fr.logits[i][j]) {
							t.Fatalf("residual fold: token %d logit %d: %v, per op %v (not the same bits)", i, j, fr.logits[i][j], v)
						}
					}
				}
				arm := "resident"
				if paged {
					arm = "paged"
				}
				if fu.runs == 0 || fu.phases == 0 {
					t.Fatalf("%s: the fused arm opened %d runs and ran %d phases: it measured the per-op path twice",
						arm, fu.runs, fu.phases)
				}
				if per.runs != 0 || per.phases != 0 {
					t.Fatalf("%s: the per-op arm fused (%d runs, %d phases)", arm, per.runs, per.phases)
				}
				for i := range per.toks {
					if per.toks[i] != fu.toks[i] {
						t.Fatalf("%s: token %d: fused %d, per op %d", arm, i, fu.toks[i], per.toks[i])
					}
					for j, v := range per.logits[i] {
						w := fu.logits[i][j]
						if math.Float32bits(v) != math.Float32bits(w) {
							t.Fatalf("%s: token %d logit %d: fused %v, per op %v (not the same bits)", arm, i, j, w, v)
						}
					}
				}
				t.Logf("%s, %d layers: per op %.1f regions/token; fused %.1f regions/token + %.1f phases/token, %.1f runs/token",
					arm, per.nlyr, float64(per.regions)/n, float64(fu.regions)/n, float64(fu.phases)/n, float64(fu.runs)/n)
				if fu.regions >= per.regions {
					t.Fatalf("%s: fused %d regions against per op %d: fusion removed none", arm, fu.regions, per.regions)
				}
			}
		})
	}
}

// fuseWatch logs how many fused runs (fuse.go) the models a test hands it
// opened for its decode tokens, and with must fails a test that ran yet opened
// none: a relocation gate that never ran a fused run says nothing about fusion.
// A batched step and a crew shared by several sessions run per op by design,
// so their gates log the count instead. A nil watch watches nothing.
type fuseWatch struct {
	t    *testing.T
	must bool
	mu   sync.Mutex
	ms   []*Model
}

func fusedRunsCheck(t *testing.T, must bool) *fuseWatch { return &fuseWatch{t: t, must: must} }

func (w *fuseWatch) watch(m *Model) {
	if w == nil {
		return
	}
	w.mu.Lock()
	w.ms = append(w.ms, m)
	w.mu.Unlock()
}

func (w *fuseWatch) done() {
	t, must := w.t, w.must
	if t.Skipped() || t.Failed() {
		return
	}
	var n, asked int64
	for _, m := range w.ms {
		n += m.fusedRuns.Load()
		asked += m.fuseTries.Load()
	}
	t.Logf("fused runs opened: %d of %d asked for", n, asked)
	if must && asked == 0 {
		t.Logf("%s decoded no host token that could fuse (every model it names skipped, or every block ran on a device)", t.Name())
		return
	}
	// sched.Pool.Fuse opens no region on a pool of one worker: with no
	// one else to join there is no round trip to remove, and the per-op
	// path is what such a host runs (a one-physical-core runner).
	if must && n == 0 && len(sched.DecodeCores()) < 2 {
		t.Logf("%s: the decode pool has %d worker(s), and a fused run needs two; the per-op path is this host's",
			t.Name(), len(sched.DecodeCores()))
		return
	}
	if must && n == 0 {
		t.Fatalf("%s opened no fused run: it gated the per-op path only", t.Name())
	}
}
