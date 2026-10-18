package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Laya against its own code (rl_common.py, rl_agent_api.py), the goldens
// written by scripts/layagold.py: the sequence each question is asked as,
// token for token, and the scorer's logits at the markers.
//
//	laya-fixture  a ModernBERT and head built from Laya's classes with every
//	              parameter and buffer random, converted by llama.cpp's own
//	              convert_hf_to_gguf.py (layagold.py fixture)
//	laya          the published checkpoint, ggml-org/Laya-GGUF BF16

type layaGold struct {
	Answers map[string]json.RawMessage `json:"answers"`
	Logits  map[string][]float64       `json:"logits"`
	IDs     map[string][]int32         `json:"ids"`
	Markers map[string][]int           `json:"markers"`
}

var layaCases = []struct {
	name, gguf, gold string
	tol              float64 // on a raw logit
}{
	// f32 weights: what is left is f32 reduction order (worst 1e-6 on the
	// host, LAYADEV on a device).
	{"fixture", "laya/fixture/laya-fixture.gguf", "laya-fixture", 2e-5},
	// BF16 weights read as the reference reads them (exactly); 28 blocks of
	// f32 order (worst 0.027).
	{"laya", "laya/gguf/Laya-BF16.gguf", "laya", 0.035},
}

// layaArm is where a Laya gate runs its Decider: the host, or every block
// of the encoder and its head on one device (encdev.go), tuning off.
type layaArm struct {
	name string
	gpu  *tier.GPU // nil: the host
}

// layaArms is the host and each device present; the devices close with t.
func layaArms(t *testing.T) []layaArm {
	t.Helper()
	arms := []layaArm{{name: "host"}}
	for _, spec := range stepDevices() {
		g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Cleanup(func() { g.Close() })
		arms = append(arms, layaArm{spec, g})
	}
	return arms
}

// place puts n of d's blocks on the arm's device (-1: every one) and checks
// the device took exactly that many: a gate on a device arm that ran on the
// host proves nothing about the device.
func (a layaArm) place(t *testing.T, m *Model, d *Decider, n int) {
	t.Helper()
	if a.gpu == nil {
		return
	}
	if err := d.SetDevice(a.gpu, n); err != nil {
		t.Fatal(err)
	}
	want := n
	if n < 0 {
		want = layaExpressible(m, d.emb.violation)
	}
	if got := d.DeviceBlocks(); got != want {
		t.Fatalf("%s took %d of the %d blocks offered: %v %s", a.name, got, want, d.emb.DeviceDeclines(), a.gpu.Err())
	}
	// The head a device leaves home is declined by name, for its geometry.
	if n < 0 && want < m.encBlocks() && !strings.Contains(fmt.Sprint(d.emb.DeviceDeclines()), "heads of") {
		t.Fatalf("%s left the head home with no named reason: %v", a.name, d.emb.DeviceDeclines())
	}
}

// layaExpressible is how many of a Laya model's blocks a device takes: every
// one, or the encoder's alone where the head attends at another geometry
// than the encoder (the random fixture: 2 heads of 64 beside 4 of 32), which
// the non-causal set, built for the encoder's, declines by name
// (tier.nonCausalConflict). Every published checkpoint's head has the
// encoder's geometry (decision-models.md, section 8).
func layaExpressible(m *Model, violation string) int {
	mb := m.enc.mb
	if mb.headHeads != m.Cfg.NHead && violation != "encoder-heads" {
		return len(mb.blocks)
	}
	return m.encBlocks()
}

// layaDevTol is how far a device's logit may sit from the reference's beyond
// the host's tolerance: the f32 reduction order of the device's own kernels
// over 28 blocks (the checkpoint) or 2 (the fixture), and on a V100 the
// binary16 operands of sm_70's prompt kernels: the fixture there reads 2.6e-4,
// and 1e-6 with tier.Config.NoMMA and NoVolta, which take them away.
var layaDevTol = map[string]float64{"fixture": 4e-4, "laya": 0.035}

func TestLayaMatchesReference(t *testing.T) {
	for _, c := range layaCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, jlm.DecisionLaya)
			defer m.Close()
			for _, arm := range layaArms(t) {
				t.Run(arm.name, func(t *testing.T) {
					tol := c.tol
					if arm.gpu != nil {
						tol += layaDevTol[c.name]
					}
					layaMatchesReference(t, m, c.gold, tol, arm)
				})
			}
		})
	}
}

// layaMatchesReference is TestLayaMatchesReference on one model and arm.
func layaMatchesReference(t *testing.T, m *Model, gold string, tol float64, arm layaArm) {
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	arm.place(t, m, d, -1)
	for _, req := range decisionRequests {
		t.Run(req, func(t *testing.T) {
			var g layaGold
			b, err := os.ReadFile(filepath.Join("testdata", "decision", gold+"."+req+".reference.json"))
			if err != nil {
				testmodels.Missing(t, "%v (scripts/layagold.py writes it)", err)
			}
			if err := json.Unmarshal(b, &g); err != nil {
				t.Fatal(err)
			}
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
			worst := 0.0
			for i := range qs {
				q := &qs[i]
				ids, markers, err := d.layaSequence(state, q)
				if err != nil {
					t.Fatal(err)
				}
				if !equalIDs(ids, g.IDs[q.ID]) || !equalInts(markers, g.Markers[q.ID]) {
					t.Fatalf("%s: the sequence differs from the reference's\n got %v %v\nwant %v %v",
						q.ID, ids, markers, g.IDs[q.ID], g.Markers[q.ID])
				}
				s, err := d.layaRun(state, q)
				if err != nil {
					t.Fatal(err)
				}
				want := g.Logits[q.ID]
				if len(s) != len(want) {
					t.Fatalf("%s: %d scores, the reference %d", q.ID, len(s), len(want))
				}
				for j := range s {
					if math.IsNaN(float64(s[j])) || math.IsInf(float64(s[j]), 0) {
						t.Fatalf("%s: score %d is %v", q.ID, j, s[j])
					}
					worst = max(worst, math.Abs(float64(s[j])-want[j]))
					if math.Abs(float64(s[j])-want[j]) > tol {
						t.Errorf("%s option %d: logit %.5f, the reference %.5f", q.ID, j, s[j], want[j])
					}
				}
			}
			t.Logf("%s: worst logit difference from the reference: %.6f", arm.name, worst)
			if arm.gpu == nil && d.emb.jit.FloatMatMulCalls() == 0 {
				t.Errorf("the encoder's float matrices never ran the float GEMM")
			}
		})
	}
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestLayaFeaturesAreLoadBearing removes each feature of the graph from the
// fixture and requires the logits to leave the reference's by ten times the
// gate's tolerance: a feature the gate cannot see is not gated (RULE 10). On
// a device arm every block is placed with the feature gone from what the
// device is handed (encPlan, encWeights): the device's own wiring of each
// feature is what is seen there. The type row is added on the host on both.
func TestLayaFeaturesAreLoadBearing(t *testing.T) {
	m := openDecision(t, "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya)
	defer m.Close()
	var g layaGold
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "laya-fixture.support.reference.json"))
	if err != nil {
		testmodels.Missing(t, "%v (scripts/layagold.py writes it)", err)
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
	run := func(t *testing.T, arm layaArm, v string) map[string][]float32 {
		d, err := m.NewDecider(0)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		d.emb.violation = v
		arm.place(t, m, d, -1)
		out := map[string][]float32{}
		for i := range qs {
			q := &qs[i]
			if v == "type-row-0" && q.Type == jlm.QuestionChoice {
				continue // row 0 is the choice row: nothing removed
			}
			s, err := d.layaRun(state, q)
			if err != nil {
				t.Fatal(err)
			}
			out[q.ID] = s
		}
		return out
	}
	for _, arm := range layaArms(t) {
		// On the host the reference is Laya's own logits; on a device the
		// same device's clean run, which is deterministic (tuning off): a
		// device's f32 order sits 1e-4 from the reference on the fixture,
		// wider than what the tanh GELU moves (3e-4).
		want := map[string][]float32{}
		for id, l := range g.Logits {
			for _, x := range l {
				want[id] = append(want[id], float32(x))
			}
		}
		if arm.gpu != nil {
			want = run(t, arm, "")
		}
		seen := 10 * layaCases[0].tol
		for _, v := range []string{"no-window", "one-base", "up-gated", "type-row-0", "encoder-heads", "tanh-gelu"} {
			t.Run(arm.name+"/"+v, func(t *testing.T) {
				worst := 0.0
				for id, s := range run(t, arm, v) {
					for j := range s {
						worst = max(worst, math.Abs(float64(s[j]-want[id][j])))
					}
				}
				t.Logf("%s: %s moves a logit by %.4f", arm.name, v, worst)
				if worst <= seen {
					t.Errorf("removing %s moved no logit past %.4f: the gate does not see it", v, seen)
				}
			})
		}
	}
}

// TestLayaPagesWithTheSameAnswer: the encoder and its head under a budget of
// one block page evict and re-read every block a question passes through, and
// answer bit for bit as with every page resident (the Paging principle) --
// on the host, and on each device with half the blocks placed and the rest
// paged through the host's one frame.
func TestLayaPagesWithTheSameAnswer(t *testing.T) {
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
	run := func(frames int, arm layaArm, seam int) ([][]float32, int64) {
		m := openDecision(t, "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya)
		defer m.Close()
		if frames > 0 {
			m.SetPageBudget(uint64(frames) * m.PageSize())
		}
		d, err := m.NewDecider(0)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		arm.place(t, m, d, seam)
		var out [][]float32
		for i := range qs {
			s, err := d.layaRun(state, &qs[i])
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, s)
		}
		return out, m.PagerReads()
	}
	blocks := int64(m0Blocks(t))
	for _, arm := range layaArms(t) {
		t.Run(arm.name, func(t *testing.T) {
			seam := int(blocks) / 2
			if arm.gpu == nil {
				seam = 0
			}
			whole, _ := run(0, arm, seam)
			paged, reads := run(1, arm, seam)
			if home := blocks - int64(seam); reads <= home {
				t.Fatalf("a one-page budget read %d pages for the %d blocks at home: nothing was evicted, "+
					"so nothing was tested", reads, home)
			}
			for i := range whole {
				for j := range whole[i] {
					if whole[i][j] != paged[i][j] {
						t.Errorf("%s option %d: %v resident, %v paged", qs[i].ID, j, whole[i][j], paged[i][j])
					}
				}
			}
			t.Logf("%s: %d of %d blocks placed, %d page reads over %d questions", arm.name, seam, blocks, reads, len(qs))
		})
	}
}

// m0Blocks is the fixture container's block page count.
func m0Blocks(t *testing.T) int {
	m := openDecision(t, "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya)
	defer m.Close()
	return m.container.NPages()
}

// layaRun is one question's scores as the gates keep them: a copy of the
// Decider's scratch, which the next question overwrites.
func (d *Decider) layaRun(state jinja.Value, q *DecisionQuestion) ([]float32, error) {
	s, err := d.run(state, q, 0)
	return slices.Clone(s), err
}
