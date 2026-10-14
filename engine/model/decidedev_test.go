package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The decision readouts on a device (RULE 11c): each decision model's answers
// with every block placed, and with a seam that leaves blocks home, against
// the host's on the same bytes, on every device present. The device's split
// tuner is off (tier.TuneOff), so its reduction order is fixed for the run.
//
// decisionDevTol is how far a probability may move between the host and a
// device: the f32 reduction-order band of the label logits (0.2-0.9 of a
// logit on a 20-wide scale; RULE 11c) seen through the readout's softmax at a
// near-tie (d1 measured 0.008-0.025 on the laptop's CUDA and Vulkan). The
// violation arm must move an answer past it.
//
// lev's is 0.08, measured by moving the seam (RULE 11c) on a V100 over 2, 4,
// 8, 12, 16, 20, 24, 28 and 32 device blocks: CUDA 0.041, 0.036, 0.020,
// 0.045, 0.043, 0.048, 0.046, 0.047, 0.046 and Vulkan 0.071, 0.028, 0.014,
// 0.017, 0.015, 0.014, 0.016, 0.017, 0.018. The error does not grow with the
// blocks below the seam -- two blocks already show it, and every block no
// more -- so it is not a device bug amplified by depth but the band at one
// near-tie, a nine-way rating distribution whose top two sit 0.02 apart
// (ambiguous question 3, p 0.47 against 0.49): the laptop's 0.058 on five
// blocks is the same. Its violations move the answers 0.65-0.69.
var decisionDevTol = map[string]float64{"d1-3B": 0.03, "lev": 0.08, "laya": 0.01, "clef": 0.05, "clef-fixture": 0.002, "d1-omni": 0.01}

// decisionBlocks is how many blocks a Decider's readout runs: the decoder's,
// or the encoder segment's (a decision head's blocks with it).
func decisionBlocks(m *Model) int {
	if m.IsEncoder() {
		return m.encBlocks()
	}
	return m.Cfg.NLayer
}

// decideOn answers both decision requests on a Decider placed with n blocks
// on dev (nil: the host), and reports the blocks it ran on the device. ok
// sees the placement before anything runs; false returns at once, no answers.
// violation is the Decider's (a readout feature a test removes); picked is
// how many prompts' labels the device picked (nn.Head.Pick).
func decideOn(t *testing.T, m *Model, dev nn.Device, n int, ok func(placed int) bool, violation ...string) ([][]DecisionAnswer, int, string, int) {
	t.Helper()
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if len(violation) > 0 {
		d.violation = violation[0]
	}
	why := ""
	gpu, _ := dev.(*tier.GPU)
	if dev != nil {
		if err := d.SetDevice(dev, n); err != nil {
			t.Fatal(err)
		}
		if gpu != nil {
			why = gpu.Err()
		}
		if d.st != nil {
			why = fmt.Sprintf("%v %s", d.st.DeviceDeclines(), why)
		} else {
			why = fmt.Sprintf("%v %s", d.emb.DeviceDeclines(), why)
		}
	}
	placed := d.DeviceBlocks()
	if ok != nil && !ok(placed) {
		return nil, placed, why, 0
	}
	var s0 int
	if gpu != nil {
		s0 = gpu.Stats().Submits
	}
	var out [][]DecisionAnswer
	for _, req := range decisionRequests {
		state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
		ans, err := d.Decide(state, qs, nil)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, ans)
	}
	if got := d.DeviceBlocks(); got != placed {
		t.Fatalf("the readout fell from %d device blocks to %d while it ran", placed, got)
	}
	// The device ran them: at least one submission a prompt, counted by the
	// device itself.
	if gpu != nil && placed > 0 && gpu.Stats().Submits-s0 < len(decisionRequests) {
		t.Fatalf("%d submission(s) for %d requests: the placed blocks did not run on the device",
			gpu.Stats().Submits-s0, len(decisionRequests))
	}
	// Clef has no labels to pick: what the device took is its head, one
	// request at a time.
	if d.kind == jlm.DecisionClef {
		return out, placed, why, d.clefDevRuns
	}
	return out, placed, why, d.picked
}

// decisionDiff is the largest difference between two runs' probabilities,
// nouls and expected levels (a score over its k-1 levels, so every value is on
// [0, 1]), infinite on a value that is not finite. It logs where it is.
func decisionDiff(t *testing.T, want, got [][]DecisionAnswer) float64 {
	t.Helper()
	worst, where := 0.0, ""
	for r := range want {
		for i := range want[r] {
			a, b := want[r][i], got[r][i]
			type val struct {
				what string
				w, g float64
			}
			vals := []val{{"noul", a.Noul, b.Noul}}
			if k := len(a.Probs); k > 1 {
				vals = append(vals, val{"score/(k-1)", a.Score / float64(k-1), b.Score / float64(k-1)})
			}
			for j := range a.Probs {
				vals = append(vals, val{fmt.Sprintf("p[%d]", j), a.Probs[j], b.Probs[j]})
			}
			for _, v := range vals {
				if math.IsNaN(v.g) || math.IsInf(v.g, 0) {
					return math.Inf(1)
				}
				if d := math.Abs(v.w - v.g); d > worst {
					worst = d
					where = fmt.Sprintf("request %s, question %d, %s: host %.4f, device %.4f",
						decisionRequests[r], i, v.what, v.w, v.g)
				}
			}
			if a.Choice != b.Choice {
				t.Logf("request %s, question %d: the device chose %q, the host %q", decisionRequests[r], i, b.Choice, a.Choice)
			}
		}
	}
	t.Logf("worst at %s", where)
	return worst
}

// decisionDevViolation is a feature taken out of the graph the device runs,
// read when the blocks are offered (the plan), so only the device arm sees it.
type decisionDevViolation struct {
	name string
	cut  func(m *Model) func()
}

var decisionDevViolations = map[string]decisionDevViolation{
	// The norm's epsilon is read into the plan (LayerPlan.RMSEps).
	"d1-3B": {"a norm epsilon of 1", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.RMSEps = 1 })
	}},
	"lev": {"a norm epsilon of 1", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.RMSEps = 1 })
	}},
	"d1-omni": {"a norm epsilon of 1", func(m *Model) func() {
		return cfgMut(m, func(c *Config) { c.RMSEps = 1 })
	}},
	// The device reads which blocks are local from the plan (mb.local): every
	// block global there attends over every row at the global base.
	// The device reads its own copy of the head (clefDevHead): with the joint
	// score gated off there, only the prior answers on the device.
	"clef":         {"the device head's joint score gated off", clefDevGateOff},
	"clef-fixture": {"the device head's joint score gated off", clefDevGateOff},
	"laya": {"every block global on the device", func(m *Model) func() {
		mb := m.enc.mb
		was := mb.local
		mb.local = func(int) bool { return false }
		return func() { mb.local = was }
	}},
}

// TestDecisionOnEveryDevice runs each decision model with every block on each
// device present and with half of them, against the host. The every-block arm
// needs a card the model fits; where it does not, the arm says how many blocks
// the card held and skips, and the seam arm still runs.
func TestDecisionOnEveryDevice(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			if !m.IsEncoder() {
				m.SetKVF16(false) // the device's history is f32
			}
			tol := decisionDevTol[c.name]
			host, _, _, _ := decideOn(t, m, nil, 0, nil)
			nb := decisionBlocks(m)
			ran := 0
			for _, spec := range stepDevices() {
				probe, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || probe == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				probe.Close()
				ran++
				t.Run(spec, func(t *testing.T) {
					open := func() *tier.GPU {
						g, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
						if err != nil {
							t.Fatal(err)
						}
						return g
					}
					t.Run("seam", func(t *testing.T) {
						g := open()
						defer g.Close()
						// Half the blocks, or what a small card holds of them: a
						// seam either way, the rest on the host.
						want := nb / 2
						seam := func(n int) bool { return n > 0 && n <= want }
						got, placed, why, _ := decideOn(t, m, g, want, seam)
						if !seam(placed) {
							t.Fatalf("placed %d of the %d blocks offered: %s", placed, want, why)
						}
						worst := decisionDiff(t, host, got)
						t.Logf("%d of %d blocks on %s: worst difference from the host %.5f", placed, nb, spec, worst)
						if worst > tol {
							t.Errorf("the answers moved %.5f from the host's, past %.3f", worst, tol)
						}
					})
					t.Run("every-block", func(t *testing.T) {
						g := open()
						defer g.Close()
						all := func(n int) bool { return n == nb }
						got, placed, why, picked := decideOn(t, m, g, -1, all)
						if placed != nb {
							t.Skipf("%s held %d of %d blocks (%s): this card is too small for the every-block arm",
								spec, placed, nb, why)
						}
						worst := decisionDiff(t, host, got)
						t.Logf("every block on %s: worst difference from the host %.5f", spec, worst)
						if worst > tol {
							t.Errorf("the answers moved %.5f from the host's, past %.3f", worst, tol)
						}
						if c.kind == jlm.DecisionClef {
							// The head ran on the device for every request, and
							// the same rows through the host's head answer within
							// the band.
							if picked != len(decisionRequests) {
								t.Fatalf("Clef's head ran on %s for %d of %d requests", spec, picked, len(decisionRequests))
							}
							home, _, _, hp := decideOn(t, m, g, -1, all, "host-head")
							if hp != 0 {
								t.Fatalf("the host-head arm ran the head on the device %d times", hp)
							}
							w := decisionDiff(t, home, got)
							t.Logf("the head on %s against the host's head over the same rows: %.5f", spec, w)
							if w > tol {
								t.Errorf("the device's head answers %.5f from the host's over the same rows", w)
							}
						} else if !m.IsEncoder() {
							// The labels' logits were picked on the device, and
							// are the row's: the same placement reading the whole
							// row back answers bit for bit. Picked one past each
							// label, the answers move (the pick is what they read).
							if picked == 0 {
								t.Fatalf("no prompt's labels were picked on %s: the head's whole row came home", spec)
							}
							// And by the picked rows' own projection, not the whole
							// head and a gather (tier.Config.NoPickMV's arm).
							if n := g.Stats().PickRows; n == 0 {
								t.Fatalf("%d prompts picked on %s, none by the picked rows' projection", picked, spec)
							}
							row, _, _, rowPicked := decideOn(t, m, g, -1, all, "no-pick")
							if rowPicked != 0 {
								t.Fatalf("the no-pick arm picked %d prompts", rowPicked)
							}
							if w := decisionDiff(t, got, row); w != 0 {
								t.Errorf("the picked labels answer %.6f from the row's on the same device", w)
							}
							off, _, _, _ := decideOn(t, m, g, -1, all, "pick-off-by-one")
							seen := decisionDiff(t, got, off)
							t.Logf("%d prompts picked on %s; picked one past each label, the answers move %.5f", picked, spec, seen)
							if seen <= tol {
								t.Errorf("a pick one past each label moved the answers %.5f: the gate does not see the pick", seen)
							}
						}
						v, ok := decisionDevViolations[c.name]
						if !ok {
							t.Fatalf("%s has no device violation", c.name)
						}
						// A fresh device, with g's room given back first: the
						// clean arm's blocks stay resident on g for whoever
						// attaches next (a decoder State's Close leaves them),
						// and would be shared rather than offered.
						g.Close()
						gv := open()
						defer gv.Close()
						undo := v.cut(m)
						bad, n, _, _ := decideOn(t, m, gv, -1, all)
						undo()
						if bad == nil {
							t.Fatalf("%s: the device held %d of %d blocks", v.name, n, nb)
						}
						seen := decisionDiff(t, host, bad)
						t.Logf("%s on %d device blocks moves the answers %.5f", v.name, n, seen)
						if seen <= tol {
							t.Errorf("%s moved the answers %.5f on %d blocks: the gate does not see it", v.name, seen, n)
						}
					})
				})
			}
			if ran == 0 {
				noDevice(t, "device", nil)
			}
		})
	}
}

// clefDevGateOff hands the device a copy of Clef's head whose joint score is
// gated off (the residual gate's value zero), which the host's head does not
// read.
func clefDevGateOff(m *Model) func() {
	if _, err := m.clefLoad(); err != nil {
		panic(err)
	}
	h := m.clefDevHead()
	cp := *h
	cp.Scales[2] = 0
	m.bind.Lock()
	m.clefDev = &cp
	m.bind.Unlock()
	return func() {
		m.bind.Lock()
		m.clefDev = h
		m.bind.Unlock()
	}
}
