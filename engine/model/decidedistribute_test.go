package model

import (
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// Relocation between devices and distribution over them, on the decision
// models: the device-to-device move (State.MoveBlocks) mid-prompt and between
// questions, blocks split across two cards and across the host and two cards,
// and a batch of requests against each alone. The two devices are
// JITLLM_TEST_DEVICES' first two (moveDevices): two cards of one backend run the same
// kernels, so a move between them is held bit for bit, and two backends to
// the reduction-order band.

// moveTier opens moveDevices' two devices as one tier, tuning off; it closes
// with t.
func moveTier(t *testing.T) (*tier.GPU, string, string) {
	t.Helper()
	a, b := moveDevices(t)
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(a + "," + b),
		tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	t.Cleanup(func() { g.Close() })
	distinctCards(t, g)
	return g, a, b
}

// sameKernels says the two devices are one backend's, so a move between them
// must not move a bit.
func sameKernels(a, b string) bool {
	return strings.SplitN(a, ":", 2)[0] == strings.SplitN(b, ":", 2)[0]
}

// decisionSegment is the State a Decider places its blocks through: a
// decoder's own, an encoder's segment.
func decisionSegment(d *Decider) *State {
	if d.emb != nil {
		return d.emb.seg
	}
	return d.st
}

// TestDecisionMovesBetweenCards moves every block of each decision model from
// one card to the other and back with MoveBlocks, the history and the
// recurrent summaries copied card to card: a decoder's a third into a prompt
// and back at two thirds, and every model's between two questions. No prompt
// row is computed again, every block lands where it was sent, and the label
// logits and the answers are the unmoved run's. The violation is a move whose
// recurrent summaries were lost on the way.
func TestDecisionMovesBetweenCards(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			g, a, b := moveTier(t)
			m := openDecision(t, c.gguf, c.kind, noTune,
				WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}))
			defer m.Close()
			if !m.IsEncoder() {
				m.SetKVF16(false)
			}
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			if err := d.SetDevice(g, -1); err != nil {
				t.Fatal(err)
			}
			st := decisionSegment(d)
			nb := decisionBlocks(m)
			if got := len(st.placedOn(a)); got != nb {
				cardTooSmall(t, g.Err(), "%s held %d of %d blocks: every block must be on it to move them all", a, got, nb)
			}
			// move sends every block to the named device and checks it went,
			// with nothing computed again.
			move := func(t *testing.T, to string) BlockMove {
				t.Helper()
				rows := st.PrefillRows()
				mv, err := st.MoveBlocks(st.lo, st.hi, to)
				if err != nil {
					t.Fatalf("to %s: %v", to, err)
				}
				if got := st.PrefillRows() - rows; got != 0 {
					t.Fatalf("the move to %s computed %d prompt rows again", to, got)
				}
				if len(mv.Moved) != nb || len(st.placedOn(to)) != nb {
					t.Fatalf("to %s: %d moved, %d there of %d (declined %v)", to, len(mv.Moved), len(st.placedOn(to)), nb, mv.Declined)
				}
				return mv
			}
			tol := decisionDevTol[c.name]
			if !sameKernels(a, b) && !m.IsEncoder() {
				tol = decisionRelocTol[c.name]
			}
			if sameKernels(a, b) {
				tol = 0
			}
			if !m.IsEncoder() {
				t.Run("mid-prompt", func(t *testing.T) {
					state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "ambiguous.json"))
					rp := readoutOf(t, d, state, qs)
					ids := rp.ids
					var err error
					x, y := len(ids)/3, 2*len(ids)/3
					// run prefills the prompt in three parts, the blocks moved
					// between them when moved is set; lose clears the device's
					// recurrent summaries before the first move.
					run := func(moved, lose bool) ([]float32, []BlockMove) {
						rp.start()
						rows0 := st.PrefillRows()
						var lg []float32
						var mvs []BlockMove
						for i, part := range [][]int32{ids[:x], ids[x:y], ids[y:]} {
							if moved && i > 0 {
								if lose && i == 1 {
									rr, ok := st.ld.(nn.RecRowsDevice)
									if !ok || !rr.ResetRecRows([]int{0}) {
										t.Fatal("the device could not clear its recurrent state")
									}
								}
								to := b
								if i == 2 {
									to = a
								}
								mvs = append(mvs, move(t, to))
							}
							if lg, err = st.Prefill(part); err != nil {
								t.Fatal(err)
							}
						}
						if got := st.PrefillRows() - rows0; got != len(ids) {
							t.Fatalf("%d prompt rows run for a %d-token prompt", got, len(ids))
						}
						return rp.finish(lg), mvs
					}
					ctl, _ := run(false, false)
					got, mvs := run(true, false)
					for i, mv := range mvs {
						t.Logf("move %d: %d blocks, %d KV and %d recurrent bytes, path %q, %d copied, head %v, pause %v",
							i, len(mv.Moved), mv.KVBytes, mv.RecBytes, mv.Path, mv.Copied, mv.Head, mv.Pause)
					}
					if mvs[0].KVBytes+mvs[0].RecBytes == 0 {
						t.Fatalf("the move carried no history: %+v", mvs[0])
					}
					if !mvs[0].Head {
						t.Errorf("the output projection stayed on %s with no block left there", a)
					}
					if _, direct := any(g.Attach()).(nn.BlockMover); direct && mvs[0].Path == "host" {
						t.Errorf("the history came home: the direct path did not run")
					}
					w := labelWorst(ctl, got)
					t.Logf("label logits moved %v, unmoved %v: worst %.6f (bound %g)", got, ctl, w, tol)
					if !(w <= tol) {
						t.Errorf("moved a third and two thirds into the prompt, a label logit moved %.6f", w)
					}
					bad, _ := run(true, true)
					seen := labelWorst(ctl, bad)
					t.Logf("the recurrent summaries lost on the way: worst %.4f", seen)
					if seen <= max(tol, decisionRelocTol[c.name]) {
						t.Errorf("a move that lost the recurrent state moved a label logit %.6f: the gate does not see it", seen)
					}
				})
			}
			t.Run("between-questions", func(t *testing.T) {
				state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
				answer := func() [][]DecisionAnswer {
					ans, err := d.Decide(state, qs, nil)
					if err != nil {
						t.Fatal(err)
					}
					return [][]DecisionAnswer{ans}
				}
				ctl := answer()
				move(t, b)
				onB := answer()
				move(t, a)
				back := answer()
				wb, wa := decisionDiffRow(t, ctl, onB), decisionDiffRow(t, ctl, back)
				t.Logf("answers on %s %.6f from %s's, back on %s %.6f (bound %g)", b, wb, a, a, wa, tol)
				// An encoder's non-causal set keeps the rows the calls before it
				// sized (tier.visShrinkCalls), so back on the first card a
				// question may run at another chunking than it first did: the
				// device band, where a decoder's history is bit for bit.
				backTol := 0.0
				if m.IsEncoder() {
					backTol = decisionDevTol["laya"]
				}
				// Answers are held to the device band, logits (above) to the
				// relocation one: vllm-d3's are 0.10 and 0.026.
				if !sameKernels(a, b) && !m.IsEncoder() {
					tol = max(tol, decisionDevTol[c.name])
				}
				if !(wb <= tol) || !(wa <= backTol) {
					t.Errorf("moved between questions, the answers moved %.6f on %s and %.6f back", wb, b, wa)
				}
			})
		})
	}
}

// labelWorst is the largest difference between two runs' label logits, infinite
// on one that is not finite.
func labelWorst(want, got []float32) float64 {
	w := 0.0
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		if math.IsNaN(d) {
			return math.Inf(1)
		}
		w = max(w, d)
	}
	return w
}

// TestDecisionDistributes runs each decision model's blocks split across two
// cards at once, and across the host and two cards, against the host's
// answers: every block where the placement names it, and the answers inside
// the device band. The violation is a feature removed from what the devices
// are handed (decisionDevViolations), which must move them past it.
func TestDecisionDistributes(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			a, b := moveDevices(t)
			hm := openDecision(t, c.gguf, c.kind, noTune)
			if !hm.IsEncoder() {
				hm.SetKVF16(false)
			}
			host, _, _, _ := decideOn(t, hm, nil, 0, nil)
			nb := decisionBlocks(hm)
			hm.Close()
			tol := decisionDevTol[c.name]
			for _, arm := range []struct {
				name  string
				place func(li int) string
			}{
				{"two-cards", func(li int) string { return []string{a, b}[li*2/nb] }},
				{"host-and-two-cards", func(li int) string { return []string{"host", a, b}[li*3/nb] }},
			} {
				t.Run(arm.name, func(t *testing.T) {
					pl := Placement{Blocks: map[int]Place{}, Head: &Place{On: b}, Strict: true}
					if c.kind == jlm.DecisionLaya || c.kind == jlm.DecisionD1Omni {
						pl.Head = nil // an encoder has no output projection: its scorer is the head's last block
					}
					want := map[string]int{}
					for li := range nb {
						pl.Blocks[li] = Place{On: arm.place(li)}
						want[arm.place(li)]++
					}
					run := func(violation bool) [][]DecisionAnswer {
						g, _, _ := moveTier(t)
						m := openDecision(t, c.gguf, c.kind, noTune, WithPlacement(pl))
						defer m.Close()
						if !m.IsEncoder() {
							m.SetKVF16(false)
						}
						if violation {
							defer decisionDevViolations[c.name].cut(m)()
						}
						got, placed, why, _ := decideOn(t, m, g, -1, func(int) bool { return true })
						if placed != nb-want["host"] {
							t.Fatalf("%d of %d blocks placed, want %d: %s", placed, nb, nb-want["host"], why)
						}
						return got
					}
					got := run(false)
					w := decisionDiff(t, host, got)
					t.Logf("%v: worst difference from the host %.5f (bound %.3f)", want, w, tol)
					if w > tol {
						t.Errorf("split %v, the answers moved %.5f from the host's", want, w)
					}
					v := decisionDevViolations[c.name]
					seen := decisionDiff(t, host, run(true))
					t.Logf("%s, split %v: the answers move %.5f", v.name, want, seen)
					if seen <= tol {
						t.Errorf("%s moved the answers %.5f split %v: the gate does not see it", v.name, seen, want)
					}
				})
			}
		})
	}
}

// TestDecisionBatchMatchesAlone answers several requests in one DecideBatch
// -- each request twice, and one that fails its checks -- against each
// answered alone, on the host and with the blocks on each device: a decoder
// readout's prompts as rows of one batch session (more than one prompt in a
// step, counted), an encoder's one pass each. The failing request fails
// alone. The violation is each prompt chunk laid out at its positions reversed.
func TestDecisionBatchMatchesAlone(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			if !m.IsEncoder() {
				m.SetKVF16(false)
			}
			var reqs []DecisionRequest
			for range 2 {
				for _, r := range decisionRequests {
					state, qs := decisionRequest(t, filepath.Join("testdata", "decision", r+".json"))
					reqs = append(reqs, DecisionRequest{State: state, Questions: qs})
				}
			}
			bad := DecisionRequest{State: reqs[0].State} // no questions: refused alone
			reqs = append(reqs[:1], append([]DecisionRequest{bad}, reqs[1:]...)...)
			arms := []*tier.GPU{nil}
			arms = append(arms, decisionDevices(t)...)
			for _, g := range arms {
				name := "host"
				if g != nil {
					name = g.Name()
				}
				t.Run(name, func(t *testing.T) {
					d, err := m.NewDecider(4096)
					if err != nil {
						t.Fatal(err)
					}
					defer d.Close()
					placed := 0
					if g != nil {
						if err := d.SetDevice(g, decisionBlocks(m)/2); err != nil {
							t.Fatal(err)
						}
						if placed = d.DeviceBlocks(); placed == 0 {
							t.Fatalf("%s took no block: %s", g.Name(), g.Err())
						}
					}
					var alone [][]DecisionAnswer
					var tokens []int
					for _, r := range reqs {
						if len(r.Questions) == 0 {
							alone, tokens = append(alone, nil), append(tokens, 0)
							continue
						}
						ans, err := d.Decide(r.State, r.Questions, nil)
						if err != nil {
							t.Fatal(err)
						}
						alone = append(alone, ans)
						tokens = append(tokens, d.InputTokens())
					}
					if g != nil && !m.IsEncoder() && c.kind != jlm.DecisionClef && c.kind != jlm.DecisionKev {
						// On a device each request runs alone (batchAlone): the
						// same answers bit for bit, and no batch step.
						got, err := d.DecideBatch(reqs)
						if err != nil {
							t.Fatal(err)
						}
						for i, r := range got {
							if (r.Err != nil) != (len(reqs[i].Questions) == 0) {
								t.Fatalf("request %d: error %v", i, r.Err)
							}
							for qi, a := range r.Answers {
								if !slices.Equal(a.Probs, alone[i][qi].Probs) {
									t.Fatalf("request %d question %d: %v in the batch, %v alone", i, qi, a.Probs, alone[i][qi].Probs)
								}
							}
						}
						if st := d.BatchStats(); st.Steps != 0 {
							t.Fatalf("a batch on the device ran %d batch steps: its requests were to run alone", st.Steps)
						}
						// The batch session itself stays held to the answers alone.
						d.violation = "device-batch"
						defer func() { d.violation = "" }()
					}
					packs0 := d.packs
					got, err := d.DecideBatch(reqs)
					if err != nil {
						t.Fatal(err)
					}
					if m.IsEncoder() {
						// Packed: fewer encoder passes than prompts, more than
						// one sequence in a pass.
						passes := d.packs - packs0
						t.Logf("%d prompts in %d packed encoder pass(es)", d.batchN, passes)
						if passes == 0 || passes >= d.batchN {
							t.Fatalf("%d prompts ran in %d encoder passes: nothing was packed", d.batchN, passes)
						}
					}
					for i, r := range got {
						if (r.Err != nil) != (len(reqs[i].Questions) == 0) {
							t.Fatalf("request %d: error %v", i, r.Err)
						}
						if r.InputTokens != tokens[i] {
							t.Fatalf("request %d ran %d tokens in the batch and %d alone", i, r.InputTokens, tokens[i])
						}
					}
					// A non-causal model's prompts run one at a time (runJobs), and Kev's each alone.
					if !m.IsEncoder() && c.kind != jlm.DecisionClef && !d.nonCausal() && c.kind != jlm.DecisionKev {
						st := d.BatchStats()
						t.Logf("batch: %d steps, %d prompt rows, at most %d prompts in a step; %d of %d blocks on the device",
							st.Steps, st.PromptRows, st.MaxSeqs, d.BatchDeviceBlocks(), decisionBlocks(m))
						if st.MaxSeqs < 2 {
							t.Fatalf("no step carried two prompts: the batch ran them one at a time")
						}
						// A device with no room for a step brings the session's
						// blocks home first (nn.RowsRoomer); otherwise the
						// session holds what the Decider holds.
						if d.BatchDeviceBlocks() != placed && st.RoomHome == 0 {
							t.Fatalf("the batch session holds %d device blocks, the Decider %d", d.BatchDeviceBlocks(), placed)
						}
						if st.RoomHome > 0 {
							t.Logf("%d step(s) found no room on the device and ran with the blocks home", st.RoomHome)
						}
					}
					// The band each readout is held to against llama.cpp: a batch
					// of rows runs the device's row kernels where a prompt alone
					// runs its prefill, another reduction order, and lev's tone
					// question is a near-tie (0.062 on the V100's Vulkan, at 16
					// of 32 blocks; the violation moves 0.7).
					tol := c.tol
					worst := 0.0
					for i := range reqs {
						if alone[i] == nil {
							continue
						}
						worst = max(worst, decisionDiffRow(t, [][]DecisionAnswer{alone[i]}, [][]DecisionAnswer{got[i].Answers}))
					}
					t.Logf("%d requests in one batch: worst difference from each alone %.6f (bound %.3f)", len(reqs), worst, tol)
					if !(worst <= tol) {
						t.Errorf("batched, the answers moved %.6f from each alone", worst)
					}
					if c.kind == jlm.DecisionClef || c.kind == jlm.DecisionKev {
						// Clef and Kev answer a request in passes of their own (its
						// head reads every row, which a batch session's rows do
						// not give back). The violation: the State not reset
						// between the batch's requests, so each runs after the
						// last one's rows.
						d.violation = "no-reset"
						vio, err := d.DecideBatch(reqs)
						d.violation = ""
						if err != nil {
							t.Fatal(err)
						}
						seen := 0.0
						for i := range reqs {
							if alone[i] != nil {
								seen = max(seen, decisionDiffRow(t, [][]DecisionAnswer{alone[i]}, [][]DecisionAnswer{vio[i].Answers}))
							}
						}
						t.Logf("requests run without a reset between them: the answers move %.5f", seen)
						if seen <= tol {
							t.Errorf("no reset between requests moved the answers %.5f: the gate does not see it", seen)
						}
						return
					}
					// The violation: a decoder's prompt chunks laid out at their
					// positions reversed, so a row's history is not the prompt's
					// (Scheduler.chunkPos); an encoder's packed sequences
					// attending across one another.
					if m.IsEncoder() {
						d.emb.violation = "sequences-attend-across"
						defer func() { d.emb.violation = "" }()
					} else if d.nonCausal() {
						// A non-causal model's prompts run alone; the
						// violation runs them as rows of a batch session,
						// where they attend causally.
						d.violation = "batch-alone"
						defer func() { d.violation = "" }()
					} else {
						d.batch.chunkPos = func(base, j, k int) int { return base + k - 1 - j }
						defer func() { d.batch.chunkPos = nil }()
					}
					vio, err := d.DecideBatch(reqs)
					if err != nil {
						t.Logf("the violation failed the batch: %v", err)
						return
					}
					seen := 0.0
					for i := range reqs {
						if alone[i] != nil {
							seen = max(seen, decisionDiffRow(t, [][]DecisionAnswer{alone[i]}, [][]DecisionAnswer{vio[i].Answers}))
						}
					}
					t.Logf("the violation moves the answers %.5f", seen)
					if seen <= max(tol, decisionSeen) {
						t.Errorf("the violation moved the answers %.5f: the gate does not see it", seen)
					}
				})
			}
		})
	}
}
