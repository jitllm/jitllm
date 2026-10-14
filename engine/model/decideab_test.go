package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDecisionDeviceAB is the paired, interleaved A/B harness for the
// decision readouts and the embedders on a device (RULE 2): it measures, it
// asserts nothing (RULE 4), and it runs only when JITLLM_DECIDE_AB names the
// arms, so a plain `go test` never takes a timing.
//
//	JITLLM_DECIDE_AB    comma list of arms: aa-host, aa-dev, host-dev, pick,
//	                    trunk-host and trunk-dev (every prompt whole against
//	                    the shared trunk), batch-host and batch-dev (both
//	                    requests as one DecideBatch, the server's path, against
//	                    each request's trunk in turn)
//	JITLLM_AB_MODELS    comma list of name=path (a .gguf or .jlm); a name
//	                    starting "embed" is an embedder, embedding JITLLM_AB_TEXT
//	JITLLM_AB_DEVICE    the device spec (default cuda:0)
//	JITLLM_AB_ROUNDS    paired rounds (default 20), after JITLLM_AB_SOAK (default 3)
//
// One round runs both arms once each, the order alternating by round
// (ABBA...); a sample is both requests in testdata/decision (or one embedding),
// and the ratio is B's time over A's. It reports the median of the per-round
// ratios with IQR/median, the rejection RULE 2 sets at 0.10, and each arm's
// median latency per question. The pick arm also counts the bytes the label
// readback moves per question, against the whole logit row the no-pick arm
// reads back.
func TestDecisionDeviceAB(t *testing.T) {
	arms := os.Getenv("JITLLM_DECIDE_AB")
	if arms == "" {
		t.Skip("a measurement harness: JITLLM_DECIDE_AB names its arms")
	}
	spec := os.Getenv("JITLLM_AB_DEVICE")
	if spec == "" {
		spec = "cuda:0"
	}
	rounds, soak := envInt("JITLLM_AB_ROUNDS", 20), envInt("JITLLM_AB_SOAK", 3)
	text := os.Getenv("JITLLM_AB_TEXT")
	if text == "" {
		text = strings.Repeat("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget. ", 4)
	}
	for _, mp := range strings.Split(os.Getenv("JITLLM_AB_MODELS"), ",") {
		name, path, ok := strings.Cut(mp, "=")
		if !ok {
			t.Fatalf("JITLLM_AB_MODELS entry %q: want name=path", mp)
		}
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			g, err := tier.OpenWith(tier.WithDevices(spec))
			if err != nil || g == nil {
				t.Fatalf("%s: %v", spec, err)
			}
			defer g.Close()
			embed := strings.HasPrefix(name, "embed")
			for _, arm := range strings.Split(arms, ",") {
				if arm == "pick" && (embed || m.IsEncoder()) {
					continue
				}
				t.Run(arm, func(t *testing.T) {
					var a, b abArm
					switch arm {
					case "aa-host":
						a, b = newABArm(t, m, embed, nil, ""), newABArm(t, m, embed, nil, "")
					case "aa-dev":
						a, b = newABArm(t, m, embed, g, ""), newABArm(t, m, embed, g, "")
					case "host-dev":
						a, b = newABArm(t, m, embed, nil, ""), newABArm(t, m, embed, g, "")
					case "pick":
						a, b = newABArm(t, m, embed, g, "no-pick"), newABArm(t, m, embed, g, "")
					case "trunk-host":
						a, b = newABArm(t, m, embed, nil, "no-trunk"), newABArm(t, m, embed, nil, "")
					case "trunk-dev":
						a, b = newABArm(t, m, embed, g, "no-trunk"), newABArm(t, m, embed, g, "")
					case "batch-host", "batch-dev":
						var dev *tier.GPU
						if arm == "batch-dev" {
							dev = g
						}
						a, b = newABArm(t, m, embed, dev, ""), newABArm(t, m, embed, dev, "")
						a.batch = true
					default:
						t.Fatalf("arm %q", arm)
					}
					defer a.close()
					defer b.close()
					t.Logf("%s on %s: arm A %d device blocks, arm B %d, of %d", name, spec, a.blocks(), b.blocks(), decisionBlocks(m))
					var ratios, la, lb []float64
					for r := 0; r < soak+rounds; r++ {
						first, second := &a, &b
						if r%2 == 1 {
							first, second = &b, &a
						}
						d1 := first.sample(t, text)
						d2 := second.sample(t, text)
						if r < soak {
							continue
						}
						da, db := d1, d2
						if r%2 == 1 {
							da, db = d2, d1
						}
						ratios = append(ratios, db.Seconds()/da.Seconds())
						la = append(la, da.Seconds()/float64(a.n))
						lb = append(lb, db.Seconds()/float64(b.n))
					}
					med, iqr := medIQR(ratios)
					ma, _ := medIQR(la)
					mb, _ := medIQR(lb)
					verdict := "ok"
					if iqr/med > 0.10 {
						verdict = "REJECT (IQR/median > 0.10)"
					}
					t.Logf("RESULT %s %s %s: B/A time ratio median %.4f IQR/median %.4f n=%d %s; per question A %.3f ms (%.1f q/s), B %.3f ms (%.1f q/s)",
						name, arm, spec, med, iqr/med, len(ratios), verdict, ma*1e3, 1/ma, mb*1e3, 1/mb)
					if arm == "pick" {
						vocab := m.Cfg.NVocab
						t.Logf("RESULT %s pick readback: %d prompts picked, %d labels a prompt, %d B a prompt picked against %d B for the %d-wide logit row; per question %.0f B against %.0f B",
							name, b.d.picked, len(b.d.pick), 4*len(b.d.pick), 4*vocab, vocab,
							float64(4*len(b.d.pick)*b.prompts)/float64(b.n), float64(4*vocab*a.prompts)/float64(a.n))
					}
				})
			}
		})
	}
}

// abArm is one arm: a Decider, or an Embedder, placed on dev (nil: the host).
type abArm struct {
	d       *Decider
	e       *Embedder
	m       *Model
	n       int // questions (or texts) answered, measured rounds and soak alike
	prompts int // prompts run, so a per-question byte count can be made
	// batch answers both requests as one DecideBatch, the server's path.
	batch bool
}

func newABArm(t *testing.T, m *Model, embed bool, g *tier.GPU, violation string) abArm {
	t.Helper()
	a := abArm{m: m}
	if embed {
		e, err := m.NewEmbedder()
		if err != nil {
			t.Fatal(err)
		}
		a.e = e
		if g != nil {
			if err := e.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
		}
		return a
	}
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	d.violation = violation
	a.d = d
	if g != nil {
		if err := d.SetDevice(g, -1); err != nil {
			t.Fatal(err)
		}
	}
	return a
}

func (a *abArm) blocks() int {
	if a.e != nil {
		return a.e.DeviceBlocks()
	}
	return a.d.DeviceBlocks()
}

func (a *abArm) close() {
	if a.e != nil {
		a.e.SetDeviceLayers(nil, 0)
		a.e.Close()
		return
	}
	a.d.SetDevice(nil, 0)
	a.d.Close()
}

// sample answers both decision requests (or embeds text once) and returns the
// time it took.
func (a *abArm) sample(t *testing.T, text string) time.Duration {
	t.Helper()
	if a.e != nil {
		ids := a.m.EmbedIDs(text)
		t0 := time.Now()
		if _, err := a.e.Embed(ids); err != nil {
			t.Fatal(err)
		}
		a.n++
		return time.Since(t0)
	}
	var total time.Duration
	if a.batch {
		var reqs []DecisionRequest
		for _, req := range decisionRequests {
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
			reqs = append(reqs, DecisionRequest{State: state, Questions: qs})
			a.n += len(qs)
			a.prompts += a.d.promptsRun(qs)
		}
		t0 := time.Now()
		out, err := a.d.DecideBatch(reqs)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range out {
			if r.Err != nil {
				t.Fatal(r.Err)
			}
		}
		return time.Since(t0)
	}
	for _, req := range decisionRequests {
		state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
		t0 := time.Now()
		if _, err := a.d.Decide(state, qs, nil); err != nil {
			t.Fatal(err)
		}
		total += time.Since(t0)
		a.n += len(qs)
		a.prompts += a.d.promptsRun(qs)
	}
	return total
}

// promptsRun is how many prompts Decide runs for qs: one per variant of each
// question.
func (d *Decider) promptsRun(qs []DecisionQuestion) int {
	n := 0
	for i := range qs {
		n += d.variants(&qs[i])
	}
	return n
}
