package engine

import (
	"image"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
)

// TestAPictureStepsBesideDecode is batching with a picture for every stated
// tower: in StepRuns a session's prompt carrying a picture prefills beside two
// sessions decoding, and then the three decode together. Teacher-forced
// against each session alone on the same device: every logit row within the
// step band, the picture's session's from its prompt's end. It selects what
// it gates -- the token rows ran in steps across sessions -- and a grey
// picture in the stepped arm alone parts the picture's session from its arm
// alone past the band (the violation): the comparison sees the picture. On
// the CPU device and on every card holding every block and the head.
func TestAPictureStepsBesideDecode(t *testing.T) {
	img := principlePicture()
	grey := image.NewGray(img.Bounds())
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	for _, src := range towerFixtures(t) {
		t.Run(src.name, func(t *testing.T) {
			m, err := model.Open(pairContainer(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, dev := range append([]string{"cpu"}, stepDevices()...) {
				t.Run(dev, func(t *testing.T) {
					var g *gpu.GPU
					bound := hostStepNMSE
					if dev != "cpu" {
						g = stepTier(t, m, dev, false)
						defer g.Close()
						bound = mlaDeviceNMSE
					}
					p := pictureStep{early: []string{"The capital of France is", "Water boils at a temperature of"}, steps: 4}
					p.plan.opts = []Option{WithKVF16(false)}
					want := p.alone(t, m, g, img)
					var s0 gpu.Stats
					if g != nil {
						s0 = g.Stats()
					}
					got, rows, routed := p.together(t, m, g, img)
					switch {
					case g != nil:
						if n := g.Stats().SessionRows - s0.SessionRows; n != rows {
							t.Fatalf("%d token rows ran across sessions, want %d: StepRuns went one instance after another (%s)", n, rows, g.Err())
						}
					case routed != int64(rows):
						t.Fatalf("%d token rows ran in routed steps across sessions, want %d", routed, rows)
					}
					w, at := worstStep(got, want)
					if !(w <= bound) {
						t.Fatalf("a picture's prompt stepped beside decoding sessions parts from each alone: NMSE %.3e at logit row %d (bound %.0e)", w, at, bound)
					}
					bad, _, _ := p.together(t, m, g, grey)
					bw, bat := worstStep(bad, want)
					if bw <= bound {
						t.Fatalf("VIOLATION: a grey picture stepped in the prompt's place matches its session alone (NMSE %.3e): this gate cannot see the picture", bw)
					}
					t.Logf("%d token rows across sessions, %d logit rows: worst NMSE %.3e (row %d) against each alone; a grey picture %.3e (row %d)",
						rows, len(want), w, at, bw, bat)
				})
			}
		})
	}
}

// pictureStep is a schedule for StepRuns: sessions decoding from the start,
// their prompts prefilled alone, and a session whose prompt carries a picture
// prefilled in the first step, after which every session decodes together.
type pictureStep struct {
	early []string
	steps int
	plan  runsPlan
	// The teacher: each session's tokens alone, which both arms feed.
	earlyToks [][]int32
	lateToks  []int32
}

// spans is the picture's session's prompt.
func (p *pictureStep) spans(t *testing.T, m *model.Model, img image.Image) []model.Span {
	t.Helper()
	sp, err := m.PictureSpan(img)
	if err != nil {
		t.Fatal(err)
	}
	return []model.Span{{Tokens: m.Vocab.Encode("Look at this picture:", true)}, sp,
		{Tokens: m.Vocab.Encode(" The colours in it are", false)}}
}

// alone runs each session by itself on g and returns the logits the stepped
// arm must give, in its order: step by step, each early session's, then the
// picture's session's.
func (p *pictureStep) alone(t *testing.T, m *model.Model, g *gpu.GPU, img image.Image) [][]float32 {
	t.Helper()
	p.earlyToks = make([][]int32, len(p.early))
	var early [][][]float32
	for i, pr := range p.early {
		ids := m.Vocab.Encode(pr, true)
		st := p.plan.placed(t, m, g, len(ids)+p.steps+2)
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		var lgs [][]float32
		for range p.steps {
			next := Greedy(lg)
			p.earlyToks[i] = append(p.earlyToks[i], next)
			if lg, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
			lgs = append(lgs, append([]float32(nil), lg...))
		}
		early = append(early, lgs)
		st.Close()
	}
	spans := p.spans(t, m, img)
	st := p.plan.placed(t, m, g, model.SpanPositions(spans, m.Cfg.NEmbd)+p.steps+2)
	defer st.Close()
	lg, err := st.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	late := [][]float32{append([]float32(nil), lg...)}
	p.lateToks = nil
	for range p.steps - 1 {
		next := Greedy(lg)
		p.lateToks = append(p.lateToks, next)
		if lg, err = st.Forward(next); err != nil {
			t.Fatal(err)
		}
		late = append(late, append([]float32(nil), lg...))
	}
	var want [][]float32
	for s := range p.steps {
		for i := range p.early {
			want = append(want, early[i][s])
		}
		want = append(want, late[s])
	}
	return want
}

// together runs the schedule through StepRuns on g with img in the picture's
// session's prompt, teacher-forced by alone's tokens, and returns the logits
// in alone's order, the token rows the steps carried, and the rows the
// instances ran as routed steps across sessions.
func (p *pictureStep) together(t *testing.T, m *model.Model, g *gpu.GPU, img image.Image) ([][]float32, int, int64) {
	t.Helper()
	var sts []*Instance
	for _, pr := range p.early {
		ids := m.Vocab.Encode(pr, true)
		st := p.plan.placed(t, m, g, len(ids)+p.steps+2)
		defer st.Close()
		if _, err := st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		sts = append(sts, st)
	}
	spans := p.spans(t, m, img)
	ls := p.plan.placed(t, m, g, model.SpanPositions(spans, m.Cfg.NEmbd)+p.steps+2)
	defer ls.Close()
	var got [][]float32
	rows := 0
	for s := range p.steps {
		var runs []Run
		for i, st := range sts {
			runs = append(runs, Run{Instance: st, Tokens: p.earlyToks[i][s : s+1], Logits: true})
		}
		if s == 0 {
			runs = append(runs, Run{Instance: ls, Spans: spans, Logits: true})
		} else {
			runs = append(runs, Run{Instance: ls, Tokens: p.lateToks[s-1 : s], Logits: true})
		}
		out, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range runs {
			rows += len(r.Tokens)
			if out[i] == nil {
				t.Fatalf("step %d run %d: no logits", s, i)
			}
			got = append(got, append([]float32(nil), out[i]...))
		}
	}
	routed := ls.RoutedStepRows()
	for _, st := range sts {
		routed += st.RoutedStepRows()
	}
	return got, rows, routed
}
