package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// TestPlanCoversEveryModel runs every model TestEveryModelRunsGenerated
// audits (auditModels, the same size rule, JITLLM_SLOW=1 admitting every
// size; JITLLM_PLAN_MODEL a substring narrowing them) through the composed plan on each device a
// backend opens, against the host, its KV cache f32 as the plan's pools are.
// Per model it reports whether main opens it, whether the composer forms it
// or the op that refuses it, each block's residuals at position 0 against
// the host's traced ones -- a wrong kernel shows at block 0, an amplified
// rounding grows with the depth (RULE 11c) -- and the logits over a prompt.
//
// It fails a model the composer forms whose first block is wrong outright: a
// residual off by more than 1e-3 or not finite.
func TestPlanCoversEveryModel(t *testing.T) {
	pick := os.Getenv("JITLLM_PLAN_MODEL")
	slow := os.Getenv("JITLLM_SLOW") != ""
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	var mainRefuses, refused, formed, wrong []string
	refusals := map[string]int{}
	for _, src := range auditModels(t) {
		if pick != "" && !strings.Contains(src.path, pick) || !slow && src.size > 2<<30 {
			continue
		}
		path := src.path
		if src.hf {
			path = jlmOfSafetensors(t, src.path)
		} else {
			p, err := jlmOfErr(src.path)
			if err != nil {
				if !errors.Is(err, convert.ErrNotImplemented) {
					mainRefuses = append(mainRefuses, src.name)
				}
				continue
			}
			path = p
		}
		m, err := Open(path, WithJITOptions(nn.WithTune(nn.TuneOff)), WithKVType(KVF32))
		if err != nil {
			mainRefuses = append(mainRefuses, src.name)
			continue
		}
		cover := planCover
		switch {
		case m.Decision() == jlm.DecisionLaya || m.Decision() == jlm.DecisionD1Omni:
			cover = decideCover
		case m.IsEncoder() || m.Cfg.Pooling != jlm.PoolNone:
			cover = encodeCover
		}
		row, bad, why := cover(t, m, devs, gpu)
		m.Close()
		switch {
		case why != "":
			refused = append(refused, src.name)
			refusals[why]++
			t.Logf("%-40s refused: %s", src.name, why)
		case bad:
			wrong = append(wrong, src.name)
			t.Errorf("%-40s %s", src.name, row)
		default:
			formed = append(formed, src.name)
			t.Logf("%-40s %s", src.name, row)
		}
	}
	t.Logf("of %d models main opens, %d form and decode, %d form and are wrong, %d are refused; main refuses %d",
		len(formed)+len(wrong)+len(refused), len(formed), len(wrong), len(refused), len(mainRefuses))
	for why, n := range refusals {
		t.Logf("  %3d refused: %s", n, why)
	}
}

// planCover is one model's row: the first block's and the last block's
// residuals at position 0 and the worst logits over a prompt, per device;
// bad where a block or the prompt's logits are wrong outright; why, the
// composer's refusal.
func planCover(t *testing.T, m *Model, devs []backend.Device, gpu jit.Device) (row string, bad bool, why string) {
	// Long enough that a block selection selects (MiniMax-M3's fixture
	// keeps 2 blocks of 4) and short of the runner's 32 positions.
	ids := m.Vocab.Encode("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is", true)
	if len(ids) > 30 {
		ids = ids[:30]
	}
	traced := map[string][]float32{}
	m.Trace(func(l int, name string, v []float32) {
		k := fmt.Sprint(l, name)
		if _, ok := traced[k]; !ok {
			traced[k] = append([]float32(nil), v...)
		}
	})
	host := m.NewState(32)
	var want [][]float32
	for _, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), l...))
		m.Trace(nil)
	}
	host.Close()
	last := m.Cfg.NLayer - 1
	for _, dev := range devs {
		r, err := run.New(m.Composing(), dev, gpu, 32)
		if err != nil {
			return "", false, strings.SplitN(err.Error(), "\n", 2)[0]
		}
		s, err := r.NewSession()
		if err != nil {
			t.Fatal(err)
		}
		plan := r.Plan()
		var codes []struct {
			part int
			c    *jit.Code
		}
		for pi := range plan.Parts {
			for ci := range plan.Parts[pi].Codes {
				codes = append(codes, struct {
					part int
					c    *jit.Code
				}{pi, &plan.Parts[pi].Codes[ci]})
			}
		}
		// Each block teacher-forced: its input the host's own residual,
		// written after the part before it, and its output held to the
		// host's, so no block inherits another's error.
		lastOf := map[int]int{}
		for i, cd := range codes {
			lastOf[cd.part] = i
		}
		// A part's residual is its graph's stated output, whatever it is
		// named (Gemma 4's is the scaled x_out).
		residOf := func(part int) (int, bool) {
			g := plan.Parts[part].Part.Graph
			if len(g.Out) == 0 {
				return 0, false
			}
			for i := lastOf[part]; i >= 0; i-- {
				for _, b := range codes[i].c.Binds {
					if b.Writes && b.Operand.Kind == jit.OperandValue && b.Operand.Value == g.Out[0] && codes[i].part == part {
						return b.Storage, true
					}
				}
			}
			return 0, false
		}
		// A block's residual: after its FFN, or after its mixer where it
		// has no FFN (Mamba-2's).
		resid := func(li int) []float32 {
			if h := traced[fmt.Sprint(li, "ffn_resid")]; h != nil {
				return h
			}
			return traced[fmt.Sprint(li, "attn_resid")]
		}
		hostIn := func(li int) []float32 {
			if li == 0 {
				// After a learned position row where the model has one.
				if h := traced[fmt.Sprint(-1, "embd_pos")]; h != nil {
					return h
				}
				return traced[fmt.Sprint(-1, "embd")]
			}
			return resid(li - 1)
		}
		// A seam a kernel spans -- part p's kernel writing what part p+1
		// reads, as a coarsened chain quantizes the next block's input with
		// the residual it writes -- cannot take the host's residual between
		// them: the block after it runs on the plan's own, and its error is
		// the run's, reported apart and bounded through the logits.
		fused := map[int]bool{}
		for _, cd := range codes {
			for _, b := range cd.c.Binds {
				if b.Writes && (b.Operand.Kind == jit.OperandValue || b.Operand.Kind == jit.OperandScratch) && b.Operand.Of > cd.part {
					fused[b.Operand.Of] = true
				}
			}
		}
		worstBlock, worstAt, worstFused := 0.0, -1, 0.0
		if err := s.StepEach(ids[0], 0, func(i int) {
			p := codes[i].part
			if i != lastOf[p] {
				return
			}
			blockOf := func(p int) int {
				if p < len(plan.Parts) && plan.Parts[p].Part.Kind == jit.PartBlock {
					return plan.Parts[p].Part.Block
				}
				return -1
			}
			if li := blockOf(p); li >= 0 && li <= last {
				if st, ok := residOf(p); ok {
					h := resid(li)
					if g := s.Storage(st); h != nil && len(g) >= 4*len(h) {
						x := make([]float32, len(h))
						for k := range x {
							x[k] = math.Float32frombits(binary.LittleEndian.Uint32(g[4*k:]))
						}
						switch e := nmse(x, h); {
						case fused[p]:
							worstFused = max(worstFused, e)
						case !(e <= worstBlock):
							worstBlock, worstAt = e, li
						}
					}
				}
			}
			if li := blockOf(p + 1); li >= 0 && li <= last {
				if st, ok := residOf(p); ok && hostIn(li) != nil {
					in := hostIn(li)
					b := make([]byte, 4*len(in))
					for k, v := range in {
						binary.LittleEndian.PutUint32(b[4*k:], math.Float32bits(v))
					}
					if err := s.SetStorage(st, b); err != nil {
						t.Fatal(err)
					}
				}
			}
		}); err != nil {
			t.Fatal(err)
		}
		// The untouched decode is a sequence of its own: a recurrent state
		// has taken the teacher-forced step, which a history's position
		// would only have rewritten.
		s.Close()
		if s, err = r.NewSession(); err != nil {
			t.Fatal(err)
		}
		worst, agree := 0.0, 0
		for pos, id := range ids {
			l, tok, err := s.Step(id, pos)
			if err != nil {
				t.Fatal(err)
			}
			worst = max(worst, nmse(l, want[pos]))
			// A flip between two tokens the host holds within a quarter of a
			// logit of each other is the arithmetic's, not a disagreement.
			if b := argmaxOf(want[pos]); int(tok) == b || want[pos][b]-want[pos][tok] < 0.25 {
				agree++
			}
		}
		row += fmt.Sprintf("%s: worst block %.2g (block %d of %d, teacher-forced), logits %.2g, top-1 %d/%d; ", dev.API(), worstBlock, worstAt, last+1, worst, agree, len(ids))
		if len(fused) > 0 {
			row += fmt.Sprintf("%d seams fused, worst block past one %.2g; ", len(fused), worstFused)
		}
		// The prompt's logits too: a block right at position 0 says nothing
		// of a history or a selection read past it. A quantized model's
		// device and host disagree by its activations' rounding -- main's
		// own device reads OLMo-2-1B at 0.115 against its host -- so the
		// logits are bounded loosely and the top-1 held at all but two
		// positions (a flip within a quarter of a logit agrees); a selection
		// that keeps every block reads 0.51 and 13 of 19.
		if !(worstBlock <= 1e-3) || worstAt < 0 || !(worst <= 0.25) || agree < len(ids)-2 {
			bad = true
		}
		s.Close()
		r.Close()
	}
	return row, bad, ""
}

func nmse(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return num / den
}

// encodeCover is an encoder's row: its composed embedding of a text, one
// step of every row, held to the host's Embedder per device. An encoder
// that pools nothing (a decision model's) is refused.
func encodeCover(t *testing.T, m *Model, devs []backend.Device, gpu jit.Device) (row string, bad bool, why string) {
	if m.Cfg.Pooling == jlm.PoolNone {
		return "", false, "an encoder with no pooling: plan mode composes no decision head"
	}
	// Long enough to pass EmbeddingGemma's window, 256 either side.
	ids := m.EmbedIDs(strings.Repeat("The capital of France is Paris, and the capital of Germany is Berlin. ", 20))
	e, err := m.NewEmbedder()
	if err != nil {
		t.Fatal(err)
	}
	if n := e.MaxTokens(); n > 0 && len(ids) > n {
		ids = ids[:n]
	}
	want, err := e.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	e.Close()
	for _, dev := range devs {
		r, err := run.New(m.Composing(), dev, gpu, len(ids), run.WithRows(len(ids)))
		if err != nil {
			return "", false, strings.SplitN(err.Error(), "\n", 2)[0]
		}
		s, err := r.NewSession()
		if err != nil {
			t.Fatal(err)
		}
		got, err := s.Encode(ids, m.IsEncoder() || m.Cfg.NonCausal)
		if err != nil {
			t.Fatal(err)
		}
		e := nmse(got, want)
		row += fmt.Sprintf("%s: embedding of %d tokens %.2g; ", dev.API(), len(ids), e)
		// A Q8_0 encoder's int8 activations on a device that takes them
		// reach 1e-4 (MiniLM-L6-v2 on CUDA), about 0.99995 in cosine.
		if !(e <= 1e-3) {
			bad = true
		}
		s.Close()
		r.Close()
	}
	return row, bad, ""
}

// decideCover is a Laya decision model's row: its composed scores at three
// markers of a text past its window, for each question type, held to the
// host's mbEncode and layaScores per device.
func decideCover(t *testing.T, m *Model, devs []backend.Device, gpu jit.Device) (row string, bad bool, why string) {
	var ids []int32
	for k := 1; len(ids) <= m.Cfg.SWAWindow+8; k++ {
		ids = m.Vocab.Encode(strings.Repeat("Is the capital of France Paris? ", k)+"Answer with one of the options below.", true)
	}
	dc, err := m.NewDecider(len(ids) + 8)
	if err != nil {
		t.Fatal(err)
	}
	defer dc.Close()
	markers := []int32{1, int32(len(ids) / 2), int32(len(ids) - 2)}
	rows := []int{int(markers[0]), int(markers[1]), int(markers[2])}
	for _, dev := range devs {
		r, err := run.New(m.Composing(), dev, gpu, len(ids), run.WithRows(len(ids)))
		if err != nil {
			return "", false, strings.SplitN(err.Error(), "\n", 2)[0]
		}
		s, err := r.NewSession()
		if err != nil {
			t.Fatal(err)
		}
		worst := 0.0
		for qt := range 3 {
			if err := hostEncode(dc.emb, ids); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, len(markers))
			if err := dc.emb.layaScores(want, len(ids), jlm.QuestionType(qt), rows); err != nil {
				t.Fatal(err)
			}
			got, err := s.Decide(ids, qt, markers)
			if err != nil {
				t.Fatal(err)
			}
			worst = max(worst, nmse(got, want))
		}
		row += fmt.Sprintf("%s: decision scores of %d tokens %.2g; ", dev.API(), len(ids), worst)
		if !(worst <= 1e-6) {
			bad = true
		}
		s.Close()
		r.Close()
	}
	return row, bad, ""
}

// hostEncode is the host's encoder pass over ids for a decision model with
// Laya's head: an LFM2 encoder's or ModernBERT's.
func hostEncode(e *Embedder, ids []int32) error {
	if e.m.enc.lf != nil {
		return e.lfm2Encode(ids, nil)
	}
	return e.mbEncode(ids)
}
