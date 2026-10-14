package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// A request's questions share their trunk: everything a label readout's
// prompt holds before the question -- the system turn, the pictures, the
// state -- is the same text for every question and variant, and only the tail
// differs. The references read the trunk once (d1's runner packs a tree of a
// trunk and its branches; lev's server shares the prefix), and so does this:
// the trunk is prefilled once, each question's tail runs on from it, and the
// sequence is taken back to the trunk's end between tails.
//
// Taking it back is what speculative decoding's rollback does: attention
// forgets the positions past the trunk by position (State.rewind), and a
// hybrid's recurrent summary -- which has no position -- is put back from a
// copy taken at the trunk's end. The copy is the summary's size (a short
// convolution's window, a delta rule's state per head), not the trunk's KV,
// which is held once. A block placed on a device has its summary brought home
// for the copy and pushed back after each tail (syncRecFromDevice,
// pushRecToDevice), the paths the prefix cache already restores through.
//
// The answers are the same arithmetic as a prompt run whole -- the trunk's
// rows are the same rows -- up to the chunking of the prefill: a prompt run
// whole is one chunk where the tail now is a chunk of its own, and the f32
// reductions then differ in order. The gate holds the two together
// (TestDecisionTrunkMatchesWhole).

// decidePrompt is one question-and-variant's prompt of a request, and what
// its readout reads.
type decidePrompt struct {
	q, v   int
	spans  []Span
	groups [][]int32
}

// trunkLen is how many leading positions every prompt shares, whole spans and
// the common head of the first token span where they part; a picture span
// counts when it is the same picture in every prompt. Every prompt keeps at
// least one position after the trunk, which its logits are read at.
func trunkLen(ps []decidePrompt, nembd int) int {
	if len(ps) < 2 {
		return 0
	}
	first := ps[0].spans
	n := 0
	for si := 0; si < len(first); si++ {
		sp := first[si]
		if sp.Picture != nil || sp.Embd != nil {
			for _, p := range ps[1:] {
				if si >= len(p.spans) || p.spans[si].Picture != sp.Picture || sp.Picture == nil {
					return n
				}
			}
			n += sp.n(nembd)
			continue
		}
		k := len(sp.Tokens)
		for _, p := range ps[1:] {
			if si >= len(p.spans) || p.spans[si].Picture != nil || p.spans[si].Embd != nil {
				return n
			}
			t := p.spans[si].Tokens
			j := 0
			for j < min(k, len(t)) && t[j] == sp.Tokens[j] {
				j++
			}
			k = j
		}
		n += k
		if k < len(sp.Tokens) {
			break
		}
		for _, p := range ps[1:] {
			if len(p.spans[si].Tokens) != k {
				return clampTrunk(n, ps, nembd)
			}
		}
	}
	return clampTrunk(n, ps, nembd)
}

// clampTrunk leaves every prompt one position past the trunk.
func clampTrunk(n int, ps []decidePrompt, nembd int) int {
	for _, p := range ps {
		n = min(n, SpanPositions(p.spans, nembd)-1)
	}
	return max(n, 0)
}

// splitSpans cuts spans at position n: the spans before it and after it, a
// token span cut in two where n falls inside it. n never falls inside a
// picture (trunkLen counts pictures whole).
func splitSpans(spans []Span, n, nembd int) (head, tail []Span) {
	at := 0
	for i, sp := range spans {
		k := sp.n(nembd)
		switch {
		case at+k <= n:
			head = append(head, sp)
		case at >= n:
			tail = append(tail, spans[i:]...)
			return head, tail
		default:
			head = append(head, Span{Tokens: sp.Tokens[:n-at]})
			tail = append(tail, Span{Tokens: sp.Tokens[n-at:]})
			tail = append(tail, spans[i+1:]...)
			return head, tail
		}
		at += k
	}
	return head, tail
}

// trunkShared reports whether a request's prompts run from one trunk: a label
// readout (lev, d1) with more than one prompt, unless a gate turns it off.
func (d *Decider) trunkShared(qs []DecisionQuestion) bool {
	if d.kind != jlm.DecisionLev && d.kind != jlm.DecisionLFM2D1 || d.violation == "no-trunk" {
		return false
	}
	n := 0
	for i := range qs {
		n += d.variants(&qs[i])
	}
	return n > 1
}

// decideTrunk answers qs running their shared trunk once. It reports false,
// with nothing answered, when the prompts share no trunk or the summary
// cannot be brought home; the caller then runs each prompt whole.
func (d *Decider) decideTrunk(state jinja.Value, qs []DecisionQuestion, out []DecisionAnswer) (bool, error) {
	ps := d.prompts[:0]
	for i := range qs {
		for v := range d.variants(&qs[i]) {
			var p decidePrompt
			var err error
			p.q, p.v = i, v
			if p.spans, p.groups, err = d.promptSpans(state, &qs[i], v); err != nil {
				return false, fmt.Errorf("questions.%s: %w", qs[i].ID, err)
			}
			ps = append(ps, p)
		}
	}
	d.prompts = ps
	nembd := d.m.Cfg.NEmbd
	t := trunkLen(ps, nembd)
	if t == 0 {
		return false, nil
	}
	st := d.st
	st.Reset()
	trunk, _ := splitSpans(ps[0].spans, t, nembd)
	if err := d.prefillSpans(trunk); err != nil {
		return false, err
	}
	failed := st.kv.syncFailed
	if err := st.syncRecFromDevice(); err != nil {
		return false, err
	}
	if st.kv.syncFailed != failed {
		return false, nil
	}
	d.saveTrunkRec()
	scores := make([][][]float32, len(qs))
	for _, p := range ps {
		_, tail := splitSpans(p.spans, t, nembd)
		s, err := d.tailScores(tail, p.groups, p.v)
		if err != nil {
			return false, fmt.Errorf("questions.%s: %w", qs[p.q].ID, err)
		}
		scores[p.q] = append(scores[p.q], slices.Clone(s))
		d.input += SpanPositions(p.spans, nembd)
		st.rewind(t)
		if d.violation != "trunk-no-rec" {
			d.loadTrunkRec()
		}
		if !st.pushRecToDevice() {
			return false, fmt.Errorf("model: the trunk's recurrent summary could not go back to the device")
		}
	}
	for i := range qs {
		a, err := d.answer(&qs[i], scores[i])
		if err != nil {
			return false, fmt.Errorf("questions.%s: %w", qs[i].ID, err)
		}
		out[i] = a
	}
	d.trunks++
	return true, nil
}

// prefillSpans runs spans on from the sequence's position.
func (d *Decider) prefillSpans(spans []Span) error {
	var err error
	if len(spans) == 1 && spans[0].Picture == nil && spans[0].Embd == nil {
		_, err = d.st.Prefill(spans[0].Tokens)
	} else {
		_, err = d.st.PrefillMixed(spans...)
	}
	return err
}

// saveTrunkRec copies every linear block's summary -- the host's, which
// syncRecFromDevice has brought a placed block's into -- and loadTrunkRec puts
// it back.
func (d *Decider) saveTrunkRec() {
	s := d.st
	if s.rconv == nil {
		return
	}
	if d.trunkRec.conv == nil {
		d.trunkRec = s.newRecSnap()
	}
	d.copyTrunkRec(true)
}

func (d *Decider) loadTrunkRec() {
	if d.st.rconv == nil {
		return
	}
	d.copyTrunkRec(false)
}

func (d *Decider) copyTrunkRec(save bool) {
	s := d.st
	g := s.dg
	nc, ns := g.convStateLen(), g.stateStride()
	for li := range s.rconv {
		if s.rconv[li] == nil || !s.c.LayerKind(li).Recurrent() {
			continue
		}
		if save {
			copy(d.trunkRec.conv[li], s.rconv[li][:nc])
			copy(d.trunkRec.state[li], s.rstate[li][:ns])
		} else {
			copy(s.rconv[li][:nc], d.trunkRec.conv[li])
			copy(s.rstate[li][:ns], d.trunkRec.state[li])
		}
	}
}

// tailScores runs a prompt's tail on from the trunk and reads its label
// groups' scores into variant v's scratch, as forwardSpans does a whole
// prompt.
func (d *Decider) tailScores(tail []Span, groups [][]int32, v int) ([]float32, error) {
	return d.readSpans(tail, groups, v, false)
}
