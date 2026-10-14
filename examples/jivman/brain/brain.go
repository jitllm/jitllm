// Package brain is jitllm embedded: a decision model opened in this process,
// placed on the best device the machine has, answering jivman's System One
// requests through the engine's Decider -- the same path jitllm's
// /v1/systemone and `jitllm decide` take.
package brain

import (
	"cmp"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// Entry is a decision model found in the models folder.
type Entry struct {
	Path string
	// Name is how the picker shows it: the file's name without .jlm.
	Name string
	// Readout is the container's decision readout (laya, lev, lfm2-d1, clef).
	Readout string
	Bytes   int64
}

// List finds every decision model under dir: each .jlm whose container states
// a decision readout. Only the containers' metadata is read. Smallest first.
func List(dir string) []Entry {
	var out []Entry
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			// An unreadable folder is skipped, not the whole walk.
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			if p != dir && strings.Count(strings.TrimPrefix(p, dir), string(filepath.Separator)) > 3 {
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".jlm") {
			return nil
		}
		if e, err := EntryOf(p); err == nil {
			out = append(out, e)
		}
		return nil
	})
	slices.SortFunc(out, func(a, b Entry) int { return cmp.Compare(a.Bytes, b.Bytes) })
	// Two copies of one file under two folders are one model to the picker.
	return slices.CompactFunc(out, func(a, b Entry) bool { return a.Name == b.Name && a.Bytes == b.Bytes })
}

// EntryOf reads the container at p's metadata (no page) and describes it, or
// says why it is not a decision model.
func EntryOf(p string) (Entry, error) {
	f, err := jlm.Open(p)
	if err != nil {
		return Entry{}, err
	}
	kind := f.Config().Decision
	f.Close()
	if kind == jlm.DecisionNone {
		return Entry{}, fmt.Errorf("%s states no decision readout: it is not a decision model", p)
	}
	info, err := os.Stat(p)
	if err != nil {
		return Entry{}, err
	}
	return Entry{Path: p, Name: strings.TrimSuffix(filepath.Base(p), ".jlm"), Readout: kind.String(), Bytes: info.Size()}, nil
}

// Brain is one opened decision model.
type Brain struct {
	Entry
	// Device names where the blocks run ("cpu" when no device took any).
	Device string
	// Blocks is how many blocks the device holds.
	Blocks int

	m        *model.Model
	d        *model.Decider
	closeDev func()
}

// Open loads e and places it: devices is a jitllm device spec ("auto" takes
// the best card there is and falls back to the host, "cpu" stays home).
func Open(e Entry, devices string) (*Brain, error) {
	m, err := model.Open(e.Path)
	if err != nil {
		return nil, err
	}
	if m.Decision() == jlm.DecisionNone {
		m.Close()
		return nil, fmt.Errorf("%s is not a decision model", e.Path)
	}
	// The longest prompt one question may render to; jivman's are near two
	// thousand tokens with the board in them.
	d, err := m.NewDecider(8192)
	if err != nil {
		m.Close()
		return nil, err
	}
	b := &Brain{Entry: e, Device: "cpu", m: m, d: d, closeDev: func() {}}
	if devices != "" && devices != "cpu" {
		g, err := tier.OpenWith(tier.WithDevices(devices), tier.WithHostBudget(sched.MemBudget()))
		switch {
		case err == nil:
			if err := d.SetDevice(g, -1); err != nil {
				g.Close()
				if devices != "auto" {
					b.Close()
					return nil, err
				}
				b.Device = "cpu (" + err.Error() + ")"
				break
			}
			b.closeDev, b.Blocks = g.Close, d.DeviceBlocks()
			b.Device = g.Name()
			if b.Blocks == 0 {
				b.Device = "cpu (no block fits " + g.Name() + ")"
			}
		case devices == "auto":
			// auto is the one spec for which a missing card is not an error.
		default:
			b.Close()
			return nil, err
		}
	}
	return b, nil
}

// Close releases the decider, the device and the model.
func (b *Brain) Close() {
	if b.d != nil {
		b.d.SetDevice(nil, 0)
		b.d.Close()
	}
	b.closeDev()
	b.m.Close()
}

// Answer is one question's answer.
type Answer struct {
	ID     string
	Choice string
	// Probs is per option in the order the request wrote them, Options the
	// option keys in that order.
	Options []string
	Probs   []float64
	// Confidence is the model's own, when its readout has one.
	Confidence float64
}

// Call is what one request cost: the model's time and the prompt tokens it
// read (every question's, summed).
type Call struct {
	Took   time.Duration
	Tokens int
}

// Decide answers a System One request body, as /v1/systemone would. The
// answers are copies: the Decider reuses its own buffers on the next call.
func (b *Brain) Decide(body []byte, out []Answer) ([]Answer, Call, error) {
	req, err := model.ParseDecisionRequest(body)
	if err != nil {
		return out, Call{}, err
	}
	state, qs := req.State, req.Questions
	t := time.Now()
	ans, err := b.d.Decide(state, qs, nil)
	c := Call{Took: time.Since(t), Tokens: b.d.InputTokens()}
	if err != nil {
		return out, c, err
	}
	out = out[:0]
	for i, q := range qs {
		a := Answer{ID: q.ID, Choice: ans[i].Choice, Confidence: ans[i].Confidence, Probs: slices.Clone(ans[i].Probs)}
		for _, o := range q.Options {
			a.Options = append(a.Options, o.Key)
		}
		out = append(out, a)
	}
	return out, c, nil
}
