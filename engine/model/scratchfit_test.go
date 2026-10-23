package model

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// ballast fills a card from a second context until free bytes remain, so a
// tier opened beside it sees a card the size the gate asks for: a budget
// alone is the tier's arithmetic, and the failure these gates exist for is the
// driver's out-of-memory. It returns the release.
type ballast struct {
	dev  backend.Device
	bufs []backend.Buf
}

// openBallast opens the card spec names in a context of its own, or skips: a
// unified-memory device has no card to fill.
func openBallast(t *testing.T, spec string) *ballast {
	t.Helper()
	var d backend.Device
	var err error
	switch api, sel, _ := strings.Cut(spec, ":"); api {
	case "cuda":
		ord, aerr := strconv.Atoi(sel)
		if aerr != nil {
			t.Fatalf("spec %q: %v", spec, aerr)
		}
		d, err = backend.OpenCUDA(ord)
	case "vulkan":
		d, err = backend.OpenVulkan(sel)
	default:
		t.Skipf("%s: a unified-memory device has no card to fill -- this gate proved nothing here", spec)
	}
	if err != nil || d == nil {
		noDevice(t, spec, err)
	}
	if u, ok := d.(backend.Unified); ok && u.UnifiedMemory() {
		d.Close()
		t.Skipf("%s shares the host's memory: no card to fill -- this gate proved nothing here", spec)
	}
	b := &ballast{dev: d}
	t.Cleanup(b.close)
	return b
}

// fill allocates until free bytes of the card remain, or until the card
// gives no more.
//
// The free figure is the driver's, and on a card the desktop shares it is
// more than the driver will hand out: a 4 GB card driving a display reported
// 130 MiB free and refused 60 MiB of it. A refused piece is halved, down to a
// MiB; a card that refuses even that is as full as a card gets, which is fuller
// than the gate asks for.
func (b *ballast) fill(t *testing.T, free uint64) {
	t.Helper()
	piece := uint64(256 << 20)
	for {
		now, _, err := b.dev.Mem()
		if err != nil {
			t.Fatalf("Mem: %v", err)
		}
		if now <= free {
			return
		}
		n := min(now-free, piece)
		buf, err := b.dev.Alloc(int(n))
		if err != nil {
			if piece = n / 2; piece < 1<<20 {
				t.Logf("the card refused %d bytes with %d reported free: full", n, now)
				return
			}
			continue
		}
		b.bufs = append(b.bufs, buf)
	}
}

func (b *ballast) empty() {
	for _, buf := range b.bufs {
		buf.Free()
	}
	b.bufs = nil
}

func (b *ballast) close() {
	b.empty()
	b.dev.Close()
}

// fitModel opens the gate's model, or skips naming why.
func fitModel(t *testing.T) *Model {
	t.Helper()
	const name = "Llama-3.2-1B-Instruct-Q4_K_M"
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// fitTier opens spec with cfg at a budget of limit bytes (0: the card's own).
func fitTier(t *testing.T, spec string, limit uint64, cfg func(*tier.Config)) *tier.GPU {
	t.Helper()
	if limit > 0 {
		spec = fmt.Sprintf("%s=%d", spec, limit)
	}
	g, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(cfg))
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	return g
}

// isOOM reports a driver's out-of-memory, in CUDA's or Vulkan's words.
func isOOM(s string) bool {
	s = strings.ToLower(s)
	// VkResult -2 is VK_ERROR_OUT_OF_DEVICE_MEMORY, which the Vulkan
	// backend reports by number.
	return strings.Contains(s, "out of memory") || strings.Contains(s, "out_of_memory") ||
		strings.Contains(s, "out_of_device_memory") || strings.Contains(s, "vkresult -2")
}

// slack is what a full card has left past its budget in these gates: less than
// any batched scratch, so a scratch the budget did not hold is the driver's
// out-of-memory.
const slack = 64 << 20

// TestAPromptFitsBesideAFullCard: a card that holds the whole model and not
// the prompt's scratch, filled to that by a second context, runs a 700-token
// prompt's 512-row chunk as one batched submission -- the reservation charged
// the scratch at placement, so the blocks left it room and fewer of them were
// placed. The violation is the same card without the reservation: the whole
// model goes on, the chunk's scratch is asked of a card with 64 MiB left,
// the driver says out of memory, and the chunk goes in pieces. That is
// a gemma-2-2b on a 4 GB card, built on purpose.
func TestAPromptFitsBesideAFullCard(t *testing.T) {
	m := fitModel(t)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) { promptFits(t, m, spec) })
	}
}

func promptFits(t *testing.T, m *Model, spec string) {
	b := openBallast(t, spec)
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 40)
	ids := m.Vocab.Encode(text, true)[:700]
	// Sizing: the whole model on the open card, with and without the
	// reservation.
	size := func(reserve bool) tier.Stats {
		g := fitTier(t, spec, 0, func(c *tier.Config) { c.NoScratchReserve = !reserve })
		defer g.Close()
		st := m.NewState(len(ids) + 16)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != m.Cfg.NLayer {
			t.Skipf("CARD TOO SMALL: %d of %d blocks on the open card -- this gate proved nothing here (%s)",
				st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		return g.Stats()
	}
	sr, sv := size(true), size(false)
	if sr.ReservedPrompt != 512 || sr.ScratchBytes <= sv.ScratchBytes {
		t.Fatalf("the open card reserved %d bytes for %d rows against %d unreserved: the reservation is not running",
			sr.ScratchBytes, sr.ReservedPrompt, sv.ScratchBytes)
	}
	// Room for the whole model and its decode scratch, and not the prompt's:
	// the violation fills the card, the reservation places fewer blocks.
	limit := sv.BudgetUsed + slack
	run := func(reserve bool) (tier.Stats, int, error) {
		g := fitTier(t, spec, limit, func(c *tier.Config) { c.NoScratchReserve = !reserve })
		defer g.Close()
		b.fill(t, limit+slack)
		defer b.empty()
		st := m.NewState(len(ids) + 16)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			return g.Stats(), 0, err
		}
		_, err := st.Prefill(ids)
		return g.Stats(), st.GPULayers(), err
	}
	st1, n1, err := run(true)
	t.Logf("reserved: budget %d, %d of %d blocks, scratch %d bytes for %d rows, %d split(s), err %v: %s",
		limit, n1, m.Cfg.NLayer, st1.ScratchBytes, st1.ReservedPrompt, st1.PromptSplits, err, st1.LastErr)
	if err != nil || n1 == 0 || st1.PromptSplits != 0 || st1.ReservedPrompt < 512 {
		t.Fatalf("the reserved card split or failed the prompt: %d blocks, %d split(s), %d rows reserved, err %v: %s",
			n1, st1.PromptSplits, st1.ReservedPrompt, err, st1.LastErr)
	}
	st2, n2, err := run(false)
	t.Logf("unreserved: %d of %d blocks, %d split(s), %d upload(s) refused, err %v: %s",
		n2, m.Cfg.NLayer, st2.PromptSplits, st2.NoRoom, err, st2.LastErr)
	// The violation is the card asked for what it does not have, and the
	// driver saying out of memory. Where that lands depends on what the
	// process left on the card: in a fresh process it is the chunk's scratch,
	// and the chunk goes in pieces; after TestQwenVLMatchesLlamaCppTeacherForced
	// in the same process it is the head's last planes at placement, and the
	// chunk runs whole on the room the head never took. Both are the driver
	// refusing what the budget admitted.
	if !isOOM(st2.LastErr) || st2.PromptSplits == 0 && st2.NoRoom == 0 {
		t.Fatalf("without the reservation the chunk ran whole on %d blocks (%d splits, %d uploads refused, %q): "+
			"the card was not full, so this gate does not discriminate", n2, st2.PromptSplits, st2.NoRoom, st2.LastErr)
	}
}

// TestAStepAtMaxRowsDoesNotOOMAFullCard is the server's step loop on a card
// that is full once its two sessions are placed: a 512-row step (one session's
// prompt chunk beside the other's decode token, model.MaxStepRows in all)
// needs the step's scratch and 512 rows of logits. With StepRows the device
// reserved them at placement, so the step runs as one across both sessions
// with nothing left to ask for. The violation asks the full card for them at
// the step: out of memory, and the step is an error.
func TestAStepAtMaxRowsDoesNotOOMAFullCard(t *testing.T) {
	m := fitModel(t)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) { stepFits(t, m, spec) })
	}
}

func stepFits(t *testing.T, m *Model, spec string) {
	b := openBallast(t, spec)
	prompt := m.Vocab.Encode(strings.Repeat("The clerk counted barrels of salt on the upper floor. ", 60), true)
	if len(prompt) < MaxStepRows {
		t.Fatalf("the prompt is %d tokens; the gate needs %d", len(prompt), MaxStepRows)
	}
	chunk, lead := prompt[:MaxStepRows-1], m.Vocab.Encode("The capital of France is", true)
	cfgFor := func(reserve bool) func(*tier.Config) {
		return func(c *tier.Config) {
			c.Sessions = 2
			if reserve {
				c.StepRows = MaxStepRows
			} else {
				c.NoScratchReserve = true
			}
		}
	}
	// open places two sessions and readies the decoding one.
	open := func(g *tier.GPU) (a, d *State, next int32, err error) {
		a, d = m.NewState(len(chunk)+8), m.NewState(len(lead)+8)
		for _, st := range []*State{a, d} {
			if err = st.SetDevice(g); err != nil {
				return
			}
		}
		lg, err := d.Prefill(lead)
		if err == nil {
			next = Greedy(lg)
		}
		return
	}
	size := func(reserve bool) tier.Stats {
		g := fitTier(t, spec, 0, cfgFor(reserve))
		defer g.Close()
		a, d, _, err := open(g)
		defer a.Close()
		defer d.Close()
		if err != nil {
			t.Fatal(err)
		}
		if a.GPULayers() != m.Cfg.NLayer || d.GPULayers() != m.Cfg.NLayer {
			t.Skipf("CARD TOO SMALL: %d and %d of %d blocks on the open card -- this gate proved nothing here (%s)",
				a.GPULayers(), d.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		return g.Stats()
	}
	if sr := size(true); sr.ReservedStep != MaxStepRows {
		t.Fatalf("the open card reserved a %d-row step: %s", sr.ReservedStep, sr.LastErr)
	}
	// Both sessions placed on the open card, then the card filled to 64 MiB
	// by the second context: the placement is the device's own, and whatever
	// the step asks for beyond it is asked of a full card.
	// grew is what the step's scratch took beyond what was charged before it:
	// with the reservation, nothing. A buffer the step sizes for itself is one
	// the full card is asked for, and whether the driver happens to have it
	// depends on the slack.
	var grew int64
	run := func(reserve bool) (tier.Stats, int, bool, error) {
		g := fitTier(t, spec, 0, cfgFor(reserve))
		defer g.Close()
		a, d, next, err := open(g)
		defer a.Close()
		defer d.Close()
		if err != nil {
			return g.Stats(), 0, false, err
		}
		// The history the step writes is each session's own and is taken
		// through the budget before the card fills: what is left to ask for
		// at the step is the step's scratch, which is the gate.
		if !a.ld.ReserveKV(len(chunk)) || !d.ld.ReserveKV(len(lead)+1) {
			t.Fatalf("the sessions' history did not fit the open card: %s", g.Err())
		}
		b.fill(t, slack)
		defer b.empty()
		joint := a.Steppable() && d.Steppable()
		before := g.Stats().ScratchBytes
		_, err = StepRuns([]Run{{State: a, Tokens: chunk, Logits: true}, {State: d, Tokens: []int32{next}, Logits: true}})
		grew = int64(g.Stats().ScratchBytes) - int64(before)
		return g.Stats(), a.GPULayers(), joint, err
	}
	st1, n1, joint1, err := run(true)
	t.Logf("reserved: %d of %d blocks, joint %v, scratch %d bytes (%+d at the step), step %d rows, err %v",
		n1, m.Cfg.NLayer, joint1, st1.ScratchBytes, grew, st1.ReservedStep, err)
	if err != nil || !joint1 || st1.ReservedStep != MaxStepRows {
		t.Fatalf("a %d-row step on the reserved card (joint %v, %d rows reserved): %v",
			MaxStepRows, joint1, st1.ReservedStep, err)
	}
	if grew > 0 {
		t.Fatalf("the reserved step took %d bytes of scratch the placement did not: a full card is asked for them",
			grew)
	}
	st2, n2, joint2, err := run(false)
	t.Logf("unreserved: %d of %d blocks, joint %v, err %v: %s", n2, m.Cfg.NLayer, joint2, err, st2.LastErr)
	if err == nil || !joint2 || !isOOM(err.Error()) {
		t.Fatalf("without the reservation the %d-row step did not run out of memory (joint %v, err %v): the card "+
			"was not full, so this gate does not discriminate", MaxStepRows, joint2, err)
	}
}
