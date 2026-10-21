package engine

import (
	"fmt"
	"github.com/jitllm/jitllm/engine/model"
	"regexp"
	"strings"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// header is what the session header says about the running model: its name,
// where its layers are, how long a conversation it holds, and what it can do.
//
// It deliberately omits the geometry (layers, heads, vocab), which the
// Models screen shows where a model is chosen.
type header struct {
	Name     string
	Layers   int
	OnDevice int
	Device   string // the card holding them; "" when none or several
	Devices  int    // how many cards hold a layer
	// Context is the window the session has; Asked is what the user asked
	// for, which the engine clamps to the model's own before the session is
	// built -- so the clamp is only visible by comparing the two.
	Context, Asked int
	Chat, Vision   bool
}

func (h header) String() string {
	// The device last: it is the longest term and the header is cut to one
	// line, so it must not push out "chat" or "sees images".
	parts := []string{h.Name}
	if h.Chat {
		parts = append(parts, "chat")
	} else {
		parts = append(parts, "completion only (no chat template)")
	}
	if h.Vision {
		parts = append(parts, "sees images")
	}

	ctx := "context " + thousands(h.Context)
	if h.Asked > h.Context {
		ctx += " (the most it supports)"
	}
	parts = append(parts, ctx)

	switch {
	case h.OnDevice <= 0:
		parts = append(parts, "on the CPU")
	case h.OnDevice >= h.Layers:
		parts = append(parts, fmt.Sprintf("all %d layers on %s", h.Layers, h.where()))
	default:
		parts = append(parts, fmt.Sprintf("%d of %d layers on %s, the rest on the CPU",
			h.OnDevice, h.Layers, h.where()))
	}
	return strings.Join(parts, " · ")
}

func (h header) where() string {
	if h.Devices > 1 {
		return fmt.Sprintf("%d GPUs", h.Devices)
	}
	if h.Device == "" {
		return "the GPU"
	}
	return "the " + h.Device
}

// thousands writes 8192 as "8,192".
func thousands(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// deviceLabel is a tier's device name without the ordinal and the API:
// "0:<card name> [cuda]" reads as the card.
func deviceLabel(name string) string {
	return strings.TrimSpace(deviceNoise.ReplaceAllString(name, ""))
}

var deviceNoise = regexp.MustCompile(`^\d+:|\s*\[[^\]]*\]$`)

// headerNow reads the header off the running session. Worker goroutine only.
func (e *Engine) headerNow() header {
	c := e.m.Cfg
	h := header{
		Name:     strings.TrimSuffix(short(e.path), ".jlm"),
		Layers:   c.NLayer,
		OnDevice: e.deviceBlocks(),
		Context:  e.sessMax,
		Asked:    e.st.MaxSeq.Get(),
		Chat:     e.m.ChatCapable(),
		Vision:   e.m.Tower() != nil,
	}
	if e.gpu != nil {
		h.Device, h.Devices = placedOn(e.gpu)
	}
	return h
}

// placedOn names the card holding blocks, and counts the cards that do.
func placedOn(g *tier.GPU) (string, int) {
	budgets, placed := g.Budgets(), g.Placed()
	name, n := "", 0
	for i, b := range budgets {
		if i < len(placed) && placed[i] > 0 {
			name, n = deviceLabel(b.Device), n+1
		}
	}
	return name, n
}

// deviceBlocks is how many of the chat session's blocks a device holds.
func (e *Engine) deviceBlocks() (n int) {
	e.inspect(func(st *model.State) { n = st.GPULayers() })
	return n
}
