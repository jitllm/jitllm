//go:build jitllmfault || jitllmtest

package tier

import "fmt"

// hooks is a tier's deliberate-violation state, compiled in only under the
// jitllmfault or jitllmtest build tag (the release build's is an empty struct,
// hooks_off.go). It lives on the Config, so it belongs to one tier and not to
// the process: two tiers in one process arm their faults independently.
//
// The fault names are documented beside fault.go's old switches; the enums are
// faultids.go and the names below are parsed only here, where a test arms one.
type hooks struct {
	// err is the first fault or violation named that does not exist: New
	// returns it, since a gate that arms nothing proves nothing (RULE 10).
	err   error
	paged pagedFaultID
	rec   recFaultID
	rows  rowsFaultID
	room  roomFaultID
	moe   moeFaultID
	peer  peerFaultID

	// The wrong-answer switches of the fold and fusion gates.
	addNormBroken, normGateBroken, qkvGemmSwapped, gateUpSwapped, gateUp2Swapped bool

	// sink keeps the decodealloc fault's allocation alive past escape analysis.
	sink []byte
}

func (c *Config) pagedFaulted(f pagedFaultID) bool { return c.hook.paged == f }
func (c *Config) recFaulted(f recFaultID) bool     { return c.hook.rec == f }
func (c *Config) rowsFaulted(f rowsFaultID) bool   { return c.hook.rows == f }
func (c *Config) roomFaulted(f roomFaultID) bool   { return c.hook.room == f }
func (c *Config) moeFaulted(f moeFaultID) bool     { return c.hook.moe == f }
func (c *Config) peerFaulted(f peerFaultID) bool   { return c.hook.peer == f }

// blockFaulted says a fault the block encoder reads (paged, ragged-step) is
// armed: such a step runs the encoder, which the fault is written against, not
// the device program.
func (c *Config) blockFaulted() bool { return c.hook.paged != pagedNone || c.hook.rows != rowsNone }

func (c *Config) addNormBroken() bool  { return c.hook.addNormBroken }
func (c *Config) normGateBroken() bool { return c.hook.normGateBroken }
func (c *Config) qkvGemmSwapped() bool { return c.hook.qkvGemmSwapped }
func (c *Config) gateUpSwapped() bool  { return c.hook.gateUpSwapped }
func (c *Config) gateUp2Swapped() bool { return c.hook.gateUp2Swapped }

// allocFault makes the decodealloc violation: one small heap allocation.
func (c *Config) allocFault() {
	if c.hook.rows == rowsDecodealloc {
		c.hook.sink = make([]byte, 32)
	}
}

// setFault arms one violation of a family; "" disarms it. A name it does
// not know is recorded (hooks.err) and refused by New: a gate that arms
// nothing proves nothing (RULE 10).
func (h *hooks) setFault(family, name string) {
	look := func(ok bool) {
		if !ok && h.err == nil {
			h.err = fmt.Errorf("tier: no %q fault named %q", family, name)
		}
	}
	var ok bool
	switch family {
	case "paged":
		h.paged, ok = pagedFaultNames[name]
	case "rec":
		h.rec, ok = recFaultNames[name]
	case "rows":
		h.rows, ok = rowsFaultNames[name]
	case "room":
		h.room, ok = roomFaultNames[name]
	case "moe":
		h.moe, ok = moeFaultNames[name]
	case "peer":
		h.peer, ok = peerFaultNames[name]
	default:
		if h.err == nil {
			h.err = fmt.Errorf("tier: no fault family %q", family)
		}
		return
	}
	look(ok || name == "")
}

// setViolation sets one of the wrong-answer switches and returns the setting
// it replaced: "gateup", "gateup2", "addnorm", "qkvgemm" or "normgate".
func (h *hooks) setViolation(name string, on bool) bool {
	var p *bool
	switch name {
	case "gateup":
		p = &h.gateUpSwapped
	case "gateup2":
		p = &h.gateUp2Swapped
	case "addnorm":
		p = &h.addNormBroken
	case "qkvgemm":
		p = &h.qkvGemmSwapped
	case "normgate":
		p = &h.normGateBroken
	default:
		if h.err == nil {
			h.err = fmt.Errorf("tier: no violation named %q", name)
		}
		return false
	}
	old := *p
	*p = on
	return old
}

// WithFault arms one named violation of a family ("paged", "rec", "rows",
// "room", "moe", "peer") on the tier this option opens, for the faults gates
// (RULE 10). A release build ignores it.
func WithFault(family, name string) Option {
	return WithConfig(func(c *Config) { c.hook.setFault(family, name) })
}

// WithViolation turns on one wrong-answer switch ("gateup", "gateup2",
// "addnorm", "qkvgemm", "normgate") on the tier this option opens.
func WithViolation(name string) Option {
	return WithConfig(func(c *Config) { c.hook.setViolation(name, true) })
}

// SetFault arms (or, with "", disarms) one violation on this tier, for a gate
// that toggles it between runs. Only this tier's devices see it.
func (g *GPU) SetFault(family, name string) { g.Config.hook.setFault(family, name) }

// SetViolation sets one wrong-answer switch on this tier and returns the
// setting it replaced.
func (g *GPU) SetViolation(name string, on bool) bool { return g.Config.hook.setViolation(name, on) }

// failure is the first fault or violation named that does not exist.
func (h *hooks) failure() error { return h.err }
