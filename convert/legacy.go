package convert

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// LegacyStated is what a container written before v31 must state that only
// the architecture code its record names says (jlm.Stating): what its
// revision states from the architecture (StateRevision), stated here where
// the code is last read, and ModernBERT's erf GELU as its activations and
// Phi-3.5-MoE's routed weight for its router, refused where unstated.
func LegacyStated(code uint16, c *jlm.Config) error {
	if c.Revision < jlm.SourceRevision {
		StateRevision(&Config{Config: c, Arch: Arch(code)})
	}
	switch a := Arch(code); {
	case a == ArchModernBERT && len(c.Acts) == 0:
		return errors.New("states no activations")
	case a == ArchPhiMoE && c.NExpert != 0 && len(c.RouterWeight.Terms) == 0:
		return errors.New("states no routed weight for its router")
	}
	return nil
}

// The rotary layout bits a container stated before it stated the table
// (jlm.Config.RopeTable): sections interleaved over a head's pairs, or over
// the rotary's elements.
const (
	legacyRopeInterleaved jlm.Flags = 1 << 29
	legacyRopeXD          jlm.Flags = 1 << 30
)

// stateStoredRopeTable states the rotary table and a picture's layout of a
// container that stated its rotary sections and their layout's bits alone,
// and clears the bits; false for one that states the table or has no
// sections.
func stateStoredRopeTable(c *jlm.Config) (bool, error) {
	if c.RopeSections == [4]uint32{} || len(c.RopeTable) != 0 {
		return false, nil
	}
	if c.Revision < 3 {
		return false, errors.New("a container below revision 3 states rotary sections and not their layout: convert the model again")
	}
	xd, il := c.Flags.Has(legacyRopeXD), c.Flags.Has(legacyRopeInterleaved)
	if xd && il {
		return false, errors.New("the container states its rotary sections over the elements and interleaved at once")
	}
	stateRopeLayout(c, xd, il)
	if len(c.RopeTable) == 0 {
		return false, fmt.Errorf("the container's rotary sections %v do not lay out over its rotary of %d", c.RopeSections, c.NRot)
	}
	c.Flags &^= legacyRopeInterleaved | legacyRopeXD
	return true, nil
}
