package convert

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// legacyCode is what a container written before v31 states only by its
// architecture code, the code read as an opaque number: what a revision
// before jlm.SourceRevision left to the code (StateRevision states it at
// conversion now), and what the v30 migration refuses where unstated.
type legacyCode struct {
	// ropeLayout lays a model's rotary sections out (stateRopeTable).
	ropeLayout   string
	actQuant     jlm.ActQuant
	ropeScaleSWA float32
	// swaGlobalFirst sets Flag2SWAGlobalFirst on a windowed model;
	// bidirSliding makes the sliding layers a picture's bidirectional
	// layers on a windowed model with no per-layer embeddings.
	swaGlobalFirst, bidirSliding bool
	// needActs refuses a container stating no activations; needRouted one
	// with experts stating no routed weight for its router.
	needActs, needRouted bool
}

func legacyCodes() map[uint16]legacyCode {
	return map[uint16]legacyCode{
		9:   {ropeLayout: "interleaved"},
		40:  {needRouted: true},
		45:  {bidirSliding: true},
		51:  {ropeLayout: "interleaved"},
		56:  {actQuant: jlm.ActQuantRowInt8},
		57:  {ropeScaleSWA: 1},
		150: {swaGlobalFirst: true, needActs: true},
		200: {ropeLayout: "interleaved"},
		201: {ropeLayout: "interleaved"},
		204: {ropeLayout: "xd"},
	}
}

// LegacyStated is what a container written before v31 must state that only
// the architecture code its record names says (jlm.Stating): what its
// revision left to the code, stated here where the code is last read, and
// what the migration cannot recover, refused where unstated.
func LegacyStated(code uint16, c *jlm.Config) error {
	l := legacyCodes()[code]
	if c.Revision < jlm.SourceRevision {
		c.Revision = jlm.StatementRevision
		stateRopeLayout(c, l.ropeLayout == "xd", l.ropeLayout == "interleaved")
		if l.actQuant != 0 {
			c.ActQuant = l.actQuant
		}
		if l.ropeScaleSWA != 0 {
			c.RopeScaleSWA = l.ropeScaleSWA
		}
		if l.swaGlobalFirst && c.SWAWindow != 0 {
			c.Flags2 |= jlm.Flag2SWAGlobalFirst
		}
		if l.bidirSliding && c.PLEDim == 0 && c.SWAWindow > 0 {
			c.BidirLayers = make([]bool, c.NLayer)
			for li := range c.BidirLayers {
				c.BidirLayers[li] = c.Sliding(li)
			}
		}
	}
	switch {
	case l.needActs && len(c.Acts) == 0:
		return errors.New("states no activations")
	case l.needRouted && c.NExpert != 0 && len(c.RouterWeight.Terms) == 0:
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
