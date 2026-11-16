package convert

import (
	"errors"

	"github.com/jitllm/jitllm/format/jlm"
)

// legacyCode is what a container written before v31 states only by its
// architecture code, the code read as an opaque number: what a revision
// before jlm.SourceRevision left to the code (StateRevision states it at
// conversion now), and what the v30 migration refuses where unstated.
type legacyCode struct {
	flags        jlm.Flags
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
		9:   {flags: jlm.FlagRopeInterleaved},
		40:  {needRouted: true},
		45:  {bidirSliding: true},
		51:  {flags: jlm.FlagRopeInterleaved},
		56:  {actQuant: jlm.ActQuantRowInt8},
		57:  {ropeScaleSWA: 1},
		150: {swaGlobalFirst: true, needActs: true},
		200: {flags: jlm.FlagRopeInterleaved},
		201: {flags: jlm.FlagRopeInterleaved},
		204: {flags: jlm.FlagRopeXD},
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
		c.Flags |= l.flags
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
