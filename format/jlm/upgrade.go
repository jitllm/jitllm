package jlm

import "fmt"

// UpgradeError is a container an older writer made that states in fields of
// their own what this build reads only as statements; What names it.
// Reconvert says an upgrade cannot state it from the container, so the
// model is converted again from its source.
type UpgradeError struct {
	What      string
	Reconvert bool
}

func (e *UpgradeError) Error() string {
	return "jlm: the container states " + e.What + " in fields of their own, which this build reads only as " +
		"statements: convert the model again"
}

// olderStatements refuses what an older writer kept in fields of their own:
// predicted streams, a hyper-connection gate or checkpoint bank with no
// stated streams, an indexer with no stated selection, an n-gram block with
// no stated n-gram op, a rotary scaled by a method (a linear factor, YaRN's
// correction range) rather than by the factors it states per pair. A
// checkpoint bank beside another residual structure is refused too.
func (c *Config) olderStatements() error {
	switch {
	case c.PredStreams != 0:
		return &UpgradeError{What: "predicted streams", Reconvert: true}
	case (c.AttnResBlock != 0 || c.HCLowRank != 0) && c.Streams.N <= 1:
		return &UpgradeError{What: "residual streams", Reconvert: true}
	case c.IdxHeads != 0 && c.Selection == nil && len(c.CompKinds) == 0:
		return &UpgradeError{What: "the history its attention selects", Reconvert: true}
	case c.NGramBlock != 0 && !c.Streams.states(TOpNGramRows):
		return &UpgradeError{What: "the n-gram block", Reconvert: true}
	case c.RopeLinear != 0 || c.YarnBetaFast != 0 || c.YarnBetaSlow != 0:
		return &UpgradeError{What: "its rotary's scaling method", Reconvert: true}
	case c.AttnResBlock != 0 && (c.HCMult != 0 || c.Flags.Has(FlagParallel) || c.PredictionBlocks() != 0):
		return fmt.Errorf("jlm: a checkpoint bank every %d blocks beside another residual structure", c.AttnResBlock)
	}
	return nil
}
