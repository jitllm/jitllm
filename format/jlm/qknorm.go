package jlm

import "fmt"

// CheckQKNorm refuses a model whose q/k norm tensors disagree with QKNormAt on
// any of its nBlocks text blocks: a flag with no tensor runs a norm of nothing,
// and a tensor with no flag is a weight no tier reads. Write runs it on what it
// is handed and Open on what it reads, so neither a converter that forgets the
// flag nor a container written before this check reaches a tier.
func (c *Config) CheckQKNorm(nBlocks int, has func(r Role, block int32) bool) error {
	for b := 0; b < nBlocks; b++ {
		// A decision head's blocks (after the trunk's) norm nothing.
		want := c.QKNormAt(b) && !(c.Decision != DecisionNone && b >= int(c.NLayer))
		for _, r := range [2]Role{RoleAttnQNorm, RoleAttnKNorm} {
			// A KV-sharing block caches nothing and so norms q alone.
			want := want && !(r == RoleAttnKNorm && c.KVShared(b))
			switch got := has(r, int32(b)); {
			case want && !got:
				return fmt.Errorf("jlm: block %d has no %v and the config norms q and k there", b, r)
			case got && !want:
				return fmt.Errorf("jlm: block %d carries %v and the config does not norm q and k there "+
					"(FlagQKNorm %v, layer %v)", b, r, c.Flags.Has(FlagQKNorm), c.layerKind(b))
			}
		}
	}
	return nil
}
