package tier

import "github.com/jitllm/jitllm/engine/nn"

// One k/v pair for every non-causal block on a device.
//
// A non-causal block's keys and values live only inside its own call
// (nn.LayerPlan.NonCausal): the block writes them, attends over them, and the
// next block writes its own -- nothing reads a block's cache after the block,
// and no picture's survives into the next. Its history is one block deep, so
// every such block writes the same pair: the KV budget's transient sequence,
// charged to KVBytes once, where a cache per block kept 27 of them for Gemma
// 3's SigLIP (37.7 MB each at 4096 patches, a gigabyte of a 4 GB card) for
// nothing. Sessions share it too: a call over a tower runs alone on the device
// (exclusiveRange). It is freed
// with its last holder.

// transientKV hands out the shared pair, allocating it on first use, and counts
// the holder. nil means it could not be had at this size (a bigger plan than
// the pair was made for while blocks still hold it, or no room): the caller
// allocates a pair of its own as before. Callers hold g.mu.
func (g *devTier) transientKV(p *nn.LayerPlan) *kvPair {
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	need := uint64(cache + vcache)
	if g.tkv != nil {
		if g.tkv.bytes < need {
			return nil
		}
		g.tkvRefs++
		return g.tkv
	}
	kvp := &kvPair{shared: true, bytes: need}
	var err error
	if kvp.kc, err = g.dev.Alloc(cache); err == nil && vcache > 0 {
		kvp.vc, err = g.dev.Alloc(vcache)
	}
	if err != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		return nil
	}
	g.charge(need)
	g.KVBytes += need
	g.tkv, g.tkvRefs = kvp, 1
	return kvp
}

// transientKVPut drops one holder and frees the pair with the last. Callers hold
// g.mu.
func (g *devTier) transientKVPut() {
	if g.tkv == nil {
		return
	}
	if g.tkvRefs--; g.tkvRefs > 0 {
		return
	}
	if g.tkv.kc != nil {
		g.tkv.kc.Free()
	}
	if g.tkv.vc != nil {
		g.tkv.vc.Free()
	}
	g.refund(g.tkv.bytes)
	g.KVBytes -= min(g.KVBytes, g.tkv.bytes)
	g.tkv, g.tkvRefs = nil, 0
}
