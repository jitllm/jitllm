package gpu

import "github.com/jitllm/jitllm/nn"

// One k/v pair for every non-causal block of a set on a device.
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
	v := g.visOf(geoOf(p))
	if v.tkv != nil {
		if v.tkv.bytes < need {
			return nil
		}
		v.tkvRefs++
		return v.tkv
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
	v.tkv, v.tkvRefs = kvp, 1
	return kvp
}

// transientKVPut drops one holder of set k's pair and frees it with the
// last. Callers hold g.mu.
func (g *devTier) transientKVPut(k geoKey) {
	v := g.visOf(k)
	if v.tkv == nil {
		return
	}
	if v.tkvRefs--; v.tkvRefs > 0 {
		return
	}
	if v.tkv.kc != nil {
		v.tkv.kc.Free()
	}
	if v.tkv.vc != nil {
		v.tkv.vc.Free()
	}
	g.refund(v.tkv.bytes)
	g.KVBytes -= min(g.KVBytes, v.tkv.bytes)
	v.tkv, v.tkvRefs = nil, 0
}
