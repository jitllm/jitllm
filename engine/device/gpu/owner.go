package gpu

import (
	"fmt"
)

// A tier is one model. Its blocks are keyed by index (GPU.own, devTier.layers)
// and each device holds one head and one scratch built from the first plan it
// saw, so a second model's block li offered beside the first's would be taken
// for the first's: shared, run against the wrong scratch, or its head answered
// by the first model's projection. Nothing in a block's index says which model
// it is, so the offer says (nn.LayerPlan.Model, nn.Head.Model) and the tier
// refuses another model's offers while anything of the first is placed.

// admit takes the tier for model, or refuses -- naming why, in every device's
// LastErr -- when another model has anything placed on it. what names the
// offer. On success the caller holds g.ownMu until it calls the returned func,
// so a second model cannot find the tier free while this offer is placing.
func (g *GPU) admit(model uint64, what string) (func(), bool) {
	g.ownMu.Lock()
	if g.held && g.model != model && g.holdsOf[g.model] == 0 {
		g.releaseHeld()
	}
	if g.held && g.model != model && g.holds() {
		other := g.model
		g.ownMu.Unlock()
		g.refuse(fmt.Sprintf("this tier holds model %d's blocks, and a tier serves one model: "+
			"model %d's %s stays on the host until model %d releases them (its Close)",
			other, model, what, other))
		return nil, false
	}
	g.model, g.held = model, true
	return g.ownMu.Unlock, true
}

// foreign reports that another model holds the tier, without taking it: for a
// call that only prepares (PrewarmLayer).
func (g *GPU) foreign(model uint64) bool {
	g.ownMu.Lock()
	defer g.ownMu.Unlock()
	return g.held && g.model != model && g.holds()
}

// holds reports that a block or a head is placed on any device. Callers hold
// g.ownMu and not g.mu.
func (g *GPU) holds() bool {
	g.mu.Lock()
	n := len(g.own)
	g.mu.Unlock()
	if n > 0 {
		return true
	}
	for _, d := range g.devs {
		if d.HeadResident() {
			return true
		}
	}
	return false
}

// refuse records why on every device, which is where Err and Stats read it.
func (g *GPU) refuse(why string) {
	for _, d := range g.devs {
		d.mu.Lock()
		d.LastErr = why
		d.mu.Unlock()
	}
}

// HoldModel counts an instance of model holding the tier (nn.ModelOwner).
func (g *GPU) HoldModel(model uint64) {
	g.ownMu.Lock()
	defer g.ownMu.Unlock()
	if g.holdsOf == nil {
		g.holdsOf = map[uint64]int{}
	}
	g.holdsOf[model]++
}

// ReleaseModel is one instance's release of model (nn.ModelOwner); the last
// gives back everything model placed: every block on every device with every
// session's history on it, the head, and what each device built from the
// model's plan -- its scratch, its KV pool, its argmax -- so the next model
// starts from a device that has seen nothing. Pages the prefix cache names
// hold the model as an instance does, so a later instance restores them: a
// model they alone hold keeps the tier until another model is admitted. A
// model that does not hold the tier releases nothing; a refused one never
// placed anything.
func (g *GPU) ReleaseModel(model uint64) {
	g.ownMu.Lock()
	defer g.ownMu.Unlock()
	if g.holdsOf[model] > 1 {
		g.holdsOf[model]--
		return
	}
	delete(g.holdsOf, model)
	if !g.held || g.model != model || g.named() {
		return
	}
	g.releaseHeld()
}

// named reports that some device names a page of the held model for the
// prefix cache. Callers hold g.ownMu.
func (g *GPU) named() bool {
	for _, d := range g.devs {
		if d.namesPages() {
			return true
		}
	}
	return false
}

// releaseHeld gives back everything the held model placed (ReleaseModel's
// last release). Callers hold g.ownMu.
func (g *GPU) releaseHeld() {
	for _, d := range g.devs {
		d.releaseModel()
		d.closeFrames()
	}
	g.mu.Lock()
	clear(g.own)
	g.head, g.cur, g.spillFrom = nil, 0, nil
	g.planN, g.share = 0, nil
	g.split = false
	clear(g.geo)
	g.mu.Unlock()
	g.held = false
}

// HoldModel and ReleaseModel forward to the tier: what it keeps is the
// model's.
func (s *gpuSession) HoldModel(model uint64)    { s.g.HoldModel(model) }
func (s *gpuSession) ReleaseModel(model uint64) { s.g.ReleaseModel(model) }

// releaseModel is ReleaseModel for one device: Close's walk without closing
// the device or its kernel caches, which are keyed by shape and serve any
// model.
func (g *devTier) releaseModel() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.dropGraph()
	for li := range g.layers {
		g.releaseBlock(li)
	}
	// The composed forms go with the model: a whole block's names the
	// model's graph, and the recordings that launched any of them are gone.
	g.partsClose()
	g.releaseRows()
	// The head's projection is keyed on its address like a lone matvec's, and
	// goes with its scratch -- the home set's, whichever set is in use: a
	// release after a picture finds the tower's set current.
	sets := []*blockScratch{g.bs}
	for _, s := range g.geoSets {
		sets = append(sets, s.bs)
	}
	for _, bs := range sets {
		if bs == nil || bs.head == nil {
			continue
		}
		r := bs.head
		g.dropHeadScratch(bs)
		for k, v := range g.res {
			if v == r {
				delete(g.res, k)
				break
			}
		}
		if r.ok {
			g.refund(r.bytes())
			g.freeResident(r)
		}
	}
	// Whatever else the model made resident goes with it: the input rows'
	// tables, which a head given back leaves held (embedTable). A resident is
	// keyed by its block or its entry, and the next model's would find it.
	for k, r := range g.res {
		delete(g.res, k)
		if r != nil && r.ok {
			g.refund(r.bytes())
			g.freeResident(r)
		}
	}
	g.afterRelease()
	g.dropAllScratch() // refunded as it goes (scratch.go)
	g.freeKVPool(g.kvp)
	g.kvp = nil
	if g.argmaxOut != nil {
		// Built for the head's row count; its word was charged as scratch.
		w := g.scratchWin()
		g.argmaxOut.Free()
		g.argmaxOut = nil
		w.close()
	}
	// The sampler's buffers were charged as scratch, like the argmax's.
	sw := g.scratchWin()
	g.lane0.freeSample()
	g.lane0.freePick()
	sw.close()
	// The capacity, the widest block and the retune mark are the model's
	// plan, read the first time a device sees one.
	g.kvCap, g.maxSeqAsked, g.paged = 0, 0, false
	g.widest = 0
	g.pinned, g.stream = nil, nil
	// The scratch sets went with dropAllScratch; which one is in use and the
	// non-causal set's widths are the next model's to state.
	g.geoCur = geoKey{}
	g.visCap, g.visMax, g.visKeys, g.visSmall, g.visFirst = 0, 0, 0, 0, 0
	g.visWarm = [2]int{}
	g.planeCap = 0
}
