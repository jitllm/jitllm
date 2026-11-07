package graph

// DecisionScores is a label readout from one prompt's next-token logits at its
// options' token ids (the head's label pick): each of N options scores by the
// largest logit of its group of G, and
// the options a question has are calibrated at its temperature into a
// distribution, averaged with a prior distribution at its weight (a second
// variant of the question averaged with the first). Its inputs are
// decision.vals (N groups of G logits, a group's unused slots repeating its
// first), decision.keep (1 for each of the question's options, 0 past
// them), decision.params (1/T, the prior's weight, then each option's index,
// N words: 2+N) and decision.prior (N); its outputs the distribution (N, 0 past
// the question's options) and its statistics (DecisionStats words).
type DecisionScores struct {
	N, G int
}

// DecisionStats are the words of DecisionScores' second output: the largest
// probability; the expected index; the mode's index (the lowest of equal
// largest); a choice's confidence (how far the largest stands above uniform);
// a score's (one minus the expected distance from the mode over a uniform
// pick's from the middle); and one minus the entropy over log k. Each
// confidence is 1 for one option, as there is nothing to weigh.
const (
	DecisionPMax = iota
	DecisionExpected
	DecisionMode
	DecisionChoiceConf
	DecisionScoreConf
	DecisionEntropyConf
	DecisionStats
)

func (d *DecisionScores) Build() *Graph {
	b := New()
	e := sampleEmitter{b}
	N, G := d.N, d.G
	vals := b.In("decision.vals", N*G)
	keep := b.In("decision.keep", N)
	params := b.In("decision.params", 2+N)
	prior := b.In("decision.prior", N)
	invT, w := e.word(params, 0, "decision.invtemp"), e.word(params, 1, "decision.priorw")
	at := e.one(OpSlice, "decision.at", N, Attrs{Offset: 2, Width: N}, params)

	scores := e.one(OpReduce, "decision.scores", N, Attrs{Fn: MapMax, Span: G}, vals)
	// The options past the question's read nothing: -inf before the softmax.
	none := e.mapc(MapMul, 0, "decision.none", N, keep)
	ninf := e.mapc(MapAdd, -3.4e38, "decision.ninf", N, none)
	kept := e.where("decision.kept", N, keep, scores, ninf)
	scaled := e.mapc(MapMul, 0, "decision.scaled", N, kept, invT)
	p := e.softmax(scaled, "decision.p")
	p = e.mapc(MapMul, 0, "decision.pk", N, p, keep)
	// The mean with the prior at its weight: (p + w prior) / (1 + w).
	mix := e.mapc(MapAdd, 0, "decision.mix", N, p, e.mapc(MapMul, 0, "decision.wprior", N, prior, w))
	p = e.mapc(MapDiv, 0, "decision.probs", N, mix, e.mapc(MapAdd, 1, "decision.norm", 1, w))

	pmax := e.one(OpReduce, "decision.pmax", 1, Attrs{Fn: MapMax}, p)
	expect := e.one(OpReduce, "decision.expect", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "decision.pi", N, p, at))
	// The mode: the lowest index whose probability is the largest.
	isMax := e.mapc(MapEqual, 0, "decision.ismax", N, p, pmax)
	big := e.mapc(MapAdd, 1e9, "decision.big", N, none)
	mode := e.one(OpReduce, "decision.mode", 1, Attrs{Fn: MapMin}, e.where("decision.modeat", N, isMax, at, big))
	dist := e.mapc(MapAbs, 0, "decision.dist", N, e.mapc(MapSub, 0, "decision.off", N, at, mode))
	modeDist := e.one(OpReduce, "decision.modedist", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "decision.pdist", N, p, dist))
	// A uniform pick's mean distance from the middle, (k-1)/2, of k options.
	k := e.one(OpReduce, "decision.k", 1, Attrs{Fn: MapAdd}, keep)
	mid := e.mapc(MapMul, 0.5, "decision.mid", 1, e.mapc(MapAdd, -1, "decision.km1", 1, k))
	udist := e.mapc(MapMul, 0, "decision.udist", N, keep, e.mapc(MapAbs, 0, "decision.uabs", N, e.mapc(MapSub, 0, "decision.uoff", N, at, mid)))
	uniform := e.mapc(MapDiv, 0, "decision.uniform", 1, e.one(OpReduce, "decision.usum", 1, Attrs{Fn: MapAdd}, udist), k)

	one := e.mapc(MapAdd, 1, "decision.one", 1, e.mapc(MapMul, 0, "decision.zero", 1, k))
	single := e.mapc(MapLess, 1.5, "decision.single", 1, k)
	u := e.mapc(MapDiv, 0, "decision.u", 1, one, k)
	choice := e.mapc(MapDiv, 0, "decision.choicer", 1, e.mapc(MapSub, 0, "decision.above", 1, pmax, u), e.mapc(MapSub, 0, "decision.span", 1, one, u))
	choice = e.where("decision.choiceconf", 1, single, one, e.mapc(MapMax, 0, "decision.choicec", 1, choice))
	score := e.mapc(MapSub, 0, "decision.scorer", 1, one, e.mapc(MapDiv, 0, "decision.rel", 1, modeDist, uniform))
	score = e.where("decision.scoreconf", 1, single, one, e.mapc(MapMax, 0, "decision.scorec", 1, score))
	// The entropy -sum p log p, log p the scaled scores' log-softmax.
	logp := e.where("decision.logpk", N, keep, e.logSoftmax(scaled, "decision.logp"), none)
	h := e.one(OpReduce, "decision.negh", 1, Attrs{Fn: MapAdd}, e.mapc(MapMul, 0, "decision.plogp", N, p, logp))
	ent := e.mapc(MapAdd, 0, "decision.entr", 1, one, e.mapc(MapDiv, 0, "decision.hrel", 1, h, e.mapc(MapLog, 0, "decision.logk", 1, k)))
	ent = e.where("decision.entconf", 1, single, one, ent)
	stats := e.one(OpConcat, "decision.stats", DecisionStats, Attrs{}, pmax, expect, mode, choice, score, ent)
	b.Out(p, stats)
	return b.G
}

// logSoftmax is x's log-softmax: each x - max - log of the sum of exp(x - max).
func (e sampleEmitter) logSoftmax(x ValueID, name string) ValueID {
	w := e.b.G.Values[x].Width
	m := e.one(OpReduce, name+".max", 1, Attrs{Fn: MapMax}, x)
	d := e.mapc(MapSub, 0, name+".d", w, x, m)
	sum := e.one(OpReduce, name+".sum", 1, Attrs{Fn: MapAdd}, e.mapc(MapExp, 0, name+".exp", w, d))
	return e.mapc(MapSub, 0, name, w, d, e.mapc(MapLog, 0, name+".log", 1, sum))
}
