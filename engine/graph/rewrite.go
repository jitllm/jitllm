package graph

import (
	"fmt"
	"sort"
)

// A Rewrite is an optimization written against the block graph
// (docs/design/composer.md, "The rewrite interface"): where it applies, found
// by matching operations and never by asking for an architecture; why a near
// match is refused; what it promises numerically on each backend; what it
// costs statically; and what makes a verdict about it stale. The recipes it
// offers are built and timed by the tier that runs the block; a rewrite adds
// candidates, not a tuner.
type Rewrite struct {
	Name string
	// Match returns the candidate regions of g: op indices in g.Ops, in
	// order, of the shape the rewrite looks for.
	Match func(g *Graph) []Region
	// Legal refuses a matched region with the reason, or returns nil: the
	// graph's own guards (dataflow, shapes, state).
	Legal func(g *Graph, r Region) error
	// Pays refuses a legal region on a target with the reason, or returns
	// nil: the guards on the device, the executed row width and the layout
	// under which the recipe can win at all. A recipe inert on a target (a
	// tile shape the ISA lacks, a row class it does not change) is refused
	// here, by name, never by an if at its call site. nil offers it on every
	// target the Contract names.
	Pays func(g *Graph, r Region, t Target) error
	// Contract is the numerical promise against the reference recipe, per
	// backend; a backend absent from it is not offered the rewrite.
	Contract map[string]Contract
	// Bounds is the region's static cost change: it may refuse a candidate,
	// never accept one (only a timed complete region accepts).
	Bounds func(g *Graph, r Region) Resources
	// Keys names what a timed verdict depends on beyond the tier's shape key
	// and Target.Key, which every verdict carries.
	Keys func(g *Graph, r Region) string
	// When is the time a legal, offered region's choice is made, and so who
	// makes it (Chooser). A rewrite whose When is WhenRun is an open
	// alternative: it stays in the plan, listed with its cost, and runs only
	// where a caller asks for it (Choice).
	When When
	// Objective is what a WhenRun rewrite buys its chooser, where it is not
	// speed ("capacity": weights a card no longer holds).
	Objective string
	// Chooser names who may choose it: the static guard, the tier's trial,
	// or the With...() option a caller passes.
	Chooser string
	// Evidence is where the cost Bounds states was measured or derived.
	Evidence string
}

// When is the time a rewrite's choice is made (docs/design/composer.md,
// "When a rewrite is decided"). Legality is always decided at compile time,
// when the graph is matched; When is about the choice among legal forms.
type When uint8

const (
	// WhenCompile: a static guard decides (Pays at the executed row class,
	// the residency, the bound): no measurement is needed.
	WhenCompile When = iota + 1
	// WhenLoad: the tier times the legal forms on the device in front of it
	// and records a verdict keyed by the shape, Target.Key and the row class.
	WhenLoad
	// WhenRun: an orchestrator (a server, a placement policy) proposes it for
	// its own objective. Never taken unasked.
	WhenRun
)

var whenNames = [...]string{"unstated", "compile time", "load time", "run time"}

func (w When) String() string {
	if int(w) < len(whenNames) {
		return whenNames[w]
	}
	return fmt.Sprintf("when(%d)", w)
}

// Choice is a caller's say over one rewrite, by name (Choices): Auto leaves
// it to its When (the static guard, the trial, or for a run-time alternative
// not taken), Take asks for it wherever it is offered, Refuse keeps the
// reference recipe. A Take where the rewrite is not offered is not met, and
// the caller sees that (the tier's unmet count and its explanation), never a
// silent reference recipe.
type Choice uint8

const (
	ChoiceAuto Choice = iota
	ChoiceTake
	ChoiceRefuse
)

// Choices is a caller's Choice per rewrite name.
type Choices map[string]Choice

// Check refuses a name no rewrite in Rewrites carries: a misspelt choice
// would otherwise select nothing and look like a choice made.
func (c Choices) Check() error {
	for name := range c {
		known := false
		for _, rw := range Rewrites {
			known = known || rw.Name == name
		}
		if !known {
			return fmt.Errorf("no rewrite is named %q", name)
		}
	}
	return nil
}

// Chosen is the decision on a region at target t under choice c: taken, or
// not; met is false only where c is Take and the rewrite is not offered at
// t. A WhenRun rewrite under Auto is not taken. A WhenLoad one under Auto is
// a candidate: taken means the tier may time it, its trial decides.
// Allocates nothing.
func (o Offered) Chosen(t Target, c Choice) (taken, met bool) {
	why := o.Offer(t)
	switch c {
	case ChoiceRefuse:
		return false, true
	case ChoiceTake:
		return why == nil, why == nil
	}
	return why == nil && o.Rewrite.When != WhenRun, true
}

// Target is what a rewrite's guards and its verdicts are keyed by: the
// device's capability, the executed row-width class and the layout facts of
// the rows. A verdict is evidence for its Target.Key only; a block on another
// card, at another row class or in another layout takes that key's verdict or
// the reference recipe. Generated code may differ per device and per model:
// the bar is the gated error margin and the speed, never bit identity across
// devices.
type Target struct {
	API       string // "ptx", "spirv", "msl", "host"
	ISA       string // the code generator's tier: "sm_70", "sm_75", "avx2", "sse", "neon-dotprod"
	MMA       string // the matrix shape the ISA offers ("m8n8k4", "m16n8k16"), "" for none
	Subgroup  int    // threads in a subgroup, warp or SIMD-group
	SharedMem int    // bytes of shared memory a workgroup may hold
	Rows      RowClass
	Layout    Layout
	Residency Residency
	// PeerCards is the cards beyond this one a split executor can run a
	// block's other part on (MLPPartition). It is not part of Key: no timed
	// verdict depends on it.
	PeerCards int
}

// Residency is where the weights a region reads are when it runs: a
// mixture's bank whole on the card, streamed through an expert cache there
// (a subset of the experts resident, the rest admitted on demand), or on the
// host. 0 is unstated, which a guard that needs it refuses.
type Residency uint8

const (
	ResidencyCard   Residency = iota + 1 // every weight the region reads is on the card
	ResidencyCached                      // the bank streams through an expert cache on the card
	ResidencyHost                        // the experts run on the host
)

// RowClass is the executed row-width class a region runs at: a lever's
// arithmetic changes form across these, so a verdict at one is no evidence
// at another.
type RowClass uint8

const (
	RowsDecode RowClass = iota + 1 // one row per sequence
	RowsFew                        // a ragged step or a speculative verify: fewer routed pairs than the bank has experts
	RowsChunk                      // a prompt chunk: a staged GEMM's width
)

// Layout is how the rows a region reads are laid out.
type Layout struct {
	// Streams is the residual's stream count where it is stream-major
	// (hyper-connections, AltUp): a row slice must take whole streams. 0 or
	// 1 is one stream.
	Streams int
}

// Key is the target's part of a persisted verdict key.
func (t Target) Key() string {
	return fmt.Sprintf("%s/%s/%s/sg%d/sh%d/r%d/s%d/x%d", t.API, t.ISA, t.MMA, t.Subgroup, t.SharedMem, t.Rows, t.Layout.Streams, t.Residency)
}

// Offer is a legal region's decision on target t: refused where the backend
// has no contract or the rewrite's Pays guard says it cannot win there.
func (rw *Rewrite) Offer(g *Graph, r Region, t Target) error {
	return rw.offer(rw.Legal(g, r), g, r, t)
}

// offer is Offer with the region's legality already decided: a graph does
// not change after it is built, so a matched region's Legal is asked once
// (Match), and the per-target part allocates nothing where it offers.
func (rw *Rewrite) offer(legal error, g *Graph, r Region, t Target) error {
	if legal != nil {
		return legal
	}
	// The static bound refuses, never accepts: a region that removes no
	// launch and adds traffic cannot win on any target. A run-time
	// alternative's objective is not speed, so its bound is its cost, stated
	// and not a refusal.
	if rw.Bounds != nil && rw.When != WhenRun {
		if b := rw.Bounds(g, r); b.Launches <= 0 && b.TrafficBytes < 0 {
			return fmt.Errorf("the static bound: no launch removed and %d bytes a row added", -b.TrafficBytes)
		}
	}
	// A run-time alternative's target guard first: whether the target can
	// execute it at all is what its chooser needs to hear.
	if rw.When == WhenRun && rw.Pays != nil {
		if err := rw.Pays(g, r, t); err != nil {
			return err
		}
	}
	if _, ok := rw.Contract[t.API]; !ok {
		return fmt.Errorf("no numerical contract on %s", t.API)
	}
	if rw.Pays != nil && rw.When != WhenRun {
		return rw.Pays(g, r, t)
	}
	return nil
}

// Region is a matched set of ops.
type Region struct {
	Ops []int
}

// Contract is a rewrite's numerical promise on one backend, against the
// reference recipe on that same backend; across devices only the gated band
// binds.
type Contract struct {
	Exact bool    // bit-identical to the reference recipe on this backend
	Band  float64 // otherwise the largest relative logit difference its gate allows
	Gate  string  // the gate that holds it
}

// Resources is a candidate's static cost change against the reference.
type Resources struct {
	Launches     int   // launches removed (negative: added)
	ScratchBytes int64 // scratch added per row
	TrafficBytes int64 // intermediate bytes per row no longer written and read (negative: partials added)
	// WeightElems is weight elements a card no longer holds (a split's
	// capacity), and Crossings the copies between cards it adds per row.
	WeightElems int64
	Crossings   int
}

func (r Resources) String() string {
	s := fmt.Sprintf("launches %+d, scratch %+d B/row, traffic %+d B/row", r.Launches, r.ScratchBytes, r.TrafficBytes)
	if r.WeightElems != 0 || r.Crossings != 0 {
		s += fmt.Sprintf(", %d weight elements off each card, %d crossings/row", r.WeightElems, r.Crossings)
	}
	return s
}

// Decision is one rewrite's verdict on one region, for a plan explanation:
// legal or refused, and for a legal one when it is chosen, by whom, and the
// cost its bound states.
type Decision struct {
	Rewrite string
	Region  Region
	Err     error // nil where legal
	When    When
	Chooser string
	Cost    *Resources // the static bound, where the rewrite states one
	Why     string     // the objective and the cost's evidence, for an open alternative
}

func (d Decision) String() string {
	if d.Err != nil {
		return fmt.Sprintf("%s at ops %v: refused: %v", d.Rewrite, d.Region.Ops, d.Err)
	}
	s := fmt.Sprintf("%s at ops %v: legal, decided at %s by %s", d.Rewrite, d.Region.Ops, d.When, d.Chooser)
	if d.When == WhenRun {
		s = fmt.Sprintf("%s at ops %v: open alternative (%s), chosen only by %s", d.Rewrite, d.Region.Ops, d.When, d.Chooser)
	}
	if d.Cost != nil {
		s += "; cost: " + d.Cost.String()
	}
	if d.Why != "" {
		s += "; " + d.Why
	}
	return s
}

// Apply runs every rewrite over g and returns each match's decision, legal
// or refused with its reason, and for a legal one its When, chooser and cost.
func Apply(g *Graph, rws ...*Rewrite) []Decision {
	var out []Decision
	for _, rw := range rws {
		for _, r := range rw.Match(g) {
			d := Decision{Rewrite: rw.Name, Region: r, Err: rw.Legal(g, r), When: rw.When, Chooser: rw.Chooser}
			if d.Err == nil && rw.Bounds != nil {
				b := rw.Bounds(g, r)
				d.Cost = &b
			}
			if d.Err == nil && rw.When == WhenRun {
				d.Why = "objective " + rw.Objective + "; " + rw.Evidence
			}
			out = append(out, d)
		}
	}
	return out
}

// users is, per value, the ops that read it, and producer the op that writes
// it (-1 for a block input).
func users(g *Graph) (use [][]int, producer []int) {
	use = make([][]int, len(g.Values))
	producer = make([]int, len(g.Values))
	for i := range producer {
		producer[i] = -1
	}
	for i, o := range g.Ops {
		for _, v := range o.In {
			use[v] = append(use[v], i)
		}
		for _, v := range o.Out {
			producer[v] = i
		}
	}
	return use, producer
}

// Gate2In is the FFN chain the device runs as one launch (the gate2in
// recipe, jit/gpu/tier/recipe.go): a norm of the residual whose output only
// two matvecs read, their outputs only an act_mul reads, its output only a
// matvec reads, and that matvec's output only the residual add reads. Matched
// by those edges, in any block graph that has them.
var Gate2In = &Rewrite{
	Name: "gate2in",
	Match: func(g *Graph) []Region {
		use, _ := users(g)
		var out []Region
		for i, o := range g.Ops {
			if o.Kind != OpNorm || len(o.Out) != 1 {
				continue
			}
			// The two matvecs reading the norm, whatever else does.
			var mvs []int
			for _, u := range use[o.Out[0]] {
				if g.Ops[u].Kind == OpMatVec {
					mvs = append(mvs, u)
				}
			}
			if len(mvs) != 2 {
				continue
			}
			am := -1
			for _, u := range use[g.Ops[mvs[0]].Out[0]] {
				if g.Ops[u].Kind == OpActMul {
					am = u
				}
			}
			if am < 0 {
				continue
			}
			r := Region{Ops: []int{i, mvs[0], mvs[1], am}}
			for _, u := range use[g.Ops[am].Out[0]] {
				if g.Ops[u].Kind == OpMatVec {
					r.Ops = append(r.Ops, u)
					d := g.Ops[u].Out[0]
					for _, a := range use[d] {
						r.Ops = append(r.Ops, a)
					}
					break
				}
			}
			sort.Ints(r.Ops)
			out = append(out, r)
		}
		return out
	},
	Legal: func(g *Graph, r Region) error {
		use, producer := users(g)
		if len(r.Ops) != 6 {
			return fmt.Errorf("the chain has %d ops, not norm, gate, up, act_mul, down and the residual add", len(r.Ops))
		}
		norm, gate, up, am, down, add := g.Ops[r.Ops[0]], g.Ops[r.Ops[1]], g.Ops[r.Ops[2]], g.Ops[r.Ops[3]], g.Ops[r.Ops[4]], g.Ops[r.Ops[5]]
		switch {
		case norm.Kind != OpNorm:
			return fmt.Errorf("the chain does not start at a norm")
		case len(use[norm.Out[0]]) != 2:
			return fmt.Errorf("the norm's output has %d readers; the recipe recomputes it inside gate and up and writes it nowhere", len(use[norm.Out[0]]))
		case len(gate.Weight) != 1 || len(up.Weight) != 1:
			return fmt.Errorf("gate or up carries a bias the recipe does not add")
		case am.Kind != OpActMul || am.In[0] != gate.Out[0] || am.In[1] != up.Out[0]:
			return fmt.Errorf("the act_mul does not read act(gate)*up")
		case len(use[gate.Out[0]]) != 1 || len(use[up.Out[0]]) != 1:
			return fmt.Errorf("gate or up has a reader besides the act_mul")
		case down.Kind != OpMatVec || len(use[am.Out[0]]) != 1:
			return fmt.Errorf("the product has a reader besides down")
		case add.Kind != OpAdd:
			return fmt.Errorf("down's output reaches the residual through a %s, not the add", add.Kind)
		case len(use[down.Out[0]]) != 1:
			return fmt.Errorf("down's output has %d readers", len(use[down.Out[0]]))
		}
		// The norm reads the residual an add wrote: the recipe takes the row
		// the attention output's launch already added.
		in := norm.In[0]
		if p := producer[in]; p < 0 || g.Ops[p].Kind != OpAdd {
			return fmt.Errorf("the norm's input is not a residual an add wrote")
		}
		if add.In[0] != in && add.In[1] != in {
			return fmt.Errorf("down adds into a row other than the one the norm read")
		}
		return nil
	},
	Contract: map[string]Contract{
		"ptx": {Exact: true, Gate: "backend.TestGate2InIsTheTwoLaunches, model.TestFFNRecipesMatchTwoLaunches"},
	},
	// It removes the FFN norm's launch on a one-row decode; a chunk runs the
	// norm once for every row, so there it removes nothing.
	Pays: func(g *Graph, r Region, t Target) error {
		if t.Rows != RowsDecode {
			return fmt.Errorf("the recipe is a decode matvec; at row class %d the norm is one launch for every row", t.Rows)
		}
		return nil
	},
	Bounds:  func(g *Graph, r Region) Resources { return Resources{Launches: 1} },
	Keys:    func(g *Graph, r Region) string { return "" },
	When:    WhenLoad,
	Chooser: "the tier's FFN trial (tuneFFN), or tier.WithFFNForm / tier.WithRewrite",
}

// Offered is a rewrite matched on one block's graph, carried to the tier
// that runs the block (nn.LayerPlan.Rewrites): the tier knows its device and
// the row class it is about to run, so it builds the Target and asks Offer
// there, for every recipe it considers. The graph and region travel with it;
// a block that relocates takes them to the next device, which asks again
// with its own Target.
type Offered struct {
	Rewrite *Rewrite
	Graph   *Graph
	Region  Region
	// legal is the region's Legal, decided when Match found it.
	legal   error
	matched bool
}

// Offer is the rewrite's decision on this region at target t: Legal, the
// backend's contract, and the Pays guard, each refusal with its reason.
func (o Offered) Offer(t Target) error {
	if !o.matched {
		return o.Rewrite.Offer(o.Graph, o.Region, t)
	}
	return o.Rewrite.offer(o.legal, o.Graph, o.Region, t)
}

// Key is what a timed verdict on this region at t is persisted under beyond
// the tier's shape key: the target's key and whatever the rewrite's Keys
// names.
func (o Offered) Key(t Target) string {
	k := t.Key()
	if o.Rewrite.Keys != nil {
		if s := o.Rewrite.Keys(o.Graph, o.Region); s != "" {
			k += "/" + s
		}
	}
	return k
}

// Match is every region each rewrite matches in g, legal or not: a tier asks
// each one's Offer at its own target, and a near miss refuses there with the
// reason Legal gives.
func Match(g *Graph, rws ...*Rewrite) []Offered {
	out := []Offered{}
	for _, rw := range rws {
		for _, r := range rw.Match(g) {
			out = append(out, Offered{Rewrite: rw, Graph: g, Region: r, legal: rw.Legal(g, r), matched: true})
		}
	}
	return out
}

// Find is the first of os made by the rewrite named name, or false.
func Find(os []Offered, name string) (Offered, bool) {
	for _, o := range os {
		if o.Rewrite.Name == name {
			return o, true
		}
	}
	return Offered{}, false
}
