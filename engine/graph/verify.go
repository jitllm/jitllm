package graph

import (
	"errors"
	"fmt"
	"sort"

	"github.com/jitllm/jitllm/format/jlm"
)

// Shape is a stored tensor's shape: K its input width (a vector's length),
// Rows its output rows (0 for a vector), Sheets a bank's experts (0 for one
// matrix).
type Shape struct{ K, Rows, Sheets int }

// Facts is what the container states about a block, independently of any
// graph built for it: the configuration (the rotary, the epsilon, the head
// counts) and the tensors stored for the block by role. The verifier holds a
// graph to them.
type Facts struct {
	Config    Dense
	Inventory map[jlm.Role]Shape
	// Delta, where set, is the gated delta sublayer the graph describes
	// instead of a dense block: its epsilon, value head width and gate are
	// what the norms and the act_mul are held to.
	Delta *Delta
	// Mixture, where set, is the routed mixture sublayer the graph
	// describes: its expert counts, router and activation are what the
	// route, the experts and the act_muls are held to.
	Mixture *Mixture
}

// Verify checks g before anything is allocated or emitted and returns every
// violation found, joined: the version; that every value has one producer
// and is defined before it is read; that widths agree with the weights each
// op binds; that every state read has its producing write ordered before it
// and every attention reads a version this block wrote; that every binding
// names a stored tensor of the shape the op needs and every stored tensor is
// bound by some op; that the rotary, norms and attention carry what the
// configuration states; and that the outputs are produced. It sizes nothing
// from the graph's counts.
func Verify(g *Graph, f Facts) error {
	var errs []error
	bad := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	if g == nil {
		return errors.New("graph: no graph")
	}
	if g.Version != Version {
		return fmt.Errorf("graph: block %d is op language v%d; this build reads v%d", g.Block, g.Version, Version)
	}
	n := len(g.Values)
	defined := make([]bool, n)
	valid := func(id ValueID) bool { return id >= 0 && int(id) < n }
	for _, id := range g.In {
		if !valid(id) {
			bad("graph: block input %d is not a value", id)
			continue
		}
		defined[id] = true
	}
	if len(g.In) == 0 {
		bad("graph: block %d has no input", g.Block)
	}
	width := func(id ValueID) int {
		if valid(id) {
			return g.Values[id].Width
		}
		return -1
	}
	state := map[string]int{}
	bound := map[jlm.Role]bool{}
	c := &f.Config
	eps, headDim, act := c.Eps, c.HeadDim, c.Act
	if f.Delta != nil {
		eps, headDim, act = f.Delta.Eps, f.Delta.VDim, f.Delta.Gate
	}
	if f.Mixture != nil {
		act = f.Mixture.Act
	}
	for i, o := range g.Ops {
		at := fmt.Sprintf("graph: block %d op %d (%s)", g.Block, i, o.Kind)
		if o.Kind <= OpInvalid || o.Kind >= numOps {
			bad("%s: unknown op", at)
			continue
		}
		for _, id := range o.In {
			if !valid(id) {
				bad("%s: input %d is not a value", at, id)
			} else if !defined[id] {
				bad("%s: reads %%%s before anything writes it", at, g.Values[id].Name)
			}
		}
		for _, id := range o.Out {
			if !valid(id) {
				bad("%s: output %d is not a value", at, id)
			} else if defined[id] {
				bad("%s: writes %%%s, which already has a producer", at, g.Values[id].Name)
			} else {
				defined[id] = true
			}
		}
		for _, r := range o.Reads {
			if state[r.Name] != r.Version {
				bad("%s: reads %s@%d, but %s@%d is the version in force here", at, r.Name, r.Version, r.Name, state[r.Name])
			}
		}
		for _, w := range o.Writes {
			if w.Version != state[w.Name]+1 {
				bad("%s: writes %s@%d over %s@%d", at, w.Name, w.Version, w.Name, state[w.Name])
			}
			state[w.Name] = w.Version
		}
		for _, w := range o.Weight {
			bound[w.Role] = true
			s, ok := f.Inventory[w.Role]
			switch {
			case !ok:
				bad("%s: binds %v, which the container does not store for the block", at, w.Role)
			case s.K != w.K || s.Rows != w.Rows || s.Sheets != w.Sheets:
				bad("%s: binds %v as %dx%dx%d, stored %dx%dx%d", at, w.Role, w.K, w.Rows, w.Sheets, s.K, s.Rows, s.Sheets)
			}
		}
		ins := func(k int) bool {
			if len(o.In) != k {
				bad("%s: %d inputs, wants %d", at, len(o.In), k)
				return false
			}
			return true
		}
		outs := func(k int) bool {
			if len(o.Out) != k {
				bad("%s: %d outputs, wants %d", at, len(o.Out), k)
				return false
			}
			return true
		}
		switch o.Kind {
		case OpNorm, OpHeadNorm:
			// A weightless head norm is Llama 4's, and only where the
			// configuration states it.
			if o.Kind == OpHeadNorm && len(o.Weight) == 0 && ins(1) && outs(1) {
				if !c.QKL2Norm || o.Attr.Eps != eps || o.Attr.HeadDim != headDim || headDim <= 0 ||
					width(o.In[0])%headDim != 0 || width(o.Out[0]) != width(o.In[0]) {
					bad("%s: a weightless head norm the configuration does not state", at)
				}
				break
			}
			if !ins(1) || !outs(1) || len(o.Weight) != 1 {
				bad("%s: wants one input, one output and one scale", at)
				break
			}
			if e := normEps(c, o, eps); o.Attr.Eps != e {
				bad("%s: epsilon %g, the configuration states %g", at, o.Attr.Eps, e)
			}
			if width(o.Out[0]) != width(o.In[0]) {
				bad("%s: width %d into %d", at, width(o.In[0]), width(o.Out[0]))
			}
			if o.Kind == OpNorm && o.Weight[0].K != width(o.In[0]) {
				bad("%s: scale of %d over a row of %d", at, o.Weight[0].K, width(o.In[0]))
			}
			if o.Kind == OpHeadNorm && (headDim <= 0 || o.Attr.HeadDim != headDim || o.Weight[0].K != headDim || width(o.In[0])%headDim != 0) {
				bad("%s: heads of %d with a scale of %d over a row of %d; the configuration states heads of %d",
					at, o.Attr.HeadDim, o.Weight[0].K, width(o.In[0]), headDim)
			}
		case OpMatVec:
			if !ins(1) || !outs(1) || len(o.Weight) == 0 || len(o.Weight) > 2 {
				bad("%s: wants one input, one output, a matrix and at most a bias", at)
				break
			}
			if w := o.Weight[0]; w.K != width(o.In[0]) || w.Rows != width(o.Out[0]) {
				bad("%s: %v is %dx%d between widths %d and %d", at, w.Role, w.K, w.Rows, width(o.In[0]), width(o.Out[0]))
			}
			if len(o.Weight) == 2 && (o.Weight[1].Rows != 0 || o.Weight[1].K != width(o.Out[0])) {
				bad("%s: bias %v of %d on a row of %d", at, o.Weight[1].Role, o.Weight[1].K, width(o.Out[0]))
			}
		case OpRope:
			if !ins(1) || !outs(1) {
				break
			}
			a := o.Attr
			// A latent attention turns its rotary parts alone: heads of NRot.
			hd := c.HeadDim
			if c.MLA != nil {
				hd = c.NRot
			}
			if a.Layout != c.Layout || a.Freqs != c.RopeFreqs || a.NRot != c.NRot || a.Base != c.Base || a.HeadDim != hd ||
				a.Mscale != c.Mscale || a.Sections != c.RopeSections || a.Interleaved != c.RopeInterleaved {
				bad("%s: %v pairs over %d of %d at base %g; the configuration states %v over %d of %d at %g",
					at, a.Layout, a.NRot, a.HeadDim, a.Base, c.Layout, c.NRot, hd, c.Base)
			}
			if width(o.In[0])%hd != 0 || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: a row of %d is not whole heads of %d", at, width(o.In[0]), hd)
			}
		case OpKVAppend:
			if !ins(2) || !outs(0) {
				break
			}
			if len(o.Writes) != 1 || o.Writes[0].Name != "kv" {
				bad("%s: writes no kv version", at)
			}
			if kvd := c.NKVHead * c.HeadDim; width(o.In[0]) != kvd || width(o.In[1]) != kvd {
				bad("%s: k of %d and v of %d into rows of %d", at, width(o.In[0]), width(o.In[1]), kvd)
			}
		case OpAttend:
			if !ins(1) || !outs(1) {
				break
			}
			a := o.Attr
			if len(o.Reads) != 1 || o.Reads[0].Name != "kv" || o.Reads[0].Version == 0 {
				bad("%s: reads no kv version this block wrote: this position's row is never in the history", at)
			}
			if a.NHead != c.NHead || a.NKVHead != c.NKVHead || a.HeadDim != c.HeadDim || a.Scale != c.Scale ||
				a.Window != c.Window || a.Chunked != c.Chunked || a.Softcap != c.Softcap || a.TempScale != c.TempScale ||
				a.TempFloor != c.TempFloor || a.TempOffs != c.TempOffs {
				bad("%s: %d/%d heads of %d, scale %g, window %d, softcap %g; the configuration states %d/%d of %d, %g, %d, %g",
					at, a.NHead, a.NKVHead, a.HeadDim, a.Scale, a.Window, a.Softcap,
					c.NHead, c.NKVHead, c.HeadDim, c.Scale, c.Window, c.Softcap)
			}
			if sk := len(o.Weight) == 1 && o.Weight[0].Role == jlm.RoleAttnSinks && o.Weight[0].K == c.NHead &&
				o.Weight[0].Rows == 0; sk != c.Sinks || len(o.Weight) > 1 {
				bad("%s: %d weights; the configuration states sinks %v", at, len(o.Weight), c.Sinks)
			}
			if width(o.In[0]) != c.NHead*c.HeadDim || width(o.Out[0]) != c.NHead*c.HeadDim {
				bad("%s: q of %d, out of %d", at, width(o.In[0]), width(o.Out[0]))
			}
		case OpAdd, OpActMul:
			if !ins(2) || !outs(1) {
				break
			}
			if width(o.In[0]) != width(o.In[1]) || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: widths %d and %d into %d", at, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
			}
			// A residual add (into the block's stream) carries the
			// configuration's scale; the mixture's own add of its shared
			// expert carries none.
			if o.Kind == OpAdd && o.Attr.Alpha != 0 && o.Attr.Alpha != c.ResidScale {
				bad("%s: residual scale %g, the configuration states %g", at, o.Attr.Alpha, c.ResidScale)
			}
			if o.Kind == OpActMul && o.Attr.Act != act && !(c.OutGate && o.Attr.Act == ActSigmoid) {
				bad("%s: activation %d, the configuration states %d", at, o.Attr.Act, act)
			}
		case OpLayerNorm:
			if !ins(1) || !outs(1) || len(o.Weight) == 0 || len(o.Weight) > 2 {
				bad("%s: wants one input, one output, a scale and at most a bias", at)
				break
			}
			if o.Attr.Eps != eps {
				bad("%s: epsilon %g, the configuration states %g", at, o.Attr.Eps, eps)
			}
			for _, w := range o.Weight {
				if w.K != width(o.In[0]) || w.Rows != 0 || width(o.Out[0]) != width(o.In[0]) {
					bad("%s: %v of %d over a row of %d into %d", at, w.Role, w.K, width(o.In[0]), width(o.Out[0]))
				}
			}
		case OpAct:
			if !ins(1) || !outs(1) {
				break
			}
			if width(o.Out[0]) != width(o.In[0]) {
				bad("%s: width %d into %d", at, width(o.In[0]), width(o.Out[0]))
			}
			if o.Attr.Act != act {
				bad("%s: activation %d, the configuration states %d", at, o.Attr.Act, act)
			}
			if xi := o.Attr.Act == ActXIELU; xi != (len(o.Weight) == 1) || len(o.Weight) > 1 ||
				xi && (o.Weight[0].Role != jlm.RoleXIELU || o.Weight[0].K != 4) {
				bad("%s: activation %d binds %d weights", at, o.Attr.Act, len(o.Weight))
			}
		case OpRecurrent:
			if !ins(1) || !outs(1) {
				break
			}
			if o.Attr.Mixer == 0 || o.Attr.Mixer != c.Mixer || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: mixer %d over %d into %d; the configuration states mixer %d", at, o.Attr.Mixer,
					width(o.In[0]), width(o.Out[0]), c.Mixer)
			}
			if len(o.Reads) != 1 || o.Reads[0].Name != "rec" || len(o.Writes) != 1 || o.Writes[0].Name != "rec" {
				bad("%s: reads and writes no recurrence version", at)
			}
		case OpClamp:
			if !ins(1) || !outs(1) {
				break
			}
			if o.Attr.Bound <= 0 || o.Attr.Bound != c.Clamp || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: bound %g, the configuration states %g", at, o.Attr.Bound, c.Clamp)
			}
		case OpSlice, OpLatentAppend, OpAttendLatent:
			verifyMLAOp(o, at, c, width, ins, outs, bad)
		case OpRoute, OpGather, OpExpertMatVec, OpCombine, OpGateScalar:
			verifyMixtureOp(o, at, f.Mixture, width, ins, outs, bad)
		case OpGatedDelta:
			if !ins(1) || !outs(1) {
				break
			}
			d := f.Delta
			if d == nil {
				bad("%s: a gated delta rule in a graph whose facts state none", at)
				break
			}
			if width(o.In[0]) != d.NEmbd || width(o.Out[0]) != d.VHeads*d.VDim {
				bad("%s: a row of %d into %d; the configuration states %d into %d value heads of %d",
					at, width(o.In[0]), width(o.Out[0]), d.NEmbd, d.VHeads, d.VDim)
			}
			if len(o.Reads) != 1 || o.Reads[0].Name != "rec" || len(o.Writes) != 1 || o.Writes[0].Name != "rec" {
				bad("%s: reads and writes no recurrence version", at)
			}
		}
	}
	// A latent attention turns both its rotary parts, or neither (NoPE).
	if c.MLA != nil {
		ropes := 0
		for _, o := range g.Ops {
			if o.Kind == OpRope {
				ropes++
			}
		}
		if want := map[bool]int{false: 2, true: 0}[c.MLA.NoRope]; ropes != want {
			bad("graph: block %d turns %d rotary parts; the configuration states %d", g.Block, ropes, want)
		}
	}
	for _, id := range g.Out {
		if !valid(id) || !defined[id] {
			bad("graph: block %d output %d is never produced", g.Block, id)
		}
	}
	if len(g.Out) == 0 {
		bad("graph: block %d has no output", g.Block)
	}
	var unbound []string
	for r := range f.Inventory {
		if !bound[r] {
			unbound = append(unbound, r.String())
		}
	}
	sort.Strings(unbound)
	for _, r := range unbound {
		bad("graph: block %d stores %s, which no op binds: the graph does not compute what the container holds", g.Block, r)
	}
	return errors.Join(errs...)
}

// verifyMixtureOp holds a mixture op to the sublayer's facts: the expert
// counts and the router the configuration states, banks of the bank's
// experts, and the pairs' widths.
func verifyMixtureOp(o Op, at string, m *Mixture, width func(ValueID) int, ins, outs func(int) bool,
	bad func(string, ...any)) {
	if m == nil {
		bad("%s: a mixture op in a graph whose facts state no mixture", at)
		return
	}
	switch o.Kind {
	case OpRoute:
		if !ins(1) || !outs(1) || len(o.Weight) == 0 {
			return
		}
		a := o.Attr
		if a.Experts != m.NExpert || a.TopK != m.K || a.Router != m.Router || a.RouteNorm != m.Norm ||
			a.Scale != m.Scale || a.Groups != m.Groups || a.GroupsUsed != m.GroupsUsed {
			bad("%s: top %d of %d by router %d (norm %v, scale %g, groups %d/%d); the configuration states top %d of %d by %d (%v, %g, %d/%d)",
				at, a.TopK, a.Experts, a.Router, a.RouteNorm, a.Scale, a.GroupsUsed, a.Groups,
				m.K, m.NExpert, m.Router, m.Norm, m.Scale, m.GroupsUsed, m.Groups)
		}
		if w := o.Weight[0]; w.Role != jlm.RoleRouter || w.K != width(o.In[0]) || w.Rows != m.NExpert {
			bad("%s: router %v of %dx%d over a row of %d and %d experts", at, w.Role, w.K, w.Rows, width(o.In[0]), m.NExpert)
		}
		for _, w := range o.Weight[1:] {
			// A vector, stored with no second dimension or as one row.
			if (w.Role != jlm.RoleRouterBias && w.Role != jlm.RoleExpProbsB) || w.K != m.NExpert || w.Rows > 1 {
				bad("%s: %v of %d is not a bias on %d experts' logits", at, w.Role, w.K, m.NExpert)
			}
		}
		if width(o.Out[0]) != m.K {
			bad("%s: a selection of %d pairs, top %d", at, width(o.Out[0]), m.K)
		}
	case OpGather:
		if ins(2) && outs(1) && width(o.Out[0]) != m.K*width(o.In[0]) {
			bad("%s: %d pairs of a row of %d into %d", at, m.K, width(o.In[0]), width(o.Out[0]))
		}
		if o.Attr.WeightIn != m.WeightIn {
			bad("%s: weight-in %v; the configuration states %v", at, o.Attr.WeightIn, m.WeightIn)
		}
	case OpExpertMatVec:
		if !ins(2) || !outs(1) || len(o.Weight) == 0 || len(o.Weight) > 2 {
			bad("%s: wants pairs, a selection, a bank and at most its bias", at)
			return
		}
		w := o.Weight[0]
		if w.Sheets != m.NExpert || m.K*w.K != width(o.In[0]) || m.K*w.Rows != width(o.Out[0]) {
			bad("%s: bank %v of %d sheets of %dx%d between %d pairs of %d and %d; the configuration states %d experts",
				at, w.Role, w.Sheets, w.K, w.Rows, m.K, width(o.In[0]), width(o.Out[0]), m.NExpert)
		}
		if len(o.Weight) == 2 && (o.Weight[1].K != w.Rows || o.Weight[1].Rows != m.NExpert) {
			bad("%s: bias %v of %dx%d on a bank of %d rows and %d experts", at, o.Weight[1].Role,
				o.Weight[1].K, o.Weight[1].Rows, w.Rows, m.NExpert)
		}
	case OpCombine:
		if !ins(2) || !outs(1) {
			return
		}
		if o.Attr.WeightIn != m.WeightIn {
			bad("%s: weight-in %v; the configuration states %v", at, o.Attr.WeightIn, m.WeightIn)
		}
		if o.Attr.Experts != m.NExpert || o.Attr.TopK != m.K || width(o.In[0]) != m.K*width(o.Out[0]) {
			bad("%s: %d pairs of %d experts from %d into %d", at, o.Attr.TopK, o.Attr.Experts, width(o.In[0]), width(o.Out[0]))
		}
		for _, w := range o.Weight {
			if w.Role != jlm.RoleExpScale || w.K != m.NExpert {
				bad("%s: %v of %d is not a scale per expert", at, w.Role, w.K)
			}
		}
	case OpGateScalar:
		if !ins(2) || !outs(1) || len(o.Weight) != 1 {
			return
		}
		// The gate is a vector, stored with no second dimension or as one row.
		if w := o.Weight[0]; w.K != width(o.In[0]) || w.Rows > 1 || width(o.In[1]) != width(o.Out[0]) {
			bad("%s: a gate of %d over a row of %d scaling %d into %d", at, w.K, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
		}
	}
}

// Unbound is every tensor role f's inventory stores that no op of g binds,
// sorted. Verify refuses such a graph; a frontend asks first, so a block
// whose container holds what the op language does not state is named as such
// (the structural reason) rather than refused as a bug.
func Unbound(g *Graph, f Facts) []jlm.Role {
	bound := map[jlm.Role]bool{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			bound[w.Role] = true
		}
	}
	var out []jlm.Role
	for r := range f.Inventory {
		if !bound[r] {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// normEps is the epsilon an RMSNorm is held to: a latent attention's norms
// over its q and kv latents state their own (Config.LatentNormEps).
func normEps(c *Dense, o Op, eps float64) float64 {
	if c.MLA != nil && len(o.Weight) == 1 && (o.Weight[0].Role == jlm.RoleAttnQANorm || o.Weight[0].Role == jlm.RoleAttnKVANorm) {
		return c.MLA.LatentEps
	}
	return eps
}
