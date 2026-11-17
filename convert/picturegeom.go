package convert

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// A processor's picture steps and the rules that state a picture's geometry
// from them are the projector library's (projectors.json's pictures). A
// stater writes a processor's own steps by name; the container holds none
// of them: the rule that reads exactly the steps a picture takes states its
// geometry (jlm.PictureGeometry) and lowers them to the container's generic
// steps -- a resize to a fixed window (jlm.PreResize) or to each square's
// own size (jlm.PreResizeLive), a canvas (jlm.PreCanvas). Every other step
// (the normalisation) is kept as it is.

// mjPictures is projectors.json's pictures: the processors' own steps by
// name, and the rules.
type mjPictures struct {
	Steps map[string]*mjPicStep `json:"steps"`
	Rules map[string]*mjPicRule `json:"rules"`
}

// mjPicStep is a step a rule reads: Code its kind where it is a processor's
// own (0 for the container's step of that name), and the names of its ints
// and floats, one ending in ? read as 0 where the step omits it.
// LongestEdge names the int a processor config's size.longest_edge
// restates.
type mjPicStep struct {
	Code        int      `json:"code"`
	Ints        []string `json:"ints"`
	Floats      []string `json:"floats"`
	LongestEdge string   `json:"longest_edge"`
}

// mjPicRule is one processor's rule: the steps it reads (a picture takes it
// where these are exactly the steps of any rule it takes), what their
// numbers must satisfy, constants, definitions (an expression written
// wherever its name is read), the bindings in order, the squares' count and
// fields, the whole-picture stages, the bounds and the container's steps
// the read ones lower to, in the place of the first.
//
// Its expressions are model.json's language over the steps' numbers (kind.
// name), the tower's facts (patch, side, marks, widest), the resize names
// (filter.*, prec.*), and, where the picture is read, its operands (w, h,
// i the square, squares their count, frames) and the bindings. A part of an
// expression that reads no operand is evaluated at conversion; ?: and where
// of a constant condition take their branch there.
type mjPicRule struct {
	Doc     string            `json:"doc"`
	Reads   []string          `json:"reads"`
	Require []mjRequire       `json:"require"`
	Const   []mjPicConst      `json:"const"`
	Def     [][2]string       `json:"def"`
	Bind    []mjPicBind       `json:"bind"`
	Count   string            `json:"count"`
	Square  map[string]string `json:"square"`
	Stages  []mjPicStage      `json:"stages"`
	Bounds  mjPicBounds       `json:"bounds"`
	Lower   []mjPreStep       `json:"lower"`
}

// mjPicConst is a constant: Value, or the largest Max over the loops Each.
type mjPicConst struct {
	Let   string      `json:"let"`
	Value string      `json:"value"`
	Max   string      `json:"max"`
	Each  [][4]string `json:"each"`
}

// mjPicBind is a binding: Let named Value (a value that is an operand or a
// constant names it, and binds nothing); a row of the table Rows chosen as
// jlm.PictureChoice states, its columns named Choose and read by Eligible,
// Cost and Tie as the candidate row's; or Bind once for each turn of the
// loops Each.
type mjPicBind struct {
	Let      string      `json:"let"`
	Value    string      `json:"value"`
	Choose   []string    `json:"choose"`
	Rows     *mjPicTable `json:"rows"`
	Eligible string      `json:"eligible"`
	Cost     []string    `json:"cost"`
	Tie      string      `json:"tie"`
	Each     [][4]string `json:"each"`
	Bind     []mjPicBind `json:"bind"`
}

// mjPicTable is a table of constant rows: Row for each turn of the loops
// Each where If holds, grouped by Group's value where stated, each group in
// the order its first row came. A loop is its variable, its first value,
// the condition it runs while and its next value.
type mjPicTable struct {
	Each  [][4]string `json:"each"`
	If    string      `json:"if"`
	Row   []string    `json:"row"`
	Group string      `json:"group"`
}

// mjPicStage is a whole-picture resize (jlm.PictureStage), stated where If
// holds.
type mjPicStage struct {
	W      string `json:"w"`
	H      string `json:"h"`
	MaxPix string `json:"max_pix"`
	If     string `json:"if"`
}

type mjPicBounds struct {
	Unit       string `json:"unit"`
	MaxPix     string `json:"max_pix"`
	MaxPatches string `json:"max_patches"`
}

// Bounds on what a rule's loops and tables run to, so a statement sizes
// nothing unchecked (RULE 9).
const (
	maxPicTurns = 1 << 16
	maxPicDepth = 256
)

// picContext is the tower's facts a rule reads: the entry's patch, a fixed
// square's grid side and whether the head writes marker rows.
type picContext struct {
	patch, side int
	marks       bool
}

// pictureLibrary is projectors.json's pictures, checked: every rule reads
// declared steps, no two read the same ones, and every expression parses.
func pictureLibrary() (*mjPictures, error) {
	var lib struct {
		Pictures mjPictures `json:"pictures"`
	}
	if err := json.Unmarshal(projectorsJSON, &lib); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: %w", err)
	}
	l := &lib.Pictures
	if err := l.check(); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: pictures: %w", err)
	}
	return l, nil
}

// PictureStep is the picture step a stater writes under name: the
// container's own or a processor's (projectors.json's pictures).
func PictureStep(name string) (jlm.PreKind, error) {
	kinds, err := towerSteps()
	if err != nil {
		return 0, err
	}
	k, ok := kinds[name]
	if !ok {
		return 0, fmt.Errorf("convert: no picture step is named %q", name)
	}
	return k, nil
}

// containerSteps is the container's own picture steps by name.
func containerSteps() map[string]jlm.PreKind {
	return map[string]jlm.PreKind{
		"resize": jlm.PreResize, "normalize": jlm.PreNormalize, "normalize_bf16": jlm.PreNormalizeBF16,
		"resize_live": jlm.PreResizeLive, "canvas": jlm.PreCanvas,
	}
}

// kinds is every step a stater may write by name: the container's and the
// processors'.
func (l *mjPictures) kinds() map[string]jlm.PreKind {
	out := containerSteps()
	for n, s := range l.Steps {
		if s.Code != 0 {
			out[n] = jlm.PreKind(s.Code)
		}
	}
	return out
}

func (l *mjPictures) check() error {
	own := containerSteps()
	codes := map[int]string{}
	for _, n := range slices.Sorted(maps.Keys(l.Steps)) {
		s := l.Steps[n]
		_, generic := own[n]
		switch {
		case s.Code == 0 && !generic:
			return fmt.Errorf("step %s is no container step and states no code", n)
		case s.Code != 0 && (generic || s.Code < 100 || s.Code > math.MaxUint8):
			return fmt.Errorf("step %s: a processor's step is a code of 100 to 255 under a name of its own", n)
		case codes[s.Code] != "" && s.Code != 0:
			return fmt.Errorf("steps %s and %s share code %d", codes[s.Code], n, s.Code)
		}
		codes[s.Code] = n
		if s.LongestEdge != "" && !slices.Contains(s.Ints, s.LongestEdge) {
			return fmt.Errorf("step %s: its longest edge %q is none of its ints", n, s.LongestEdge)
		}
	}
	seen := map[string]string{}
	for _, n := range slices.Sorted(maps.Keys(l.Rules)) {
		r := l.Rules[n]
		if r.Doc == "" || len(r.Reads) == 0 || r.Count == "" {
			return fmt.Errorf("rule %s states what it is, the steps it reads and its count", n)
		}
		for _, k := range r.Reads {
			if l.Steps[k] == nil {
				return fmt.Errorf("rule %s reads step %s, which is not declared", n, k)
			}
		}
		key := strings.Join(slices.Sorted(slices.Values(r.Reads)), " ")
		if seen[key] != "" {
			return fmt.Errorf("rules %s and %s read the same steps", seen[key], n)
		}
		seen[key] = n
		for f := range r.Square {
			if _, ok := squareFields()[f]; !ok {
				return fmt.Errorf("rule %s: %q is no square field", n, f)
			}
		}
		for _, s := range r.Lower {
			if _, ok := own[s.Kind]; !ok {
				return fmt.Errorf("rule %s lowers to %q, which is no container step", n, s.Kind)
			}
		}
		for _, src := range r.exprs() {
			if _, err := parseExpr(src); err != nil {
				return fmt.Errorf("rule %s: %w", n, err)
			}
		}
	}
	return nil
}

// exprs is every expression r states.
func (r *mjPicRule) exprs() []string {
	out := []string{r.Count, r.Bounds.Unit, r.Bounds.MaxPix, r.Bounds.MaxPatches}
	for _, q := range r.Require {
		out = append(out, q.That)
	}
	for _, c := range r.Const {
		out = append(append(out, c.Value, c.Max), loopExprs(c.Each)...)
	}
	for _, d := range r.Def {
		out = append(out, d[1])
	}
	var binds func(bs []mjPicBind)
	binds = func(bs []mjPicBind) {
		for _, b := range bs {
			out = append(append(append(out, b.Value, b.Eligible, b.Tie), b.Cost...), loopExprs(b.Each)...)
			if t := b.Rows; t != nil {
				out = append(append(append(out, t.If, t.Group), t.Row...), loopExprs(t.Each)...)
			}
			binds(b.Bind)
		}
	}
	binds(r.Bind)
	for _, f := range slices.Sorted(maps.Keys(r.Square)) {
		out = append(out, r.Square[f])
	}
	for _, s := range r.Stages {
		out = append(out, s.W, s.H, s.MaxPix, s.If)
	}
	for _, s := range r.Lower {
		out = append(append(out, s.Ints...), s.Floats...)
	}
	return slices.DeleteFunc(out, func(s string) bool { return s == "" })
}

func loopExprs(each [][4]string) []string {
	var out []string
	for _, l := range each {
		out = append(out, l[1], l[2], l[3])
	}
	return out
}

// squareFields is a square's fields (jlm.SqOw, ...) by name.
func squareFields() map[string]int {
	return map[string]int{"ow": jlm.SqOw, "oh": jlm.SqOh, "x0": jlm.SqX0, "y0": jlm.SqY0, "pw": jlm.SqPw, "ph": jlm.SqPh,
		"gw": jlm.SqGw, "gh": jlm.SqGh, "cw": jlm.SqCw, "ch": jlm.SqCh, "cx": jlm.SqCx, "cy": jlm.SqCy,
		"mark": jlm.SqMark, "mark_n": jlm.SqMarkN, "trail": jlm.SqTrail, "trail_n": jlm.SqTrailN}
}

// squareDefaults is what a field a rule does not state is: the window and
// the canvas the whole resize, nothing else.
func squareDefaults() map[string]string {
	return map[string]string{"pw": "ow", "ph": "oh", "cw": "ow", "ch": "oh"}
}

// pictureOperands is the operands a picture's expressions read by name.
func pictureOperands() map[string]jlm.Arg {
	return map[string]jlm.Arg{"w": jlm.Operand(jlm.PicW), "h": jlm.Operand(jlm.PicH), "i": jlm.Operand(jlm.PicSquare),
		"squares": jlm.Operand(jlm.PicCount), "frames": jlm.Operand(jlm.PicFrames)}
}

// towerPicture is the geometry of tower t's picture under steps (its own,
// or a video's frames read as pictures), and steps lowered: read in the
// entry's patches, a fixed square's grid side x side, with the markers
// where the head writes them.
func towerPicture(l *mjPictures, t *jlm.Tower, v *jlm.Vision, steps []jlm.PreStep) (*jlm.PictureGeometry, []jlm.PreStep, error) {
	ops := t.Entry.Ops
	if len(ops) == 0 || ops[0].Code != jlm.TOpPatchify || ops[0].Span < 1 || v.PatchSize == 0 {
		return nil, nil, fmt.Errorf("convert: a picture's geometry of an entry that does not begin with its patches")
	}
	c := picContext{patch: int(ops[0].Span), side: int(v.ImageSize / v.PatchSize)}
	if len(t.Chain) > 0 {
		c.side = int(v.ImageSize) / c.patch
	}
	for _, op := range t.Head.Ops {
		c.marks = c.marks || op.Code == jlm.TOpMarkRow
	}
	return l.picture(steps, c)
}

// videoFrames is the geometry of video v's frames where it reads them
// through an entry of its own, read in its patches, and its steps lowered.
func videoFrames(l *mjPictures, v *jlm.Video) (*jlm.PictureGeometry, []jlm.PreStep, error) {
	if len(v.Entry.Ops) == 0 || v.Entry.Ops[0].Code != jlm.TOpPatchify || v.Entry.Ops[0].Span < 1 {
		return nil, nil, fmt.Errorf("convert: a video's frames' geometry of an entry that does not begin with their patches")
	}
	g, pre, err := l.picture(v.Pre, picContext{patch: int(v.Entry.Ops[0].Span)})
	if err != nil {
		return nil, nil, fmt.Errorf("convert: the video's frames: %w", err)
	}
	return g, pre, nil
}

// restateLongestEdge sets the int each step's kind names its longest edge
// to edge, a processor config's size.longest_edge.
func (l *mjPictures) restateLongestEdge(steps []jlm.PreStep, edge int32) {
	byKind := map[jlm.PreKind]*mjPicStep{}
	for _, s := range l.Steps {
		if s.Code != 0 {
			byKind[jlm.PreKind(s.Code)] = s
		}
	}
	for i := range steps {
		if d := byKind[steps[i].Kind]; d != nil && d.LongestEdge != "" {
			if k := slices.Index(d.Ints, d.LongestEdge); k < len(steps[i].Ints) {
				steps[i].Ints[k] = edge
			}
		}
	}
}

// picture is the geometry the rule reading exactly steps' processor steps
// states, and steps lowered.
func (l *mjPictures) picture(steps []jlm.PreStep, c picContext) (*jlm.PictureGeometry, []jlm.PreStep, error) {
	read := map[string]bool{}
	for _, r := range l.Rules {
		for _, k := range r.Reads {
			read[k] = true
		}
	}
	names := map[jlm.PreKind]string{}
	for n, k := range l.kinds() {
		names[k] = n
	}
	at := map[string]int{}
	for i, s := range steps {
		n := names[s.Kind]
		if !read[n] {
			continue
		}
		if _, twice := at[n]; twice {
			return nil, nil, fmt.Errorf("convert: a picture takes two %s steps", n)
		}
		at[n] = i
	}
	taken := strings.Join(slices.Sorted(maps.Keys(at)), " ")
	for _, rn := range slices.Sorted(maps.Keys(l.Rules)) {
		r := l.Rules[rn]
		if strings.Join(slices.Sorted(slices.Values(r.Reads)), " ") != taken {
			continue
		}
		env := newPicEnv(c)
		first := len(steps)
		for _, k := range r.Reads {
			first = min(first, at[k])
			if err := env.params(k, l.Steps[k], &steps[at[k]]); err != nil {
				return nil, nil, fmt.Errorf("convert: picture rule %s: %w", rn, err)
			}
		}
		g, err := env.state(r)
		if err != nil {
			return nil, nil, fmt.Errorf("convert: picture rule %s: %w", rn, err)
		}
		lowered, err := env.lower(r.Lower)
		if err != nil {
			return nil, nil, fmt.Errorf("convert: picture rule %s: %w", rn, err)
		}
		var out []jlm.PreStep
		for i, s := range steps {
			if i == first {
				out = append(out, lowered...)
			}
			if read[names[s.Kind]] {
				continue
			}
			out = append(out, jlm.PreStep{Kind: s.Kind, Ints: slices.Clone(s.Ints), Floats: slices.Clone(s.Floats)})
		}
		return g, out, nil
	}
	if taken == "" {
		return nil, nil, fmt.Errorf("convert: a picture never resized to the tower's size")
	}
	return nil, nil, fmt.Errorf("convert: no picture rule of the library reads exactly the steps %s", taken)
}

// picEnv is a rule's scope while it is stated: constants by name (the
// steps' numbers, the tower's facts, the resize names, constants and the
// loops' variables), definitions, and the operands names read.
type picEnv struct {
	consts map[string]float64
	defs   map[string]*exprNode
	args   map[string]jlm.Arg
	depth  int
}

func newPicEnv(c picContext) *picEnv {
	e := &picEnv{consts: map[string]float64{"patch": float64(c.patch), "side": float64(c.side), "widest": widestPicture},
		defs: map[string]*exprNode{}, args: pictureOperands()}
	if c.marks {
		e.consts["marks"] = 1
	} else {
		e.consts["marks"] = 0
	}
	for k, f := range map[string]jlm.Filter{"two_tap": jlm.FilterTwoTap, "pil_bilinear": jlm.FilterPILBilinear,
		"pil_bicubic": jlm.FilterPILBicubic, "pil_lanczos": jlm.FilterPILLanczos} {
		e.consts["filter."+k] = float64(f)
	}
	for k, p := range map[string]jlm.Precision{"float": jlm.PrecFloat, "pil": jlm.PrecPIL, "torch": jlm.PrecTorch} {
		e.consts["prec."+k] = float64(p)
	}
	return e
}

// params names step s's numbers kind.name as d declares them.
func (e *picEnv) params(kind string, d *mjPicStep, s *jlm.PreStep) error {
	bind := func(names []string, n int, at func(i int) float64) error {
		need := 0
		for _, nm := range names {
			if !strings.HasSuffix(nm, "?") {
				need++
			}
		}
		if n < need || n > len(names) {
			return fmt.Errorf("a %s step of %d numbers where it states %d to %d", kind, n, need, len(names))
		}
		for i, nm := range names {
			v := 0.0
			if i < n {
				v = at(i)
			}
			e.consts[kind+"."+strings.TrimSuffix(nm, "?")] = v
		}
		return nil
	}
	if err := bind(d.Ints, len(s.Ints), func(i int) float64 { return float64(s.Ints[i]) }); err != nil {
		return err
	}
	return bind(d.Floats, len(s.Floats), func(i int) float64 { return float64(s.Floats[i]) })
}

// state is the geometry r states.
func (e *picEnv) state(r *mjPicRule) (*jlm.PictureGeometry, error) {
	for _, c := range r.Const {
		v, err := e.constant(c)
		if err != nil {
			return nil, fmt.Errorf("const %s: %w", c.Let, err)
		}
		e.consts[c.Let] = v
	}
	for _, q := range r.Require {
		v, err := e.number(q.That)
		if err != nil {
			return nil, err
		}
		if !(v > 0) {
			return nil, fmt.Errorf("%s", q.Else)
		}
	}
	for _, d := range r.Def {
		n, err := parseExpr(d[1])
		if err != nil {
			return nil, err
		}
		e.defs[d[0]] = n
	}
	g := &jlm.PictureGeometry{}
	n := uint32(jlm.PicBound)
	if err := e.binds(g, &n, r.Bind); err != nil {
		return nil, err
	}
	var err error
	if g.Count, err = e.expr(r.Count); err != nil {
		return nil, fmt.Errorf("count: %w", err)
	}
	src := maps.Clone(r.Square)
	for _, f := range []string{"ow", "oh", "gw", "gh"} {
		if src[f] == "" {
			return nil, fmt.Errorf("the square's %s is not stated", f)
		}
	}
	for f, d := range squareDefaults() {
		if src[f] == "" {
			src[f] = src[d]
		}
	}
	for f, k := range squareFields() {
		s := src[f]
		if s == "" {
			s = "0"
		}
		if g.Square[k], err = e.expr(s); err != nil {
			return nil, fmt.Errorf("square %s: %w", f, err)
		}
	}
	for _, s := range r.Stages {
		if s.If != "" {
			on, err := e.number(s.If)
			if err != nil {
				return nil, err
			}
			if !(on > 0) {
				continue
			}
		}
		var st jlm.PictureStage
		if st.W, err = e.expr(s.W); err != nil {
			return nil, err
		}
		if st.H, err = e.expr(s.H); err != nil {
			return nil, err
		}
		px, err := e.whole(s.MaxPix)
		if err != nil {
			return nil, err
		}
		st.MaxPix = px
		g.Stages = append(g.Stages, st)
	}
	b := &g.Bounds
	for _, f := range []struct {
		to  *int32
		src string
	}{{&b.Unit, r.Bounds.Unit}, {&b.MaxPix, r.Bounds.MaxPix}, {&b.MaxPatches, r.Bounds.MaxPatches}} {
		if *f.to, err = e.whole(f.src); err != nil {
			return nil, fmt.Errorf("bounds: %w", err)
		}
	}
	return g, nil
}

// lower is the container's steps ss states.
func (e *picEnv) lower(ss []mjPreStep) ([]jlm.PreStep, error) {
	var out []jlm.PreStep
	for _, s := range ss {
		step := jlm.PreStep{Kind: containerSteps()[s.Kind]}
		for _, src := range s.Ints {
			v, err := e.whole(src)
			if err != nil {
				return nil, err
			}
			step.Ints = append(step.Ints, v)
		}
		for _, src := range s.Floats {
			v, err := e.number(src)
			if err != nil {
				return nil, err
			}
			step.Floats = append(step.Floats, float32(v))
		}
		out = append(out, step)
	}
	return out, nil
}

// binds states bs's bindings into g, n the next operand.
func (e *picEnv) binds(g *jlm.PictureGeometry, n *uint32, bs []mjPicBind) error {
	for _, b := range bs {
		switch {
		case b.Each != nil:
			if err := e.loop(b.Each, func() error { return e.binds(g, n, b.Bind) }); err != nil {
				return err
			}
		case b.Choose != nil:
			if err := e.choose(g, n, &b); err != nil {
				return fmt.Errorf("choice of %v: %w", b.Choose, err)
			}
		default:
			if b.Let == "" {
				return fmt.Errorf("a binding of no name")
			}
			if _, ok := pictureOperands()[b.Let]; ok {
				return fmt.Errorf("binding %s names a picture operand", b.Let)
			}
			var x sizes
			a, err := e.lowered(&x, b.Value)
			if err != nil {
				return fmt.Errorf("%s: %w", b.Let, err)
			}
			if len(x.T) == 0 {
				e.args[b.Let] = a
				continue
			}
			g.Bind = append(g.Bind, jlm.PictureBind{Value: finished(&x, a)})
			e.args[b.Let] = jlm.Operand(*n)
			*n++
		}
	}
	return nil
}

// choose binds a row of b's table, its columns the operands from n.
func (e *picEnv) choose(g *jlm.PictureGeometry, n *uint32, b *mjPicBind) error {
	if b.Rows == nil {
		return fmt.Errorf("a choice of no table")
	}
	rows, err := e.table(b.Rows)
	if err != nil {
		return err
	}
	cols := len(b.Choose)
	if len(rows) == 0 || cols == 0 {
		return fmt.Errorf("a table of %d rows of %d columns", len(rows), cols)
	}
	c := &jlm.PictureChoice{Cols: int32(cols)}
	for _, r := range rows {
		if len(r) != cols {
			return fmt.Errorf("a row of %d values where the choice names %d", len(r), cols)
		}
		c.Rows = append(c.Rows, r...)
	}
	for j, nm := range b.Choose {
		e.args[nm] = jlm.Operand(*n + uint32(j))
	}
	of := func(src string) (jlm.Expr, error) {
		if src == "" {
			return jlm.Expr{}, nil
		}
		return e.expr(src)
	}
	if c.Eligible, err = of(b.Eligible); err != nil {
		return err
	}
	if c.Tie, err = of(b.Tie); err != nil {
		return err
	}
	for _, src := range b.Cost {
		x, err := of(src)
		if err != nil {
			return err
		}
		c.Cost = append(c.Cost, x)
	}
	g.Bind = append(g.Bind, jlm.PictureBind{Choice: c})
	*n += uint32(cols)
	return nil
}

// table is t's rows.
func (e *picEnv) table(t *mjPicTable) ([][]float64, error) {
	var keys []float64
	var rows [][]float64
	err := e.loop(t.Each, func() error {
		if t.If != "" {
			ok, err := e.number(t.If)
			if err != nil || !(ok > 0) {
				return err
			}
		}
		row := make([]float64, len(t.Row))
		for j, src := range t.Row {
			v, err := e.number(src)
			if err != nil {
				return err
			}
			row[j] = v
		}
		key := 0.0
		if t.Group != "" {
			v, err := e.number(t.Group)
			if err != nil {
				return err
			}
			key = v
		}
		if len(rows) >= maxPicTurns {
			return fmt.Errorf("a table past %d rows", maxPicTurns)
		}
		keys, rows = append(keys, key), append(rows, row)
		return nil
	})
	if err != nil || t.Group == "" {
		return rows, err
	}
	var order []float64
	for _, k := range keys {
		if !slices.Contains(order, k) {
			order = append(order, k)
		}
	}
	var out [][]float64
	for _, k := range order {
		for i, r := range rows {
			if keys[i] == k {
				out = append(out, r)
			}
		}
	}
	return out, nil
}

// loop runs body for every turn of each's loops, outermost first, each
// variable a constant while its loop runs.
func (e *picEnv) loop(each [][4]string, body func() error) error {
	if len(each) == 0 {
		return body()
	}
	l := each[0]
	was, had := e.consts[l[0]]
	defer func() {
		if had {
			e.consts[l[0]] = was
		} else {
			delete(e.consts, l[0])
		}
	}()
	v, err := e.number(l[1])
	if err != nil {
		return err
	}
	for turn := 0; ; turn++ {
		if turn >= maxPicTurns {
			return fmt.Errorf("loop %s past %d turns", l[0], maxPicTurns)
		}
		e.consts[l[0]] = v
		on, err := e.number(l[2])
		if err != nil {
			return err
		}
		if !(on > 0) {
			return nil
		}
		if err := e.loop(each[1:], body); err != nil {
			return err
		}
		e.consts[l[0]] = v
		if v, err = e.number(l[3]); err != nil {
			return err
		}
	}
}

// constant is c's value.
func (e *picEnv) constant(c mjPicConst) (float64, error) {
	if c.Each == nil {
		return e.number(c.Value)
	}
	best, found := 0.0, false
	err := e.loop(c.Each, func() error {
		v, err := e.number(c.Max)
		if err == nil && (!found || v > best) {
			best, found = v, true
		}
		return err
	})
	if err == nil && !found {
		err = fmt.Errorf("a largest value over no turn")
	}
	return best, err
}

// number is src's value at conversion, an error where it reads an operand.
func (e *picEnv) number(src string) (float64, error) {
	n, err := parseExpr(src)
	if err != nil {
		return 0, err
	}
	v, ok, err := e.fold(n)
	if err == nil && !ok {
		err = fmt.Errorf("%q reads the picture, where a constant belongs", src)
	}
	return v, err
}

// whole is number(src) as an int32, an error where it is not one.
func (e *picEnv) whole(src string) (int32, error) {
	v, err := e.number(src)
	if err == nil && (v != math.Trunc(v) || math.Abs(v) > math.MaxInt32) {
		err = fmt.Errorf("%q is %v, not a whole number", src, v)
	}
	return int32(v), err
}

// expr is src's expression, its value its last term.
func (e *picEnv) expr(src string) (jlm.Expr, error) {
	var x sizes
	a, err := e.lowered(&x, src)
	if err != nil {
		return jlm.Expr{}, err
	}
	return finished(&x, a), nil
}

func (e *picEnv) lowered(x *sizes, src string) (jlm.Arg, error) {
	n, err := parseExpr(src)
	if err != nil {
		return jlm.Arg{}, err
	}
	return e.lowerNode(x, n)
}

// finished is x with a its value: an operand, a constant or an earlier term
// is written as it plus zero.
func finished(x *sizes, a jlm.Arg) jlm.Expr {
	if a.Kind != jlm.ArgTerm || int(a.Index) != len(x.T)-1 {
		x.add(a, jlm.ExactNum(0))
	}
	return x.expr()
}

func (e *picEnv) enter() error {
	if e.depth++; e.depth > maxPicDepth {
		return fmt.Errorf("an expression nested past %d (a definition that reads itself?)", maxPicDepth)
	}
	return nil
}

// fold is n's value where it reads no operand (ok), evaluated as the host
// evaluates the expression.
func (e *picEnv) fold(n *exprNode) (float64, bool, error) {
	if err := e.enter(); err != nil {
		return 0, false, err
	}
	defer func() { e.depth-- }()
	switch n.op {
	case "num":
		return n.num, true, nil
	case "bool":
		if n.b {
			return 1, true, nil
		}
		return 0, true, nil
	case "path":
		if a, ok := e.args[n.str]; ok {
			return a.Exact, a.Kind == jlm.ArgExact, nil
		}
		if v, ok := e.consts[n.str]; ok {
			return v, true, nil
		}
		if d, ok := e.defs[n.str]; ok {
			return e.fold(d)
		}
		return 0, false, fmt.Errorf("%q is no picture operand, binding, definition or constant", n.str)
	case "?:":
		c, ok, err := e.fold(n.args[0])
		if err != nil || !ok {
			return 0, false, err
		}
		if c > 0 {
			return e.fold(n.args[1])
		}
		return e.fold(n.args[2])
	case "call":
		if n.str == "where" && len(n.args) == 3 {
			c, ok, err := e.fold(n.args[0])
			if err != nil || !ok {
				return 0, false, err
			}
			if c > 0 {
				return e.fold(n.args[1])
			}
			return e.fold(n.args[2])
		}
	}
	var v [3]float64
	if len(n.args) > len(v) {
		return 0, false, fmt.Errorf("%s of %d operands", n.op, len(n.args))
	}
	all := true
	for i, a := range n.args {
		x, ok, err := e.fold(a)
		if err != nil {
			return 0, false, err
		}
		v[i], all = x, all && ok
	}
	if !all {
		return 0, false, nil
	}
	f, swap, err := picFn(n)
	if err != nil {
		return 0, false, err
	}
	if swap {
		v[0], v[1] = v[1], v[0]
	}
	return applyFn(f, n.op, v[0], v[1], v[2]), true, nil
}

// picFn is the function n applies, its two operands swapped where swap (a >
// b is b < a), and the composite operators' leading function.
func picFn(n *exprNode) (f jlm.Fn, swap bool, err error) {
	switch n.op {
	case "+":
		return jlm.FnAdd, false, nil
	case "-", "neg", "!", "!=", "<=", ">=", "%":
		return jlm.FnSub, false, nil
	case "*", "&&":
		return jlm.FnMul, false, nil
	case "||":
		return jlm.FnAdd, false, nil
	case "/":
		return jlm.FnDiv, false, nil
	case "<":
		return jlm.FnLess, false, nil
	case ">":
		return jlm.FnLess, true, nil
	case "==":
		return jlm.FnEqual, false, nil
	case "call":
		f, ok := picFns()[n.str]
		if !ok {
			return 0, false, fmt.Errorf("%s is no function a picture's expression calls", n.str)
		}
		if f.Arity() != len(n.args) {
			return 0, false, fmt.Errorf("%s of %d operands", n.str, len(n.args))
		}
		return f, false, nil
	}
	return 0, false, fmt.Errorf("%q is no operation a picture's expression writes", n.op)
}

func picFns() map[string]jlm.Fn {
	return map[string]jlm.Fn{"min": jlm.FnMin, "max": jlm.FnMax, "where": jlm.FnWhere, "floor": jlm.FnFloor,
		"ceil": jlm.FnCeil, "trunc": jlm.FnTrunc, "round": jlm.FnRound, "round_even": jlm.FnRoundEven, "abs": jlm.FnAbs,
		"sqrt": jlm.FnSqrt, "log": jlm.FnLog, "f32": jlm.FnF32}
}

// applyFn is op over a, b and c in float64, as the host evaluates its
// terms (lowerNode writes the same terms).
func applyFn(f jlm.Fn, op string, a, b, c float64) float64 {
	is := func(x bool) float64 {
		if x {
			return 1
		}
		return 0
	}
	switch op {
	case "neg":
		return 0 - a
	case "!":
		return 1 - a
	case "!=":
		return 1 - is(a == b)
	case "<=":
		return 1 - is(b < a)
	case ">=":
		return 1 - is(a < b)
	case "%":
		return a - b*math.Floor(a/b)
	}
	switch f {
	case jlm.FnAdd:
		return a + b
	case jlm.FnSub:
		return a - b
	case jlm.FnMul:
		return a * b
	case jlm.FnDiv:
		return a / b
	case jlm.FnMin:
		return min(a, b)
	case jlm.FnMax:
		return max(a, b)
	case jlm.FnLess:
		return is(a < b)
	case jlm.FnEqual:
		return is(a == b)
	case jlm.FnWhere:
		if a > 0 {
			return b
		}
		return c
	case jlm.FnFloor:
		return math.Floor(a)
	case jlm.FnCeil:
		return math.Ceil(a)
	case jlm.FnTrunc:
		return math.Trunc(a)
	case jlm.FnRound:
		return math.Round(a)
	case jlm.FnRoundEven:
		return math.RoundToEven(a)
	case jlm.FnAbs:
		return math.Abs(a)
	case jlm.FnSqrt:
		return math.Sqrt(a)
	case jlm.FnLog:
		return math.Log(a)
	case jlm.FnF32:
		return float64(float32(a))
	}
	return math.NaN()
}

// lowerNode writes n's terms into x, its value the operand returned: a part
// that reads no operand a constant, a condition that reads none its branch.
func (e *picEnv) lowerNode(x *sizes, n *exprNode) (jlm.Arg, error) {
	v, ok, err := e.fold(n)
	if err != nil {
		return jlm.Arg{}, err
	}
	if ok {
		return jlm.ExactNum(v), nil
	}
	if err := e.enter(); err != nil {
		return jlm.Arg{}, err
	}
	defer func() { e.depth-- }()
	switch n.op {
	case "path":
		if a, ok := e.args[n.str]; ok {
			return a, nil
		}
		return e.lowerNode(x, e.defs[n.str])
	case "?:", "call":
		if n.op == "?:" || n.str == "where" && len(n.args) == 3 {
			if c, ok, err := e.fold(n.args[0]); err != nil {
				return jlm.Arg{}, err
			} else if ok && c > 0 {
				return e.lowerNode(x, n.args[1])
			} else if ok {
				return e.lowerNode(x, n.args[2])
			}
		}
	}
	var a [3]jlm.Arg
	for i, m := range n.args {
		if a[i], err = e.lowerNode(x, m); err != nil {
			return jlm.Arg{}, err
		}
	}
	if n.op == "?:" {
		return x.where(a[0], a[1], a[2]), nil
	}
	f, swap, err := picFn(n)
	if err != nil {
		return jlm.Arg{}, err
	}
	one := jlm.ExactNum(1)
	switch n.op {
	case "neg":
		return x.sub(jlm.ExactNum(0), a[0]), nil
	case "!":
		return x.sub(one, a[0]), nil
	case "!=":
		return x.sub(one, x.Fn(jlm.FnEqual, a[0], a[1])), nil
	case "<=":
		return x.sub(one, x.less(a[1], a[0])), nil
	case ">=":
		return x.sub(one, x.less(a[0], a[1])), nil
	case "%":
		return x.sub(a[0], x.mul(a[1], x.fn(jlm.FnFloor, x.div(a[0], a[1])))), nil
	}
	if swap {
		a[0], a[1] = a[1], a[0]
	}
	return x.Fn(f, a[:f.Arity()]...), nil
}
