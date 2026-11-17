package convert

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

//go:embed projectors.json
var projectorsJSON []byte

// projStates is what projectors.json states for one clip.projector_type: the
// container's projector code, the matrix whose rows are the projector's output
// width, and the tower's geometry and features, each number an expression
// over the file's keys (src), its tensors' shapes (shape) and, for the
// geometry, the tower's patch, picture and merge (tower).
type projStates struct {
	// Describe is docs/models.md's line for the projector: the models it
	// covers.
	Describe string `json:"describe"`
	// As names another entry whose statements this one takes.
	As   string `json:"as"`
	Code uint16 `json:"code"`
	// When is the condition the entry holds under, and States what it
	// covers; a file outside it is refused.
	When   string     `json:"when"`
	States string     `json:"states"`
	Output projOutput `json:"output"`
	// Flags are tower features by name (projFlags).
	Flags []string `json:"flags"`
	// Requires is the tensors the tower reads that a file must carry.
	Requires []string `json:"requires"`
	// Taps is whether the projector reads taps off the tower's blocks.
	Taps bool `json:"taps"`
	// Blocks "staged" names the file's blocks by a stage and a place in it
	// (stagedBlocks), one container block each in that order, counted in place
	// of the file's block count, each carrying its place (stagePlaces).
	Blocks    string `json:"blocks"`
	PatchSize string `json:"patch_size"`
	Scale     string `json:"scale"`
	// MergeOf is the matrix that takes Scale*Scale patches at once, so its
	// k over the tower's width states the merge.
	MergeOf string           `json:"merge_of"`
	Tiles   *projStatesRange `json:"tiles"`
	Pixels  *projStatesRange `json:"pixels"`
	Tile    *struct {
		Size      string `json:"size"`
		Tolerance string `json:"tolerance"`
	} `json:"tile"`
	Window *struct {
		Size    string `json:"size"`
		Pattern string `json:"pattern"`
	} `json:"window"`
	// Resize is the resampling filter by name (projResizes).
	Resize string `json:"resize"`
	// Mean and Std replace the file's normalisation, and Activation its
	// activation ("gelu").
	Mean       *[3]float32 `json:"mean"`
	Std        *[3]float32 `json:"std"`
	Activation string      `json:"activation"`
	// TextRows are the tower's rows that are text tokens' embeddings
	// (textRows).
	TextRows []projTextRow `json:"text_rows"`
	// Tower is the tower as its reference runs it (mjTower).
	Tower *mjTower `json:"tower"`
	// Tensors is the tower's tensor table over its file's names: model
	// entries, block entries by the name past v.blk.N. (as v.<name>), sizes,
	// and gathers, each run for every block holding its first source (named
	// past v.blk.N.), or the model's.
	Tensors *mjTensors `json:"tensors"`
	exprs   map[string]*jinja.Expr
}

// projTextRow is a tower tensor that is a text token's embedding: Token's
// row of the text table, stored as Role.
type projTextRow struct {
	Role  string `json:"role"`
	Token string `json:"token"`
}

// textRows gives a tower each text row p states that its file does not
// carry: the text table's own row of the token, the row the reference looks
// up for it, at the precision the text file keeps it. An mmproj carries such a row
// only where its writer could see the text table, and the container holds the
// text model too.
func textRows(s *jlm.Source, p *projStates) error {
	if p == nil {
		return nil
	}
	roles := rolesByName()
	for _, tr := range p.TextRows {
		r := roles[tr.Role]
		if slices.ContainsFunc(s.Tensors, func(t jlm.Tensor) bool { return t.Role == r }) {
			continue
		}
		id := -1
		if s.Vocab != nil {
			id = slices.Index(s.Vocab.Tokens, tr.Token)
		}
		if id < 0 {
			return fmt.Errorf("convert: the tower carries no %s and the text vocabulary has no %q", tr.Role, tr.Token)
		}
		row, err := tokenRow(s, id)
		if err != nil {
			return err
		}
		e := jlm.Tensor{Role: r, Block: jlm.DenseBlock, Index: -1, Name: r.String()}
		setF32Dims(&e, row, uint64(len(row)))
		s.Tensors = append(s.Tensors, e)
	}
	return nil
}

// statement is the tower's tensor table as the GGUF table's runner reads it:
// model and block entries, sizes and gathers, which is all a tower's table
// states.
func (p *projStates) statement() (*Statement, error) {
	st := &Statement{}
	t := p.Tensors
	if t == nil {
		return st, nil
	}
	if len(t.BlockLinear)+len(t.Fuse)+len(t.Joined)+len(t.Drop)+len(t.Keys) > 0 ||
		t.BlockPrefix != "" || t.Ignore != nil || t.IgnorePrefix != nil || t.Experts != nil || t.Ternary != nil || t.Present != nil {
		return nil, fmt.Errorf("tensors: a tower's table states its model and block tensors, sizes and gathers")
	}
	names, err := readTensorNames()
	if err != nil {
		return nil, err
	}
	st.modelTensors, st.blockTensors = map[string][]statedCut{}, map[string][]statedCut{}
	for _, x := range []struct {
		table map[string]mjTargets
		to    map[string][]statedCut
	}{{t.Model, st.modelTensors}, {t.Block, st.blockTensors}} {
		for _, n := range slices.Sorted(maps.Keys(x.table)) {
			if x.to[n], err = statedCuts(names, n, x.table[n]); err != nil {
				return nil, fmt.Errorf("tensors: %w", err)
			}
		}
	}
	if st.sizes, err = statedSizes(t.Sizes); err != nil {
		return nil, fmt.Errorf("tensors: %w", err)
	}
	st.sources = map[string]bool{}
	for _, g := range t.Gather {
		r, ok := names.roleOf(g.Role)
		if !ok {
			r, ok = rolesByName()[g.Role]
		}
		if !ok || g.Every || g.Count != "" || len(g.From) == 0 {
			return nil, fmt.Errorf("tensors: the gather of %s: a tower's gather is of a role the container knows, "+
				"from sources, run where its first is", g.Role)
		}
		if err := g.Do.check(); err != nil {
			return nil, fmt.Errorf("tensors: the gather of %s: %w", g.Role, err)
		}
		step := hfGatherStep{role: r, from: g.From, do: g.Do, model: g.Model, name: g.Name}
		if step.f64, err = g.wide(); err != nil {
			return nil, fmt.Errorf("tensors: %w", err)
		}
		if g.If != "" {
			if step.cond, err = parseExpr(g.If); err != nil {
				return nil, fmt.Errorf("tensors: the gather of %s: %w", g.Role, err)
			}
		}
		for _, o := range g.From {
			if err := o.Do.check(); err != nil {
				return nil, fmt.Errorf("tensors: the gather of %s: %w", g.Role, err)
			}
			for _, n := range o.names() {
				st.sources[n] = true
			}
		}
		st.gathers = append(st.gathers, step)
	}
	return st, nil
}

// projectorFor is what the library states for the container's projector
// code, nil for a code no entry states.
func projectorFor(code jlm.Projector) (*projStates, error) {
	lib, err := projectorLibrary()
	if err != nil {
		return nil, err
	}
	for _, name := range slices.Sorted(maps.Keys(lib)) {
		if p := lib[name]; jlm.Projector(p.Code) == code {
			return p, nil
		}
	}
	return nil, nil
}

// projOutput is the matrix whose rows are the projector's output width, and
// the dimension they are.
type projOutput struct {
	Tensor string `json:"tensor"`
	Dim    int    `json:"dim"`
}

type projStatesRange struct {
	Min string `json:"min"`
	Max string `json:"max"`
}

func projFlags() map[string]jlm.Flags {
	return map[string]jlm.Flags{"vision_rope": jlm.FlagVisionRope, "gelu": jlm.FlagGELU}
}

func projResizes() map[string]jlm.Flags {
	return map[string]jlm.Flags{"": 0, "bilinear": 0, "bicubic": jlm.FlagVisionBicubic}
}

// projectorFile is projectors.json: the vision projectors by
// clip.projector_type, and the audio towers (clip.audio.projector_type) that
// convert.
type projectorFile struct {
	// Describe is docs/models.md's line for the vision tower's own name
	// (towerArch).
	Describe   string                  `json:"describe"`
	Audio      map[string]*audioStates `json:"audio"`
	Projectors map[string]*projStates  `json:"projectors"`
}

// projectorLibrary is projectors.json's vision projectors, each entry checked and its
// expressions compiled, an "as" entry resolved to the one it names.
func projectorLibrary() (map[string]*projStates, error) {
	var lib projectorFile
	if err := json.Unmarshal(projectorsJSON, &lib); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: %w", err)
	}
	kinds, err := towerSteps()
	if err != nil {
		return nil, err
	}
	for _, name := range slices.Sorted(maps.Keys(lib.Projectors)) {
		p := lib.Projectors[name]
		if p.As != "" {
			to, ok := lib.Projectors[p.As]
			if !ok || to.As != "" || p.Code != 0 || p.Output.Tensor != "" {
				return nil, fmt.Errorf("convert: projectors.json: %s is as %q, which is no entry of its own, and states nothing beside it", name, p.As)
			}
			lib.Projectors[name] = to
			continue
		}
		if err := p.check(kinds); err != nil {
			return nil, fmt.Errorf("convert: projectors.json: %s: %w", name, err)
		}
	}
	return lib.Projectors, nil
}

// check refuses an entry that states less than it must, or a tower whose
// steps are none of kinds (towerSteps).
func (p *projStates) check(kinds map[string]jlm.PreKind) error {
	if p.Code == 0 || p.Output.Tensor == "" || p.Output.Dim < 0 || p.Output.Dim > 1 {
		return fmt.Errorf("an entry states its code and its output matrix's tensor and dim (0 or 1)")
	}
	if t := p.Tower; t != nil {
		if err := t.check(kinds); err != nil {
			return fmt.Errorf("tower: %w", err)
		}
	}
	if _, err := p.statement(); err != nil {
		return err
	}
	for _, f := range p.Flags {
		if _, ok := projFlags()[f]; !ok {
			return fmt.Errorf("flag %q is no tower feature", f)
		}
	}
	if p.Blocks != "" && p.Blocks != "staged" {
		return fmt.Errorf("blocks %q: only \"staged\" is read", p.Blocks)
	}
	if p.Activation != "" && p.Activation != "gelu" {
		return fmt.Errorf("activation %q: only \"gelu\" is stated", p.Activation)
	}
	for _, tr := range p.TextRows {
		if r, ok := rolesByName()[tr.Role]; !ok || !r.Vision() || tr.Token == "" {
			return fmt.Errorf("text row %q of %q: a tower role and a token", tr.Role, tr.Token)
		}
	}
	if (p.When == "") != (p.States == "") {
		return fmt.Errorf("a condition states what it covers")
	}
	srcs := []string{p.When, p.PatchSize, p.Scale, p.Resize}
	for _, r := range []*projStatesRange{p.Tiles, p.Pixels} {
		if r != nil {
			srcs = append(srcs, r.Min, r.Max)
		}
	}
	if p.Tile != nil {
		srcs = append(srcs, p.Tile.Size, p.Tile.Tolerance)
	}
	if p.Window != nil {
		srcs = append(srcs, p.Window.Size, p.Window.Pattern)
	}
	p.exprs = map[string]*jinja.Expr{}
	for _, src := range srcs {
		if src == "" || p.exprs[src] != nil {
			continue
		}
		x, err := jinja.CompileExpr(src)
		if err != nil {
			return err
		}
		for _, n := range x.Names() {
			if n != "src" && n != "shape" && n != "tower" {
				return fmt.Errorf("%q reads %q, which is not the keys (src), the shapes (shape) or the tower", src, n)
			}
		}
		p.exprs[src] = x
	}
	return nil
}

// shapesOf is f's tensors' dims by name.
func shapesOf(f *meta.File) jinja.Value {
	d := jinja.NewDict()
	for i := range f.Tensors {
		t := &f.Tensors[i]
		dims := make([]jinja.Value, len(t.Dims))
		for j, n := range t.Dims {
			dims[j] = jinja.NewInt(int64(n))
		}
		d.AsDict().Set(t.Name, jinja.NewList(dims))
	}
	return d
}

// state sets what p states for the projector named name, read off f, in v.
func (p *projStates) state(name string, f *meta.File, v *jlm.Vision) error {
	vars := map[string]jinja.Value{"src": ggufKeys(f), "shape": shapesOf(f)}
	value := func(what, src string) (jinja.Value, error) {
		x, err := p.exprs[src].Eval(vars, blockExprSteps)
		if err == nil && x.IsUndefined() {
			err = fmt.Errorf("%q is undefined", src)
		}
		if err != nil {
			return jinja.Undefined(), fmt.Errorf("convert: clip: %s: %s: %w", name, what, err)
		}
		return x, nil
	}
	number := func(what, src string) (float64, error) {
		x, err := value(what, src)
		switch {
		case err != nil:
			return 0, err
		case x.IsInt():
			return float64(x.AsInt()), nil
		case x.IsFloat():
			return x.AsFloat(), nil
		}
		return 0, fmt.Errorf("convert: clip: %s: %s is %v, not a number", name, what, x)
	}
	whole := func(what, src string, to *uint32) error {
		x, err := number(what, src)
		if err == nil && (x < 0 || x > math.MaxUint32 || x != math.Trunc(x)) {
			err = fmt.Errorf("convert: clip: %s: %s is %g, not a count", name, what, x)
		}
		*to = uint32(x)
		return err
	}
	if p.When != "" {
		x, err := value("when", p.When)
		if err != nil {
			return err
		}
		if !x.IsBool() || !x.AsBool() {
			return fmt.Errorf("convert: clip: %s: only %s is implemented: %w", name, p.States, ErrNotImplemented)
		}
	}
	for _, t := range p.Requires {
		if _, ok := f.Get(t); !ok {
			return fmt.Errorf("convert: clip: %s with no %s", name, t)
		}
	}
	v.Projector = jlm.Projector(p.Code)
	for _, fl := range p.Flags {
		v.Flags |= projFlags()[fl]
	}
	if p.Blocks == "staged" {
		v.NLayer = uint32(len(stagedBlocks(f)))
	}
	if p.PatchSize != "" {
		if err := whole("patch_size", p.PatchSize, &v.PatchSize); err != nil {
			return err
		}
	}
	if p.Scale != "" {
		if err := whole("scale", p.Scale, &v.Scale); err != nil {
			return err
		}
	}
	vars["tower"] = jinja.NewDict()
	for k, n := range map[string]uint32{"patch_size": v.PatchSize, "image_size": v.ImageSize, "scale": v.Scale} {
		vars["tower"].AsDict().Set(k, jinja.NewInt(int64(n)))
	}
	var err error
	if r := p.Tiles; r != nil {
		if err = whole("tiles.min", r.Min, &v.MinTiles); err == nil {
			err = whole("tiles.max", r.Max, &v.MaxTiles)
		}
		if err == nil && (v.MinTiles == 0 || v.MinTiles > v.MaxTiles) {
			err = fmt.Errorf("convert: clip: %s tiles %d..%d", name, v.MinTiles, v.MaxTiles)
		}
		if err != nil {
			return err
		}
	}
	if r := p.Pixels; r != nil {
		if err = whole("pixels.min", r.Min, &v.MinPixels); err == nil {
			err = whole("pixels.max", r.Max, &v.MaxPixels)
		}
		if err == nil && (v.MinPixels == 0 || v.MaxPixels < v.MinPixels) {
			err = fmt.Errorf("convert: clip: %s states image_min_pixels %d and "+
				"image_max_pixels %d; a picture read at its own size needs both", name, v.MinPixels, v.MaxPixels)
		}
		if err != nil {
			return err
		}
	}
	if t := p.Tile; t != nil {
		if v.PatchSize == 0 || v.ImageSize%v.PatchSize != 0 {
			return fmt.Errorf("convert: clip: %s tiles %d-pixel patches of a %d table", name, v.PatchSize, v.ImageSize)
		}
		if err := whole("tile.size", t.Size, &v.TileSize); err != nil {
			return err
		}
		tol, err := number("tile.tolerance", t.Tolerance)
		if err != nil {
			return err
		}
		v.Tolerance = float32(tol)
	}
	if w := p.Window; w != nil {
		if err = whole("window.pattern", w.Pattern, &v.WinPattern); err == nil {
			err = whole("window.size", w.Size, &v.WinSize)
		}
		if err == nil && v.WinPattern == 0 {
			err = fmt.Errorf("convert: clip: %s with no window pattern, so which blocks attend inside windows is unknown", name)
		}
		if err != nil {
			return err
		}
	}
	if p.Resize != "" {
		x, err := value("resize", p.Resize)
		if err != nil {
			return err
		}
		fl, ok := jlm.Flags(0), false
		if x.IsString() {
			fl, ok = projResizes()[x.AsString()]
		}
		if !ok {
			return fmt.Errorf("convert: clip: %s resamples %v, which is not implemented: %w", name, x, ErrNotImplemented)
		}
		v.Flags |= fl
	}
	// projection_dim is not the projector's output width: llava-phi-3 carries
	// CLIP's contrastive projection dim (768) there while mm.2 emits 3072. The
	// output matrix's rows are the fact.
	lt, ok := f.Get(p.Output.Tensor)
	if !ok || len(lt.Dims) < 2 {
		return fmt.Errorf("convert: clip: %s is missing or not a matrix, so the "+
			"projector's output width cannot be derived", p.Output.Tensor)
	}
	v.ProjDim = uint32(lt.Dims[p.Output.Dim])
	if p.MergeOf != "" {
		mm, ok := f.Get(p.MergeOf)
		if !ok || len(mm.Dims) < 2 {
			return fmt.Errorf("convert: clip: %s has no %s matrix", name, p.MergeOf)
		}
		k := uint32(mm.Dims[0])
		if k == 0 || v.NEmbd == 0 || k%v.NEmbd != 0 {
			return fmt.Errorf("convert: clip: %s k=%d is not a multiple of NEmbd=%d", p.MergeOf, k, v.NEmbd)
		}
		sq := k / v.NEmbd
		v.Scale = uint32(math.Sqrt(float64(sq)))
		if v.Scale*v.Scale != sq {
			return fmt.Errorf("convert: clip: the merger takes %d patches at once, "+
				"which is not a square grouping", sq)
		}
	}
	if v.Scale == 0 {
		return fmt.Errorf("convert: clip: %s states a merge of 0", name)
	}
	return nil
}

// restate applies what p replaces of the file's own statements, after they
// are read.
func (p *projStates) restate(v *jlm.Vision) {
	if p.Mean != nil {
		v.Mean = *p.Mean
	}
	if p.Std != nil {
		v.Std = *p.Std
	}
	if p.Activation == "gelu" {
		v.Flags = v.Flags&^jlm.FlagQuickGELU | jlm.FlagGELU
	}
}

// Projectors is every clip.projector_type projectors.json states, sorted;
// none where the library does not read (TestTheProjectorLibraryReads).
func Projectors() []string {
	lib, err := projectorLibrary()
	if err != nil {
		return nil
	}
	return slices.Sorted(maps.Keys(lib))
}

// AudioProjectors is the audio towers projectors.json lists, as Projectors
// is the vision towers'.
func AudioProjectors() []string {
	var lib projectorFile
	if err := json.Unmarshal(projectorsJSON, &lib); err != nil {
		return nil
	}
	return slices.Sorted(maps.Keys(lib.Audio))
}

// audioStates is an audio tower as the projector library states it: its
// projector's code, its geometry (each field an expression over the
// mmproj's keys, src), what the geometry must satisfy, its tensors' names
// and layouts, what the geometry must satisfy once the tensors have stated
// their widths, and its tower as its reference runs it.
type audioStates struct {
	Describe    string            `json:"describe"`
	Code        uint16            `json:"code"`
	Geometry    map[string]string `json:"geometry"`
	Require     []mjRequire       `json:"require"`
	Names       audioNames        `json:"names"`
	Layout      audioLayout       `json:"layout"`
	RequireRead []mjRequire       `json:"require_read"`
	Tower       *mjTower          `json:"tower"`
}

// audioNames is an audio tower's tensor vocabulary, each name to its
// container role by name: a block's after BlockPrefix and its index, an
// indexed layer's after IndexedPrefix (its index, then weight or bias), the
// dense ones (under DensePrefixes) whole.
type audioNames struct {
	BlockPrefix   string            `json:"block_prefix"`
	Block         map[string]string `json:"block"`
	IndexedPrefix string            `json:"indexed_prefix"`
	Indexed       map[string]string `json:"indexed"`
	DensePrefixes []string          `json:"dense_prefixes"`
	Dense         map[string]string `json:"dense"`
}

// audioRoles is an audio vocabulary's roles resolved.
type audioRoles struct {
	Block, Indexed, Dense map[string]jlm.Role
}

// roles is st's names with their roles resolved, refusing a role the
// container does not name.
func (st *audioStates) roles(name string) (audioRoles, error) {
	named := rolesByName()
	out := audioRoles{Block: map[string]jlm.Role{}, Indexed: map[string]jlm.Role{}, Dense: map[string]jlm.Role{}}
	for _, t := range []struct {
		from map[string]string
		to   map[string]jlm.Role
	}{{st.Names.Block, out.Block}, {st.Names.Indexed, out.Indexed}, {st.Names.Dense, out.Dense}} {
		for n, r := range t.from {
			role, ok := named[r]
			if !ok {
				return out, fmt.Errorf("convert: projectors.json: audio %s: %q is role %q, which the container does not know", name, n, r)
			}
			t.to[n] = role
		}
	}
	return out, nil
}

// audioFields is where each geometry field of an audio statement lands.
func audioFields(a *jlm.Audio) map[string]*uint32 {
	return map[string]*uint32{"layers": &a.NLayer, "embd": &a.NEmbd, "heads": &a.NHead, "ffn": &a.NFFN,
		"mel_bins": &a.MelBins, "proj_dim": &a.ProjDim, "sample_rate": &a.SampleRate, "n_fft": &a.NFFT,
		"window": &a.Window, "hop": &a.Hop, "max_samples": &a.MaxSamples, "conv_kernel": &a.ConvKernel,
		"sub_channels": &a.SubChannels, "res_width": &a.ResWidth}
}

// libraryEval is expr of the library statement by over vars, refusing a
// name vars does not hold.
func libraryEval(by, what, expr string, vars map[string]jinja.Value) (jinja.Value, error) {
	x, err := jinja.CompileExpr(expr)
	if err == nil {
		for _, n := range x.Names() {
			if _, ok := vars[n]; !ok {
				err = fmt.Errorf("%q reads %q, which is none of %v", expr, n, slices.Sorted(maps.Keys(vars)))
				break
			}
		}
	}
	var v jinja.Value
	if err == nil {
		v, err = x.Eval(vars, blockExprSteps)
	}
	if err == nil && v.IsUndefined() {
		err = fmt.Errorf("%q is undefined", expr)
	}
	if err != nil {
		return jinja.Undefined(), fmt.Errorf("convert: %s: %s: %w", by, what, err)
	}
	return v, nil
}

// geometry is the audio geometry st states for an mmproj of keys src, held
// to st.Require.
func (st *audioStates) geometry(name string, src jinja.Value) (*jlm.Audio, error) {
	a := &jlm.Audio{Projector: jlm.Projector(st.Code)}
	fields := audioFields(a)
	vars := map[string]jinja.Value{"src": src}
	for _, k := range slices.Sorted(maps.Keys(st.Geometry)) {
		v, err := libraryEval("projectors.json: audio "+name, k, st.Geometry[k], vars)
		if err != nil {
			return nil, err
		}
		var f float64
		switch {
		case v.IsInt():
			f = float64(v.AsInt())
		case v.IsFloat():
			f = v.AsFloat()
		default:
			return nil, fmt.Errorf("convert: projectors.json: audio %s: %s is not a number", name, k)
		}
		if k == "eps" {
			a.Eps = float32(f)
			continue
		}
		to, ok := fields[k]
		if !ok {
			return nil, fmt.Errorf("convert: projectors.json: audio %s: %q is no audio geometry field", name, k)
		}
		*to = uint32(f)
	}
	if err := st.holds(name, src, a, st.Require); err != nil {
		return nil, err
	}
	return a, nil
}

// holds is nil when geometry a of an mmproj of keys src satisfies every one
// of reqs.
func (st *audioStates) holds(name string, src jinja.Value, a *jlm.Audio, reqs []mjRequire) error {
	geom := jinja.NewDict()
	for k, x := range audioFields(a) {
		geom.AsDict().Set(k, jinja.NewInt(int64(*x)))
	}
	geom.AsDict().Set("eps", jinja.NewFloat(float64(a.Eps)))
	vars := map[string]jinja.Value{"src": src, "a": geom}
	for _, r := range reqs {
		v, err := libraryEval("projectors.json: audio "+name, "require", r.That, vars)
		if err != nil {
			return err
		}
		if !v.IsTruthy() {
			return fmt.Errorf("convert: clip: the audio encoder %s", r.Else)
		}
	}
	return nil
}

// libraryAudio is what the projector library states for the audio tower of
// clip.audio.projector_type name.
func libraryAudio(name string) (*audioStates, error) {
	var lib projectorFile
	if err := json.Unmarshal(projectorsJSON, &lib); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: %w", err)
	}
	a := lib.Audio[name]
	if a == nil || a.Tower == nil || a.Code == 0 {
		return nil, fmt.Errorf("convert: projectors.json states no audio tower %q, or no code for it", name)
	}
	kinds, err := towerSteps()
	if err != nil {
		return nil, err
	}
	if err := a.Tower.check(kinds); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: audio %s: %w", name, err)
	}
	return a, nil
}
