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
	// of the file's block count.
	Blocks string `json:"blocks"`
	// HardTokens is the first id of the token table the projector embeds
	// (hard_token_embeddings): the ids from it on are the tower's own markers.
	HardTokens float32 `json:"hard_tokens"`
	PatchSize  string  `json:"patch_size"`
	Scale      string  `json:"scale"`
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
	// Transforms are the tensor transforms (towerTransforms) the tower's
	// tensors take at each phase of conversion (towerPhases).
	Transforms map[string][]string `json:"transforms"`
	exprs      map[string]*jinja.Expr
}

// towerPhases is where a tower's transforms run: joined, once the text
// model's table is beside the tower; source, once the tower's tensors are
// read; prepare, before the blocks' FFNs are padded; finish, after the
// patch and merge kernels are reshaped, before quantization.
func towerPhases() []string { return []string{"joined", "source", "prepare", "finish"} }

// towerTransforms is every tensor transform a projector's entry may name.
// f is the tower's file, nil outside the source phase.
func towerTransforms() map[string]func(s *jlm.Source, f *meta.File, p *projStates) error {
	return map[string]func(*jlm.Source, *meta.File, *projStates) error{
		"image_break_row":      func(s *jlm.Source, _ *meta.File, _ *projStates) error { return imgBreakRow(s) },
		"transpose_projection": func(s *jlm.Source, _ *meta.File, _ *projStates) error { return transposeProjection(s) },
		"permute_merger":       func(s *jlm.Source, _ *meta.File, _ *projStates) error { return permuteMerger(s) },
		"half_neox_to_neox":    func(s *jlm.Source, _ *meta.File, _ *projStates) error { return halfNeoxToNeox(s) },
		"negate_std_bias": func(s *jlm.Source, _ *meta.File, _ *projStates) error {
			return negateRole(s, jlm.RoleVStdBias)
		},
		"point_conv_projection": func(s *jlm.Source, _ *meta.File, _ *projStates) error {
			for i := range s.Tensors {
				if s.Tensors[i].Role == jlm.RoleVProj {
					if err := reshapePointConv(&s.Tensors[i]); err != nil {
						return err
					}
				}
			}
			return nil
		},
		"norm_frame_rows": func(s *jlm.Source, _ *meta.File, _ *projStates) error { return normFrameRows(s, s.Vision.Eps) },
		"stage_places": func(s *jlm.Source, f *meta.File, _ *projStates) error {
			if f == nil {
				return fmt.Errorf("convert: stage_places reads the tower's file, which only the source phase has")
			}
			stagePlaces(s, stagedBlocks(f))
			return nil
		},
		"hard_token_embeddings": func(s *jlm.Source, _ *meta.File, p *projStates) error {
			return hardTokenEmbeddings(s, p.HardTokens)
		},
		"conv_layouts":        func(s *jlm.Source, _ *meta.File, _ *projStates) error { return convLayouts(s) },
		"fold_channel_scales": func(s *jlm.Source, _ *meta.File, _ *projStates) error { return foldChannelScales(s) },
	}
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

// transformTower runs the transforms p names for phase; p is nil for a
// source with no tower.
func transformTower(s *jlm.Source, f *meta.File, p *projStates, phase string) error {
	if p == nil {
		return nil
	}
	ts := towerTransforms()
	for _, t := range p.Transforms[phase] {
		if err := ts[t](s, f, p); err != nil {
			return err
		}
	}
	return nil
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
	Audio      []string               `json:"audio"`
	Projectors map[string]*projStates `json:"projectors"`
}

// projectorLibrary is projectors.json's vision projectors, each entry checked and its
// expressions compiled, an "as" entry resolved to the one it names.
func projectorLibrary() (map[string]*projStates, error) {
	var lib projectorFile
	if err := json.Unmarshal(projectorsJSON, &lib); err != nil {
		return nil, fmt.Errorf("convert: projectors.json: %w", err)
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
		if err := p.check(); err != nil {
			return nil, fmt.Errorf("convert: projectors.json: %s: %w", name, err)
		}
	}
	return lib.Projectors, nil
}

func (p *projStates) check() error {
	if p.Code == 0 || p.Output.Tensor == "" || p.Output.Dim < 0 || p.Output.Dim > 1 {
		return fmt.Errorf("an entry states its code and its output matrix's tensor and dim (0 or 1)")
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
	ts := towerTransforms()
	for _, phase := range slices.Sorted(maps.Keys(p.Transforms)) {
		if !slices.Contains(towerPhases(), phase) {
			return fmt.Errorf("transforms at %q, which is no phase of conversion", phase)
		}
		for _, t := range p.Transforms[phase] {
			if _, ok := ts[t]; !ok {
				return fmt.Errorf("transform %q is no tower transform", t)
			}
		}
	}
	embeds := false
	for _, phase := range p.Transforms {
		embeds = embeds || slices.Contains(phase, "hard_token_embeddings")
	}
	if embeds != (p.HardTokens > 0) {
		return fmt.Errorf("hard_tokens %g and the hard_token_embeddings transform go together", p.HardTokens)
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
	return lib.Audio
}
