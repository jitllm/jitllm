package convert

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/gather"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// mjTower is a projector's tower as the projector library states it (an
// entry's tower, convert/projectors.json): the tensors it must hold, what its facts must
// satisfy, the steps a picture takes, the entry's and the head's programs,
// the block's attributes and the picture's span in the prompt. Every number
// is an expression over the tower's facts (towerFacts).
type mjTower struct {
	Requires []string     `json:"requires"`
	Require  []mjRequire  `json:"require"`
	Pre      []mjPreStep  `json:"pre"`
	Entry    []mjTowerOp  `json:"entry"`
	Block    mjTowerBlock `json:"block"`
	Head     []mjTowerOp  `json:"head"`
	Span     mjTowerSpan  `json:"span"`
	Tables   []mjTable    `json:"tables"`
	Marks    *mjMarks     `json:"marks"`
	Video    *mjVideo     `json:"video"`
	Taps     *mjTaps      `json:"taps"`
	exprs    map[string]*jinja.Expr
}

// mjTaps is a tower's taps (jlm.TowerTap): one program for each index the
// tensors of role Of are stored at, the tapped block's, its expressions
// reading tap (the block, and dims, the tensor's shape there).
type mjTaps struct {
	Of   string      `json:"of"`
	Prog []mjTowerOp `json:"prog"`
}

// mjEach is the places i below Count where If holds.
type mjEach struct {
	Count string `json:"count"`
	If    string `json:"if"`
}

// mjTable is a constant table the converter writes for the tower, by its
// kind: sincos_axes is a resampler's 2-D sin/cos key positions, Width wide
// and Length rows an axis (sincosAxes).
type mjTable struct {
	Kind   string `json:"kind"`
	Width  string `json:"width"`
	Length string `json:"length"`
}

// mjMarks is a picture's marker rows (jlm.RoleVTileMarks): the text table's
// rows of the tokens Names names (markTable), or of the token ids IDs'
// expressions give, each an id or a list of them (markRows); then Grid's --
// a side x side grid whose cell r, c (from 0) is the token Name names where
// If holds and a zero row where not, or the ids of Each's expressions --
// then the tokens After names.
type mjMarks struct {
	Names []string    `json:"names"`
	IDs   []string    `json:"ids"`
	Grid  *mjMarkGrid `json:"grid"`
	After []string    `json:"after"`
}

type mjMarkGrid struct {
	Side string   `json:"side"`
	Name string   `json:"name"`
	Each []string `json:"each"`
	If   string   `json:"if"`
}

// mjPreStep is a step a picture takes, by its kind's name (towerSteps), its
// integers and its floats.
type mjPreStep struct {
	Kind   string   `json:"kind"`
	Ints   []string `json:"ints"`
	Floats []string `json:"floats"`
}

// mjTowerOp is one op of a tower program, by its name (towerOps): In its
// operands by the values earlier ops name with Out (the program's input is
// "in"), W its tensors by container role name
// (one ending in ? bound where the tower holds it; an op of no In reads the
// op before, and one of "in": [] reads nothing), If the condition the op
// is the program's under, Fn a stated op's function by name (towerFns), and
// its numbers.
type mjTowerOp struct {
	Op     string  `json:"op"`
	If     string  `json:"if"`
	In     mjNames `json:"in"`
	W      mjNames `json:"w"`
	Out    string  `json:"out"`
	Fn     string  `json:"fn"`
	Span   string  `json:"span"`
	Offset string  `json:"offset"`
	Width  string  `json:"width"`
	Eps    string  `json:"eps"`
	// Chain and Taps are a position resample's, by name (towerChains,
	// towerTaps).
	Chain string `json:"chain"`
	Taps  string `json:"taps"`
	// Const is the constant a stated op's function reads (scale).
	Const string `json:"const"`
}

// mjTowerBlock is the tower block's attributes: its heads, their width, its
// norm by name (layer or rms) and epsilon, and its MLP's activation: the
// file's (Vision.Activation), gated where Gated, or where Act is gelu_erf
// the erf form of the GELU the file states, which a class whose hidden_act
// is "gelu" runs and the file cannot say; or Fn, a function by name
// (towerFns) whatever the file says.
type mjTowerBlock struct {
	Heads   string       `json:"heads"`
	KVHeads string       `json:"kv_heads"`
	HeadDim string       `json:"head_dim"`
	Norm    string       `json:"norm"`
	Eps     string       `json:"eps"`
	Gated   bool         `json:"gated"`
	Act     string       `json:"act"`
	Fn      string       `json:"fn"`
	Rope    *mjTowerRope `json:"rope"`
	// AttnScale is the score scale where it is not 1/sqrt(head_dim), and
	// VNorm a weightless RMS norm over each head of v.
	AttnScale string `json:"attn_scale"`
	VNorm     bool   `json:"v_norm"`
	// Window is the side in patches of the windows the blocks attend in,
	// and Full the blocks that attend the whole picture.
	Window string  `json:"window"`
	Full   *mjEach `json:"full"`
}

// mjTowerRope is the block's rotary over the picture's grid
// (jlm.TowerRope): its base, width, NEOX pairs and origin, and its runs in
// groups: for each place j of a group's Count, its Each runs in order, their
// pairs, axis and frequency expressions of j.
type mjTowerRope struct {
	Base   string       `json:"base"`
	Dim    string       `json:"dim"`
	Neox   bool         `json:"neox"`
	Origin string       `json:"origin"`
	Runs   []mjRopeRuns `json:"runs"`
}

type mjRopeRuns struct {
	Count string      `json:"count"`
	Each  []mjRopeRun `json:"each"`
}

type mjRopeRun struct {
	Pairs string `json:"pairs"`
	Axis  string `json:"axis"`
	Freq  string `json:"freq"`
}

// mjTowerSpan is where a picture's rows sit in the prompt (jlm.TowerSpan):
// the text around them, or the marker tokens by their pieces (before_marks,
// after_marks), the text around a picture's number, its mark, whether its
// rows attend both ways (or where an expression over the text model's facts
// holds, bidir_if), and the grid divisor of an M-RoPE grid.
type mjTowerSpan struct {
	Before      string   `json:"before"`
	After       string   `json:"after"`
	BeforeMarks []string `json:"before_marks"`
	AfterMarks  []string `json:"after_marks"`
	IndexBefore string   `json:"index_before"`
	IndexAfter  string   `json:"index_after"`
	Mark        string   `json:"mark"`
	Bidir       bool     `json:"bidir"`
	BidirIf     string   `json:"bidir_if"`
	GridDiv     string   `json:"grid_div"`
}

func towerOps() map[string]jlm.OpCode {
	return map[string]jlm.OpCode{
		"patchify": jlm.TOpPatchify, "linear": jlm.TOpLinear, "pos_table": jlm.TOpPosTable, "add": jlm.TOpAdd,
		"layer_norm": jlm.TOpLayerNorm, "rms_norm": jlm.TOpRMSNorm, "regroup": jlm.TOpRegroup, "lead_row": jlm.TOpLeadRow,
		"drop_rows": jlm.TOpDropRows, "stated": jlm.TOpStated, "pool": jlm.TOpPool, "fold": jlm.TOpFold,
		"scale": jlm.TOpScale, "sum": jlm.TOpSum, "max": jlm.TOpMax, "concat": jlm.TOpConcat, "slice": jlm.TOpSlice,
		"mark_row": jlm.TOpMarkRow, "tail_row": jlm.TOpTailRow, "append_row": jlm.TOpAppendRow, "sub": jlm.TOpSub,
		"pos_resample": jlm.TOpPosResample, "rope": jlm.TOpRope, "ngram_rows": jlm.TOpNGramRows, "tail_conv": jlm.TOpTailConv,
		"shift": jlm.TOpShift, "row_break": jlm.TOpRowBreak, "wrap_rows": jlm.TOpWrapRows, "pos_axes": jlm.TOpPosAxes,
		"pos_buckets": jlm.TOpPosBuckets, "table_rows": jlm.TOpTableRows, "rows_attend": jlm.TOpRowsAttend,
		"unfold": jlm.TOpUnfold, "conv": jlm.TOpConv, "upsample": jlm.TOpUpsample, "frames": jlm.TOpFrames,
		"feature_norm": jlm.TOpFeatureNorm, "split_rows": jlm.TOpSplitRows, "row_conv": jlm.TOpRowConv, "gather": jlm.TOpGather,
	}
}

func towerSteps() map[string]jlm.PreKind {
	return map[string]jlm.PreKind{
		"resize": jlm.PreResize, "normalize": jlm.PreNormalize, "normalize_bf16": jlm.PreNormalizeBF16,
		"resize_live": jlm.PreResizeLive, "canvas": jlm.PreCanvas,
		"resize_short": PreResizeShort, "center_crop": PreCenterCrop, "tile": PreTile, "smart_resize": PreSmartResize,
		"limit_pad": PreLimitPad, "long_box": PreLongBox, "naflex": PreNaFlex, "cap_unit": PreCapUnit,
		"aspect_budget": PreAspectBudget, "tile_smart": PreTileSmart, "best_fit": PreBestFit, "slice_uhd": PreSliceUHD,
		"video_resize": PreVideoResize, "long_tiles": PreLongTiles,
	}
}

func towerChains() map[string]uint8 {
	return map[string]uint8{"aligned": jlm.ResampleAligned, "half_pixel": jlm.ResampleHalfPixel}
}

func towerTaps() map[string]uint8 {
	return map[string]uint8{"bilinear": jlm.TapsBilinear, "linear": jlm.TapsLinear, "cubic": jlm.TapsCubic,
		"antialias": jlm.TapsAntialias}
}

func towerFns() map[string]func(e *jlm.ExprBuilder, c float32) {
	return map[string]func(e *jlm.ExprBuilder, c float32){
		"gelu_erf":  func(e *jlm.ExprBuilder, _ float32) { e.GELUErf(jlm.Operand(0)) },
		"gelu_tanh": func(e *jlm.ExprBuilder, _ float32) { e.GELUTanh(jlm.Operand(0)) },
		"scale":     func(e *jlm.ExprBuilder, c float32) { e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(c)) },
		"swiglu":    func(e *jlm.ExprBuilder, _ float32) { e.Fn(jlm.FnMul, e.SiLU(jlm.Operand(0)), jlm.Operand(1)) },
	}
}

func towerNorms() map[string]jlm.NormKind {
	return map[string]jlm.NormKind{"layer": jlm.NormLayer, "rms": jlm.NormRMS}
}

// check refuses a statement naming an op, a step, a function, a norm or a
// role the container does not know, and compiles its expressions, each
// reading only the tower's facts.
func (t *mjTower) check() error {
	roles := rolesByName()
	role := func(n string) error {
		if _, ok := roles[strings.TrimSuffix(n, "?")]; !ok {
			return fmt.Errorf("%q is no role the container names", n)
		}
		return nil
	}
	for _, n := range t.Requires {
		if err := role(n); err != nil {
			return err
		}
	}
	var srcs []string
	for _, r := range t.Require {
		srcs = append(srcs, r.That)
	}
	pre, progs := t.Pre, [][]mjTowerOp{t.Entry, t.Head}
	if tp := t.Taps; tp != nil {
		if err := role(tp.Of); err != nil {
			return err
		}
		progs = append(progs, tp.Prog)
	}
	if vd := t.Video; vd != nil {
		pre, progs = append(slices.Clone(pre), vd.Pre...), append(progs, vd.Entry)
		srcs = append(srcs, vd.DefaultFps, vd.MaxFrames, vd.Group, vd.If)
		var pieces func([]mjPiece) error
		pieces = func(ps []mjPiece) error {
			for _, p := range ps {
				if f := p.Format; f != nil && f.Words != "" {
					srcs = append(srcs, f.Words)
				}
				if p.Value != nil {
					if _, err := videoExpr(*p.Value); err != nil {
						return err
					}
				}
				if err := pieces(p.Groups); err != nil {
					return err
				}
				if err := pieces(p.Between); err != nil {
					return err
				}
			}
			return nil
		}
		if err := pieces(vd.Prompt); err != nil {
			return fmt.Errorf("the video's prompt: %w", err)
		}
		for _, x := range []*mjVideoExpr{&vd.Count, &vd.Index, &vd.Sized, vd.Stride} {
			if x == nil {
				continue
			}
			if _, err := videoExpr(*x); err != nil {
				return fmt.Errorf("the video: %w", err)
			}
		}
	}
	for _, p := range pre {
		if _, ok := towerSteps()[p.Kind]; !ok {
			return fmt.Errorf("step %q is not one the converter states", p.Kind)
		}
		srcs = append(append(srcs, p.Ints...), p.Floats...)
	}
	for _, prog := range progs {
		for _, o := range prog {
			if _, ok := towerOps()[o.Op]; !ok && o.Op != "untile" {
				return fmt.Errorf("op %q is not one a tower program runs", o.Op)
			}
			if _, ok := towerFns()[o.Fn]; o.Fn != "" && !ok {
				return fmt.Errorf("function %q is not one a stated op runs", o.Fn)
			}
			if _, ok := towerChains()[o.Chain]; o.Chain != "" && !ok {
				return fmt.Errorf("resample chain %q is not aligned or half_pixel", o.Chain)
			}
			if _, ok := towerTaps()[o.Taps]; o.Taps != "" && !ok {
				return fmt.Errorf("resample taps %q are not ones the container names", o.Taps)
			}
			for _, w := range o.W {
				if err := role(w); err != nil {
					return err
				}
			}
			srcs = append(srcs, o.If, o.Span, o.Offset, o.Width, o.Eps, o.Const)
		}
	}
	b := t.Block
	if _, ok := towerNorms()[b.Norm]; !ok {
		return fmt.Errorf("norm %q is not layer or rms", b.Norm)
	}
	if b.Act != "" && b.Act != "gelu_erf" {
		return fmt.Errorf("activation %q is not the file's or gelu_erf", b.Act)
	}
	if _, ok := towerFns()[b.Fn]; b.Fn != "" && (!ok || b.Act != "" || b.Gated) {
		return fmt.Errorf("block function %q is not one a stated op runs, or is stated beside the file's activation", b.Fn)
	}
	for _, tb := range t.Tables {
		if tb.Kind != "sincos_axes" {
			return fmt.Errorf("table %q is not one the converter writes", tb.Kind)
		}
		srcs = append(srcs, tb.Width, tb.Length)
	}
	if m := t.Marks; m != nil {
		ids := len(m.IDs) > 0 || m.Grid != nil && len(m.Grid.Each) > 0
		if ids && (len(m.Names) > 0 || len(m.After) > 0 || m.Grid != nil && m.Grid.Name != "") {
			return fmt.Errorf("a picture's marks are rows of named tokens or of token ids, not both")
		}
		srcs = append(srcs, m.IDs...)
		if g := m.Grid; g != nil {
			srcs = append(append(srcs, g.Side, g.Name, g.If), g.Each...)
		}
	}
	srcs = append(srcs, b.Heads, b.KVHeads, b.HeadDim, b.Eps, b.AttnScale, b.Window, t.Span.GridDiv, t.Span.BidirIf)
	if f := b.Full; f != nil {
		srcs = append(srcs, f.Count, f.If)
	}
	if r := b.Rope; r != nil {
		srcs = append(srcs, r.Base, r.Dim, r.Origin)
		for _, g := range r.Runs {
			srcs = append(srcs, g.Count)
			for _, e := range g.Each {
				srcs = append(srcs, e.Pairs, e.Axis, e.Freq)
			}
		}
	}
	t.exprs = map[string]*jinja.Expr{}
	for _, src := range srcs {
		if src == "" || t.exprs[src] != nil {
			continue
		}
		x, err := jinja.CompileExpr(src)
		if err != nil {
			return err
		}
		for _, n := range x.Names() {
			if !slices.Contains([]string{"v", "has", "dims", "text", "filter", "prec", "j", "r", "c", "side", "i", "tap", "token", "tokens"}, n) {
				return fmt.Errorf("%q reads %q, which is not the tower's facts (v, has, dims, text), a resize's names "+
					"(filter, prec), a rotary run's place (j), a mark's cell (r, c, side), a block's place (i) or a tap's facts (tap)", src, n)
			}
		}
		t.exprs[src] = x
	}
	return nil
}

// towerFacts is what a tower statement's expressions read: v the tower's
// geometry, has its tensors outside the blocks by role name, dims their
// shapes, text the text model's facts a tower reads, filter and prec a
// resize's filters and arithmetics, and token and tokens a marker's ids.
func towerFacts(s *jlm.Source) map[string]jinja.Value {
	v := s.Vision
	list := func(xs [3]float32) jinja.Value {
		out := make([]jinja.Value, len(xs))
		for i, x := range xs {
			out[i] = jinja.NewFloat(float64(x))
		}
		return jinja.NewList(out)
	}
	geom := jinja.NewDict()
	for k, x := range map[string]uint32{"layers": v.NLayer, "embd": v.NEmbd, "heads": v.NHead, "ffn": v.NFFN,
		"image_size": v.ImageSize, "patch_size": v.PatchSize, "proj_dim": v.ProjDim, "scale": v.Scale,
		"min_tiles": v.MinTiles, "max_tiles": v.MaxTiles, "min_pixels": v.MinPixels, "max_pixels": v.MaxPixels,
		"win_pattern": v.WinPattern, "win_size": v.WinSize, "tile_size": v.TileSize, "widest_picture": widestPicture} {
		geom.AsDict().Set(k, jinja.NewInt(int64(x)))
	}
	geom.AsDict().Set("eps", jinja.NewFloat(float64(v.Eps)))
	geom.AsDict().Set("tolerance", jinja.NewFloat(float64(v.Tolerance)))
	geom.AsDict().Set("bicubic", jinja.NewBool(v.Flags.Has(jlm.FlagVisionBicubic)))
	geom.AsDict().Set("mean", list(v.Mean))
	geom.AsDict().Set("std", list(v.Std))
	has, dims := jinja.NewDict(), jinja.NewDict()
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Block != jlm.DenseBlock || !t.Role.Vision() {
			continue
		}
		has.AsDict().Set(t.Role.String(), jinja.NewBool(true))
		ds := make([]jinja.Value, t.NDim)
		for d := range ds {
			ds[d] = jinja.NewInt(int64(t.Dims[d]))
		}
		dims.AsDict().Set(t.Role.String(), jinja.NewList(ds))
	}
	filter, prec := jinja.NewDict(), jinja.NewDict()
	for k, f := range map[string]jlm.Filter{"two_tap": jlm.FilterTwoTap, "pil_bilinear": jlm.FilterPILBilinear,
		"pil_bicubic": jlm.FilterPILBicubic, "pil_lanczos": jlm.FilterPILLanczos} {
		filter.AsDict().Set(k, jinja.NewInt(int64(f)))
	}
	for k, p := range map[string]jlm.Precision{"float": jlm.PrecFloat, "pil": jlm.PrecPIL, "torch": jlm.PrecTorch} {
		prec.AsDict().Set(k, jinja.NewInt(int64(p)))
	}
	text := jinja.NewDict()
	if s.Config != nil {
		text.AsDict().Set("bidir_layers", jinja.NewInt(int64(len(s.Config.BidirLayers))))
	}
	piece := func(args []jinja.Value) (string, error) {
		if len(args) != 1 || !args[0].IsString() {
			return "", fmt.Errorf("a token's piece is one string")
		}
		return args[0].AsString(), nil
	}
	token := jinja.NewCallable("token", func(args []jinja.Value, _ map[string]jinja.Value) (jinja.Value, error) {
		p, err := piece(args)
		if err != nil {
			return jinja.Undefined(), err
		}
		ids, err := markerIDs(s, p)
		if err != nil {
			return jinja.Undefined(), err
		}
		return jinja.NewInt(int64(ids[0])), nil
	})
	tokens := jinja.NewCallable("tokens", func(args []jinja.Value, _ map[string]jinja.Value) (jinja.Value, error) {
		p, err := piece(args)
		if err != nil {
			return jinja.Undefined(), err
		}
		ids, err := markerIDs(s, p)
		if err != nil {
			// A piece the vocabulary has no token for is its characters'.
			ids = nil
			for _, ch := range p {
				id, err := markerIDs(s, string(ch))
				if err != nil {
					return jinja.Undefined(), err
				}
				ids = append(ids, id...)
			}
		}
		out := make([]jinja.Value, len(ids))
		for i, id := range ids {
			out[i] = jinja.NewInt(int64(id))
		}
		return jinja.NewList(out), nil
	})
	return map[string]jinja.Value{"v": geom, "has": has, "dims": dims, "text": text, "filter": filter, "prec": prec,
		"token": token, "tokens": tokens}
}

// state is the tower t states for s, refused where s lacks a tensor it
// requires or its facts fail a requirement.
func (t *mjTower) state(by string, s *jlm.Source) (*jlm.Tower, error) {
	proj := s.Vision.Projector
	facts := towerFacts(s)
	has := facts["has"].AsDict()
	held := func(n string) bool {
		v, ok := has.Get(n)
		return ok && v.IsTruthy()
	}
	value := func(src string) (jinja.Value, error) {
		x, err := t.exprs[src].Eval(facts, blockExprSteps)
		if err == nil && x.IsUndefined() {
			err = fmt.Errorf("%q is undefined", src)
		}
		if err != nil {
			return jinja.Undefined(), fmt.Errorf("convert: %s: %v: %w", by, proj, err)
		}
		return x, nil
	}
	num := func(src string) (float64, error) {
		if src == "" {
			return 0, nil
		}
		x, err := value(src)
		if err != nil {
			return 0, err
		}
		switch {
		case x.IsInt():
			return float64(x.AsInt()), nil
		case x.IsFloat():
			return x.AsFloat(), nil
		}
		return 0, fmt.Errorf("convert: %s: %v: %q is not a number", by, proj, src)
	}
	for _, n := range t.Requires {
		if !held(n) {
			return nil, fmt.Errorf("convert: %s: %v: the tower has no %s", by, proj, n)
		}
	}
	for _, r := range t.Require {
		x, err := value(r.That)
		if err != nil {
			return nil, err
		}
		if !x.IsTruthy() {
			return nil, fmt.Errorf("convert: %s: %v: %s", by, proj, r.Else)
		}
	}
	roles := rolesByName()
	prog := func(ops []mjTowerOp, nin int) (jlm.Program, error) {
		p := jlm.Program{NIn: uint16(nin)}
		named := map[string]uint16{"in": 0, "in1": 1}
		last := uint16(0)
		for _, o := range ops {
			if o.If != "" {
				on, err := value(o.If)
				if err != nil {
					return p, err
				}
				if !on.IsTruthy() {
					continue
				}
			}
			op := jlm.TowerOp{Code: towerOps()[o.Op]}
			if o.Op == "untile" {
				merge, err := num(o.Span)
				if err != nil {
					return p, err
				}
				window, err := num(o.Width)
				if err != nil {
					return p, err
				}
				if op, err = gather.Untile(int(merge), int(window)); err != nil {
					return p, fmt.Errorf("convert: %s: %v: %w", by, proj, err)
				}
			}
			for _, n := range o.In {
				i, ok := named[n]
				if !ok {
					return p, fmt.Errorf("convert: %s: %v: op %s reads %q, which no earlier op names", by, proj, o.Op, n)
				}
				op.In = append(op.In, i)
			}
			if o.In == nil {
				op.In = []uint16{last}
			}
			for _, w := range o.W {
				n, optional := strings.CutSuffix(w, "?")
				if optional && !held(n) {
					continue
				}
				op.Bind = append(op.Bind, roles[n])
			}
			for _, f := range []struct {
				src string
				to  *int32
			}{{o.Span, &op.Span}, {o.Offset, &op.Offset}, {o.Width, &op.Width}} {
				if o.Op == "untile" {
					break
				}
				x, err := num(f.src)
				if err != nil {
					return p, err
				}
				*f.to = int32(x)
			}
			eps, err := num(o.Eps)
			if err != nil {
				return p, err
			}
			op.Eps = float32(eps)
			op.Chain, op.Filter = towerChains()[o.Chain], towerTaps()[o.Taps]
			if o.Fn != "" {
				c, err := num(o.Const)
				if err != nil {
					return p, err
				}
				var e jlm.ExprBuilder
				towerFns()[o.Fn](&e, float32(c))
				op.Expr = &jlm.Expr{Terms: e.T}
			}
			p.Ops = append(p.Ops, op)
			last = uint16(nin + len(p.Ops) - 1)
			if o.Out != "" {
				named[o.Out] = last
			}
		}
		p.Out = last
		return p, nil
	}
	steps := func(sts []mjPreStep) ([]jlm.PreStep, error) {
		var out []jlm.PreStep
		for _, st := range sts {
			step := jlm.PreStep{Kind: towerSteps()[st.Kind]}
			for _, src := range st.Ints {
				x, err := num(src)
				if err != nil {
					return nil, err
				}
				step.Ints = append(step.Ints, int32(x))
			}
			for _, src := range st.Floats {
				x, err := num(src)
				if err != nil {
					return nil, err
				}
				step.Floats = append(step.Floats, float32(x))
			}
			out = append(out, step)
		}
		return out, nil
	}
	out := &jlm.Tower{Version: jlm.TowerVersion}
	var err error
	if out.Pre, err = steps(t.Pre); err != nil {
		return nil, err
	}
	if out.Entry, err = prog(t.Entry, 1); err != nil {
		return nil, err
	}
	if out.Head, err = prog(t.Head, 1); err != nil {
		return nil, err
	}
	b := t.Block
	var heads, kv, hd, eps float64
	for _, f := range []struct {
		src string
		to  *float64
	}{{b.Heads, &heads}, {b.KVHeads, &kv}, {b.HeadDim, &hd}, {b.Eps, &eps}} {
		if *f.to, err = num(f.src); err != nil {
			return nil, err
		}
	}
	act, err := s.Vision.Activation(b.Gated)
	switch {
	case b.Act == "gelu_erf":
		act, err = erfGELU(s.Vision, by)
	case b.Fn != "":
		var e jlm.ExprBuilder
		towerFns()[b.Fn](&e, 0)
		act, err = jlm.Expr{Terms: e.T}, nil
	}
	if err != nil {
		return nil, err
	}
	attn, err := num(b.AttnScale)
	if err != nil {
		return nil, err
	}
	window, err := num(b.Window)
	if err != nil {
		return nil, err
	}
	out.Block = jlm.TowerBlock{NHead: uint32(heads), NKVHead: uint32(kv), HeadDim: uint32(hd), Norm: towerNorms()[b.Norm],
		Eps: float32(eps), AttnScale: float32(attn), Act: act, VNorm: b.VNorm, Window: int32(window)}
	if f := b.Full; f != nil {
		n, err := num(f.Count)
		if err != nil {
			return nil, err
		}
		for i := range int(n) {
			facts["i"] = jinja.NewInt(int64(i))
			on, err := value(f.If)
			if err != nil {
				return nil, err
			}
			if on.IsTruthy() {
				out.Block.Full = append(out.Block.Full, int32(i))
			}
		}
		delete(facts, "i")
	}
	if r := b.Rope; r != nil {
		rope := &out.Block.Rope
		var base, dim, origin float64
		for _, f := range []struct {
			src string
			to  *float64
		}{{r.Base, &base}, {r.Dim, &dim}, {r.Origin, &origin}} {
			if *f.to, err = num(f.src); err != nil {
				return nil, err
			}
		}
		rope.Base, rope.Dim, rope.Neox, rope.Origin = float32(base), int32(dim), r.Neox, int32(origin)
		for _, g := range r.Runs {
			n, err := num(g.Count)
			if err != nil {
				return nil, err
			}
			for j := range int(n) {
				facts["j"] = jinja.NewInt(int64(j))
				for _, e := range g.Each {
					var run [3]float64
					for k, src := range []string{e.Pairs, e.Axis, e.Freq} {
						if run[k], err = num(src); err != nil {
							return nil, err
						}
					}
					rope.Runs = append(rope.Runs, jlm.TowerRopeRun{Pairs: int32(run[0]), Axis: int32(run[1]), Freq: int32(run[2])})
				}
			}
			delete(facts, "j")
		}
	}
	sp := t.Span
	div, err := num(sp.GridDiv)
	if err != nil {
		return nil, err
	}
	out.Span = jlm.TowerSpan{BeforeText: sp.Before, AfterText: sp.After, IndexBefore: sp.IndexBefore, IndexAfter: sp.IndexAfter,
		Mark: sp.Mark, Bidir: sp.Bidir, GridDiv: int32(div)}
	if sp.BidirIf != "" {
		on, err := value(sp.BidirIf)
		if err != nil {
			return nil, err
		}
		out.Span.Bidir = on.IsTruthy()
	}
	if len(sp.BeforeMarks) != 0 {
		if out.Span.Before, err = markerIDs(s, sp.BeforeMarks...); err != nil {
			return nil, err
		}
	}
	if len(sp.AfterMarks) != 0 {
		if out.Span.After, err = markerIDs(s, sp.AfterMarks...); err != nil {
			return nil, err
		}
	}
	video := t.Video != nil
	if video && t.Video.If != "" {
		on, err := value(t.Video.If)
		if err != nil {
			return nil, err
		}
		video = on.IsTruthy()
	}
	if vd := t.Video; video {
		v := &jlm.Video{Once: vd.Once}
		var fps, most, group float64
		for _, f := range []struct {
			src string
			to  *float64
		}{{vd.DefaultFps, &fps}, {vd.MaxFrames, &most}, {vd.Group, &group}} {
			if *f.to, err = num(f.src); err != nil {
				return nil, err
			}
		}
		v.DefaultFps, v.MaxFrames, v.Group = float32(fps), int32(most), int32(group)
		for _, x := range []struct {
			from mjVideoExpr
			to   *jlm.Expr
		}{{vd.Count, &v.Count}, {vd.Index, &v.Index}, {vd.Sized, &v.Sized}} {
			if *x.to, err = videoExpr(x.from); err != nil {
				return nil, fmt.Errorf("convert: %s: %v: the video: %w", by, proj, err)
			}
		}
		if vd.Stride != nil {
			x, err := videoExpr(*vd.Stride)
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %v: the video's stride: %w", by, proj, err)
			}
			v.Stride = &x
		}
		if v.Pre, err = steps(vd.Pre); err != nil {
			return nil, err
		}
		if len(vd.Entry) != 0 {
			if v.Entry, err = prog(vd.Entry, max(vd.Inputs, 1)); err != nil {
				return nil, err
			}
		}
		if v.Prompt, err = prompt(vd.Prompt, s, &out.Span, num); err != nil {
			return nil, fmt.Errorf("convert: %s: %v: the video's prompt: %w", by, proj, err)
		}
		out.Video = v
		if vd.Template != "" {
			withVideoTemplate(s.Vocab, vd.Template)
		}
	}
	if tp := t.Taps; tp != nil {
		at := map[int32][]uint64{}
		r := roles[tp.Of]
		for i := range s.Tensors {
			if x := &s.Tensors[i]; x.Block == jlm.DenseBlock && x.Role == r {
				at[x.Index] = x.Dims[:x.NDim]
			}
		}
		for _, b := range slices.Sorted(maps.Keys(at)) {
			tap := jinja.NewDict()
			tap.AsDict().Set("block", jinja.NewInt(int64(b)))
			ds := make([]jinja.Value, len(at[b]))
			for d, x := range at[b] {
				ds[d] = jinja.NewInt(int64(x))
			}
			tap.AsDict().Set("dims", jinja.NewList(ds))
			facts["tap"] = tap
			p, err := prog(tp.Prog, 1)
			if err != nil {
				return nil, err
			}
			out.Taps = append(out.Taps, jlm.TowerTap{Block: b, Prog: p})
		}
		delete(facts, "tap")
	}
	for _, tb := range t.Tables {
		w, err := num(tb.Width)
		if err != nil {
			return nil, err
		}
		n, err := num(tb.Length)
		if err != nil {
			return nil, err
		}
		if err := sincosAxes(s, int(w), int(n)); err != nil {
			return nil, err
		}
	}
	if m := t.Marks; m != nil && (len(m.IDs) > 0 || m.Grid != nil && len(m.Grid.Each) > 0) {
		var ids []int32
		add := func(src string) error {
			x, err := value(src)
			if err != nil {
				return err
			}
			switch {
			case x.IsInt():
				ids = append(ids, int32(x.AsInt()))
			case x.IsList():
				for _, y := range x.AsList().Items {
					ids = append(ids, int32(y.AsInt()))
				}
			default:
				return fmt.Errorf("convert: %s: %v: a mark %q is no token id", by, proj, src)
			}
			return nil
		}
		for _, src := range m.IDs {
			if err := add(src); err != nil {
				return nil, err
			}
		}
		if g := m.Grid; g != nil {
			n, err := num(g.Side)
			if err != nil {
				return nil, err
			}
			for r := range int(n) {
				for c := range int(n) {
					facts["r"], facts["c"] = jinja.NewInt(int64(r)), jinja.NewInt(int64(c))
					for _, src := range g.Each {
						if err := add(src); err != nil {
							return nil, err
						}
					}
				}
			}
			delete(facts, "r")
			delete(facts, "c")
		}
		if err := markRows(s, ids); err != nil {
			return nil, err
		}
	} else if m != nil {
		names := slices.Clone(m.Names)
		if g := m.Grid; g != nil {
			n, err := num(g.Side)
			if err != nil {
				return nil, err
			}
			side := int(n)
			facts["side"] = jinja.NewInt(int64(side))
			grid := make([]string, side*side)
			for r := range side {
				for c := range side {
					facts["r"], facts["c"] = jinja.NewInt(int64(r)), jinja.NewInt(int64(c))
					on, err := value(g.If)
					if err != nil {
						return nil, err
					}
					if !on.IsTruthy() {
						continue
					}
					x, err := value(g.Name)
					if err != nil {
						return nil, err
					}
					if !x.IsString() {
						return nil, fmt.Errorf("convert: %s: %v: a mark's name %q is not a string", by, proj, g.Name)
					}
					grid[r*side+c] = x.AsString()
				}
			}
			delete(facts, "r")
			delete(facts, "c")
			delete(facts, "side")
			names = append(names, grid...)
		}
		if err := markTable(s, append(names, m.After...)); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// libraryTower is what the projector library states for projector p's
// tower, and the file stating it; nil where it states none.
func libraryTower(p jlm.Projector) (*mjTower, string, error) {
	lib, err := projectorLibrary()
	if err != nil {
		return nil, "", err
	}
	if e := lib[p.String()]; e != nil && e.Tower != nil {
		return e.Tower, "convert/projectors.json", nil
	}
	return nil, "", nil
}
