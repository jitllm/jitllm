package jlm

import "fmt"

// TowerVersion is the stated tower's revision. A record of a later one is
// refused, not read as this one.
const TowerVersion = 3

// Tower is a vision tower as the converter states it: what the engine builds
// the tower from, with no family's name in it. The converter knows a family
// (RULE 7) and writes down what its reference does: the steps a picture
// passes through, the program that turns its pixels into the first block's
// rows (Entry), the repeated block (Block), the program that turns the last
// block's rows into the language model's (Head), and where the picture's rows
// stand in a prompt (Span). Projector stays the family's label, which the
// engine never switches on.
type Tower struct {
	Version uint16
	Pre     []PreStep
	Entry   Program
	Block   TowerBlock
	Head    Program
	Span    TowerSpan
}

// PreKind is one preprocessing step: an algorithm with its parameters, never
// a family.
type PreKind uint16

const (
	// PreResize resamples the whole picture, aspect not kept, to Ints[0] x
	// Ints[1] pixels with filter Ints[2] (a Filter) in arithmetic Ints[3]
	// (a Precision).
	PreResize PreKind = 1
	// PreNormalize turns each channel c of a sample read as [0, 1] into
	// (v - Floats[c]) / Floats[3+c]: the mean, then the std.
	PreNormalize PreKind = 2
	// PreResizeShort resamples the picture, aspect kept, so its shorter side
	// is Ints[0] pixels and its longer int(Ints[0] * long / short), with
	// filter Ints[1] in arithmetic Ints[2].
	PreResizeShort PreKind = 3
	// PreCenterCrop is the Ints[0] x Ints[1] window of the picture as the
	// step before left it, at ((w - Ints[0]) / 2, (h - Ints[1]) / 2)
	// truncated.
	PreCenterCrop PreKind = 4
)

// Filter is a resampling filter.
type Filter uint16

const (
	// FilterTwoTap is bilinear between the two source samples around each
	// output's centre ((i+0.5)*src/n - 0.5), clamped at the edges, with no
	// antialiasing on a reduction.
	FilterTwoTap Filter = 1
	// FilterPILBilinear and FilterPILBicubic are Pillow's antialiased
	// filters: the kernel widened by the reduction's scale.
	FilterPILBilinear Filter = 2
	FilterPILBicubic  Filter = 3
)

// Precision is the arithmetic a resample keeps its samples, taps and sums in.
type Precision uint16

const (
	// PrecFloat is float32 over the decoded samples at their own depth,
	// nothing rounded between the passes.
	PrecFloat Precision = 1
	// PrecPIL is Pillow's: 8-bit RGB in, fixed-point taps of the precision
	// Pillow picks for the filter, rounded to 8 bits after each pass.
	PrecPIL Precision = 2
)

// PreStep is one step of the picture's preprocessing.
type PreStep struct {
	Kind   PreKind
	Ints   []int32
	Floats []float32
}

// OpCode is a program's operation. The codes are the container's: the
// engine maps each to its own graph's op, and a code it does not know refuses
// the container.
type OpCode uint16

const (
	// TOpPatchify is the picture's Span x Span patches as rows, raster
	// order, each patch channel-planar, after Offset rows left for a class
	// row (graph.OpPatchify).
	TOpPatchify OpCode = 1
	// TOpLinear is In[0] through the matrix Bind[0] (out width its rows),
	// plus the bias Bind[1] where bound.
	TOpLinear OpCode = 2
	// TOpPosTable is each row's row of the position table Bind[0], from row
	// Offset on (a table at the tower's own grid).
	TOpPosTable OpCode = 3
	// TOpAdd is In[0] plus In[1].
	TOpAdd OpCode = 4
	// TOpLayerNorm is In[0]'s LayerNorm, weight Bind[0], bias Bind[1]
	// where bound, epsilon Eps.
	TOpLayerNorm OpCode = 5
	// TOpRMSNorm is In[0]'s RMS norm, weight Bind[0], epsilon Eps.
	TOpRMSNorm OpCode = 6
	// TOpRegroup is space-to-depth: each Span x Span block of the grid's
	// rows one row, the block's rows in raster order (graph.OpRegroup).
	TOpRegroup OpCode = 7
	// TOpLeadRow puts Bind[0]'s one row in In[0]'s first Offset rows (a
	// class row before the patches).
	TOpLeadRow OpCode = 8
	// TOpDropRows is In[0] from row Offset on.
	TOpDropRows OpCode = 9
	// TOpStated is the expression Expr over the In values, Width wide.
	TOpStated OpCode = 10
	// TOpPool is an average pool: each Span x Span block of the grid's rows
	// their mean, the blocks in raster order.
	TOpPool  OpCode = 11
	topCodes OpCode = 12
)

// TowerOp is one operation of a program: its inputs are values (a program's
// inputs first, then each op's output in order), its weights roles of the
// container's tensors.
type TowerOp struct {
	Code   OpCode
	In     []uint16
	Bind   []Role
	Span   int32
	Offset int32
	Width  int32
	Eps    float32
	Expr   *Expr
}

// Program is a short graph: NIn inputs, then each op's output, and the value
// Out is the program's result.
type Program struct {
	NIn uint16
	Ops []TowerOp
	Out uint16
}

// NormKind is a block's normalization.
type NormKind uint8

const (
	NormLayer NormKind = 1
	NormRMS   NormKind = 2
)

// TowerBlock is the tower's repeated block. Which tensors a block has (the
// biases, a gate, post-norms) is its inventory; what that cannot say is here.
type TowerBlock struct {
	NHead, NKVHead, HeadDim uint32
	Norm                    NormKind
	Eps                     float32
	// AttnScale is the score scale, 0 for 1/sqrt(HeadDim).
	AttnScale float32
	// Act is the MLP's activation over its gate and up where gated, its up
	// alone where not.
	Act Expr
}

// TowerSpan is a picture's place in a prompt: the token ids before its rows,
// the ids after them, and text before and after those -- text the prompt's
// tokenizer encodes with the prompt beside it, as a reference processor
// expands its placeholder into a string before tokenizing the whole. The
// rows are the head's, one a token; Bidir says they attend to each other
// both ways in the language model. Mark is the text a message read as a
// string carries for each of its pictures, before its text -- what the
// reference processor's template writes for an image part -- for a chat
// template that reads no typed parts; empty where the family has none.
type TowerSpan struct {
	Before, After         []int32
	BeforeText, AfterText string
	Bidir                 bool
	Mark                  string
}

// Valid says the record is one this build can read: a known version, known
// steps and op codes, every value read after it is made, every op's
// required weights and attributes stated. Whether the bound tensors exist is
// the container's to say (ValidAgainst).
func (t *Tower) Valid() error {
	if t.Version != TowerVersion {
		return fmt.Errorf("jlm: the stated tower is revision %d and this build reads %d", t.Version, TowerVersion)
	}
	for i, s := range t.Pre {
		switch s.Kind {
		case PreResize:
			if len(s.Ints) != 4 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || !resizes(Filter(s.Ints[2]), Precision(s.Ints[3])) {
				return fmt.Errorf("jlm: the tower's step %d resizes with %v", i, s.Ints)
			}
		case PreResizeShort:
			if len(s.Ints) != 3 || s.Ints[0] <= 0 || !resizes(Filter(s.Ints[1]), Precision(s.Ints[2])) {
				return fmt.Errorf("jlm: the tower's step %d resizes the shorter side with %v", i, s.Ints)
			}
		case PreCenterCrop:
			if len(s.Ints) != 2 || s.Ints[0] <= 0 || s.Ints[1] <= 0 {
				return fmt.Errorf("jlm: the tower's step %d crops to %v", i, s.Ints)
			}
		case PreNormalize:
			if len(s.Floats) != 6 {
				return fmt.Errorf("jlm: the tower's step %d normalizes with %d numbers, want 6", i, len(s.Floats))
			}
			for c := range 3 {
				if s.Floats[3+c] == 0 {
					return fmt.Errorf("jlm: the tower's step %d divides channel %d by a std of 0", i, c)
				}
			}
		default:
			return fmt.Errorf("jlm: the tower's step %d is kind %d, which this build does not know", i, s.Kind)
		}
	}
	if err := t.Entry.valid("entry"); err != nil {
		return err
	}
	if err := t.Head.valid("head"); err != nil {
		return err
	}
	b := t.Block
	switch {
	case b.NHead == 0 || b.NKVHead == 0 || b.HeadDim == 0 || b.NHead%b.NKVHead != 0:
		return fmt.Errorf("jlm: the tower's block has %d heads over %d kv heads of %d", b.NHead, b.NKVHead, b.HeadDim)
	case b.Norm != NormLayer && b.Norm != NormRMS:
		return fmt.Errorf("jlm: the tower's block normalizes by kind %d", b.Norm)
	case b.Eps <= 0:
		return fmt.Errorf("jlm: the tower's block has an epsilon of %v", b.Eps)
	}
	if err := b.Act.Valid(); err != nil {
		return fmt.Errorf("jlm: the tower's block activation: %w", err)
	}
	return nil
}

// resizes says filter f is run in arithmetic p: the two-tap filter in float,
// Pillow's in Pillow's fixed point.
func resizes(f Filter, p Precision) bool {
	switch f {
	case FilterTwoTap:
		return p == PrecFloat
	case FilterPILBilinear, FilterPILBicubic:
		return p == PrecPIL
	}
	return false
}

func (p *Program) valid(name string) error {
	if p.NIn == 0 || len(p.Ops) == 0 {
		return fmt.Errorf("jlm: the tower's %s has %d inputs and %d ops", name, p.NIn, len(p.Ops))
	}
	for i, op := range p.Ops {
		made := int(p.NIn) + i
		for _, v := range op.In {
			if int(v) >= made {
				return fmt.Errorf("jlm: the tower's %s op %d reads value %d before it is made", name, i, v)
			}
		}
		in, bind := op.arity()
		switch {
		case op.Code == 0 || op.Code >= topCodes:
			return fmt.Errorf("jlm: the tower's %s op %d is code %d, which this build does not know", name, i, op.Code)
		case in >= 0 && len(op.In) != in:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) reads %d values, want %d", name, i, op.Code, len(op.In), in)
		case len(op.Bind) < bind[0] || len(op.Bind) > bind[1]:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) binds %d weights, want %d to %d", name, i, op.Code, len(op.Bind), bind[0], bind[1])
		case (op.Code == TOpPatchify || op.Code == TOpRegroup || op.Code == TOpPool) && op.Span <= 0:
			return fmt.Errorf("jlm: the tower's %s op %d groups by %d", name, i, op.Span)
		case (op.Code == TOpLayerNorm || op.Code == TOpRMSNorm) && op.Eps <= 0:
			return fmt.Errorf("jlm: the tower's %s op %d has an epsilon of %v", name, i, op.Eps)
		case op.Code == TOpStated && (op.Expr == nil || op.Width <= 0):
			return fmt.Errorf("jlm: the tower's %s op %d states no expression or no width", name, i)
		case op.Offset < 0:
			return fmt.Errorf("jlm: the tower's %s op %d offsets by %d", name, i, op.Offset)
		}
		if op.Expr != nil {
			if err := op.Expr.Valid(); err != nil {
				return fmt.Errorf("jlm: the tower's %s op %d: %w", name, i, err)
			}
			if n := op.Expr.Reads(); n != len(op.In) {
				return fmt.Errorf("jlm: the tower's %s op %d states an expression of %d operands over %d", name, i, n, len(op.In))
			}
		}
	}
	if int(p.Out) >= int(p.NIn)+len(p.Ops) {
		return fmt.Errorf("jlm: the tower's %s returns value %d, past its last", name, p.Out)
	}
	return nil
}

// arity is the values op reads (-1 for any number) and the least and most
// weights it binds.
func (op *TowerOp) arity() (in int, bind [2]int) {
	switch op.Code {
	case TOpPatchify, TOpDropRows, TOpRegroup, TOpPool:
		return 1, [2]int{0, 0}
	case TOpLinear, TOpLayerNorm:
		return 1, [2]int{1, 2}
	case TOpRMSNorm, TOpLeadRow:
		return 1, [2]int{1, 1}
	case TOpPosTable:
		return 0, [2]int{1, 1}
	case TOpAdd:
		return 2, [2]int{0, 0}
	case TOpStated:
		return -1, [2]int{0, 0}
	}
	return -1, [2]int{0, 0}
}

// Roles is every role the tower's programs bind, each once, in order.
func (t *Tower) Roles() []Role {
	var out []Role
	seen := map[Role]bool{}
	for _, p := range []*Program{&t.Entry, &t.Head} {
		for _, op := range p.Ops {
			for _, r := range op.Bind {
				if !seen[r] {
					seen[r] = true
					out = append(out, r)
				}
			}
		}
	}
	return out
}

func encodeTower(w *wbuf, t *Tower) {
	w.u16(t.Version)
	w.u32(uint32(len(t.Pre)))
	for _, s := range t.Pre {
		w.u16(uint16(s.Kind))
		w.u32(uint32(len(s.Ints)))
		for _, v := range s.Ints {
			w.i32(v)
		}
		w.u32(uint32(len(s.Floats)))
		for _, v := range s.Floats {
			w.f32(v)
		}
	}
	encodeProgram(w, &t.Entry)
	b := &t.Block
	w.u32(b.NHead)
	w.u32(b.NKVHead)
	w.u32(b.HeadDim)
	w.u8(uint8(b.Norm))
	w.f32(b.Eps)
	w.f32(b.AttnScale)
	encodeExpr(w, &b.Act)
	encodeProgram(w, &t.Head)
	for _, ids := range [][]int32{t.Span.Before, t.Span.After} {
		w.u32(uint32(len(ids)))
		for _, id := range ids {
			w.i32(id)
		}
	}
	w.str(t.Span.BeforeText)
	w.str(t.Span.AfterText)
	w.bool(t.Span.Bidir)
	w.str(t.Span.Mark)
}

func encodeProgram(w *wbuf, p *Program) {
	w.u16(p.NIn)
	w.u32(uint32(len(p.Ops)))
	for i := range p.Ops {
		op := &p.Ops[i]
		w.u16(uint16(op.Code))
		w.u32(uint32(len(op.In)))
		for _, v := range op.In {
			w.u16(v)
		}
		w.u32(uint32(len(op.Bind)))
		for _, r := range op.Bind {
			w.u16(uint16(r))
		}
		w.i32(op.Span)
		w.i32(op.Offset)
		w.i32(op.Width)
		w.f32(op.Eps)
		w.bool(op.Expr != nil)
		if op.Expr != nil {
			encodeExpr(w, op.Expr)
		}
	}
	w.u16(p.Out)
}

func decodeTower(r *rbuf) *Tower {
	t := &Tower{Version: r.u16()}
	// Each count is bounded by its element's least size in what is left, so
	// a corrupt record cannot size an allocation (RULE 9).
	for range r.count(10) {
		s := PreStep{Kind: PreKind(r.u16())}
		for range r.count(4) {
			s.Ints = append(s.Ints, r.i32())
		}
		for range r.count(4) {
			s.Floats = append(s.Floats, r.f32())
		}
		t.Pre = append(t.Pre, s)
	}
	decodeProgram(r, &t.Entry)
	b := &t.Block
	b.NHead, b.NKVHead, b.HeadDim = r.u32(), r.u32(), r.u32()
	b.Norm = NormKind(r.u8())
	b.Eps, b.AttnScale = r.f32(), r.f32()
	decodeExpr(r, &b.Act)
	decodeProgram(r, &t.Head)
	for _, ids := range []*[]int32{&t.Span.Before, &t.Span.After} {
		for range r.count(4) {
			*ids = append(*ids, r.i32())
		}
	}
	t.Span.BeforeText, t.Span.AfterText = r.str(), r.str()
	t.Span.Bidir = r.bool()
	t.Span.Mark = r.str()
	return t
}

func decodeProgram(r *rbuf, p *Program) {
	p.NIn = r.u16()
	for range r.count(27) {
		op := TowerOp{Code: OpCode(r.u16())}
		for range r.count(2) {
			op.In = append(op.In, r.u16())
		}
		for range r.count(2) {
			op.Bind = append(op.Bind, Role(r.u16()))
		}
		op.Span, op.Offset, op.Width = r.i32(), r.i32(), r.i32()
		op.Eps = r.f32()
		if r.bool() {
			op.Expr = &Expr{}
			decodeExpr(r, op.Expr)
		}
		p.Ops = append(p.Ops, op)
	}
	p.Out = r.u16()
}
