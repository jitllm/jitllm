package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The tensor primitives a model.json composes its transforms from. Each is one
// general operation on a tensor -- an elementwise map, a reshape, an axis
// permutation, a slice, a concatenation, a stack, a repeat, a storage type --
// and knows no model. A model's recipe (a rotary row order, a (1 + w) norm, two
// projections interleaved by head group) is a composition of them written in
// its model.json, every size and axis an expression over config.json.

// mjVal is a tensor between primitives: its shape slowest axis first, as torch
// writes it, its element type (F32 or F16) and its bytes. A dry value has no
// bytes: it sizes a result before anything is read.
type mjVal struct {
	shape []uint64
	typ   jlm.Type
	data  []byte
}

func (v mjVal) dry() bool { return v.data == nil }

func (v mjVal) count() uint64 { return mjProd(v.shape) }

func mjProd(s []uint64) uint64 {
	n := uint64(1)
	for _, d := range s {
		n *= d
	}
	return n
}

func mjElemSize(t jlm.Type) (uint64, error) {
	switch t {
	case jlm.TypeF32:
		return 4, nil
	case jlm.TypeF16:
		return 2, nil
	}
	return 0, fmt.Errorf("a %v tensor is not elements a primitive can move", t)
}

// mjSize is a size, an axis or a count as a file writes it: a number, or an
// expression over config.json.
type mjSize string

func (s *mjSize) UnmarshalJSON(b []byte) error {
	var str string
	if json.Unmarshal(b, &str) == nil {
		*s = mjSize(str)
		return nil
	}
	var f float64
	if err := json.Unmarshal(b, &f); err != nil {
		return fmt.Errorf("a size is a number or an expression, not %s", b)
	}
	*s = mjSize(strconv.FormatFloat(f, 'g', -1, 64))
	return nil
}

// mjOp is one primitive as a file writes it; exactly one field is set.
type mjOp struct {
	// Map is an elementwise expression over the tensors in hand, x the first,
	// y the second, z the third, with exp, log, sqrt, tanh and softplus; config paths
	// are constants. The result is float32.
	Map string `json:"map"`
	// Reshape is the new shape, slowest axis first; one size may be -1, the
	// rest's quotient. No byte moves.
	Reshape []mjSize `json:"reshape"`
	// Transpose permutes the axes: the result's axis i is the input's
	// Transpose[i].
	Transpose []mjSize `json:"transpose"`
	// Slice is [axis, start, length].
	Slice []mjSize `json:"slice"`
	// Concat joins the tensors in hand along an existing axis, Stack along a
	// new one at that place.
	Concat *mjSize `json:"concat"`
	Stack  *mjSize `json:"stack"`
	// Repeat is [axis, times]: each index along the axis, times times over
	// in place (torch's repeat_interleave).
	Repeat []mjSize `json:"repeat"`
	// Type is the storage type: "f32" or "f16".
	Type string `json:"type"`
}

func (o mjOp) name() string {
	var set []string
	if o.Map != "" {
		set = append(set, "map")
	}
	if o.Reshape != nil {
		set = append(set, "reshape")
	}
	if o.Transpose != nil {
		set = append(set, "transpose")
	}
	if o.Slice != nil {
		set = append(set, "slice")
	}
	if o.Concat != nil {
		set = append(set, "concat")
	}
	if o.Stack != nil {
		set = append(set, "stack")
	}
	if o.Repeat != nil {
		set = append(set, "repeat")
	}
	if o.Type != "" {
		set = append(set, "type")
	}
	if len(set) != 1 {
		return strings.Join(set, "+")
	}
	return set[0]
}

// mjPipe is a tensor's primitives in order.
type mjPipe []mjOp

// check refuses at lowering what no config could make right.
func (p mjPipe) check() error {
	for i, o := range p {
		switch n := o.name(); n {
		case "map":
			x, err := parseExpr(o.Map)
			if err != nil {
				return err
			}
			if err := mjMapCheck(x); err != nil {
				return fmt.Errorf("map %q: %w", o.Map, err)
			}
		case "slice", "repeat":
			if want := map[string]int{"slice": 3, "repeat": 2}[n]; len(argsOf(o)) != want {
				return fmt.Errorf("%s takes %d values, not %d", n, want, len(argsOf(o)))
			}
		case "type":
			if o.Type != "f32" && o.Type != "f16" {
				return fmt.Errorf("type %q is not f32 or f16", o.Type)
			}
		case "reshape", "transpose", "concat", "stack":
		default:
			return fmt.Errorf("step %d names %q; one primitive a step is the rule", i, n)
		}
		for _, s := range argsOf(o) {
			if _, err := parseExpr(string(s)); err != nil {
				return err
			}
		}
	}
	return nil
}

func argsOf(o mjOp) []mjSize {
	switch {
	case o.Reshape != nil:
		return o.Reshape
	case o.Transpose != nil:
		return o.Transpose
	case o.Slice != nil:
		return o.Slice
	case o.Repeat != nil:
		return o.Repeat
	case o.Concat != nil:
		return []mjSize{*o.Concat}
	case o.Stack != nil:
		return []mjSize{*o.Stack}
	}
	return nil
}

// run applies the primitives to the tensors in hand; a dry input sizes the
// result without bytes. What is left must be one tensor.
func (p mjPipe) run(env exprEnv, in []mjVal) (mjVal, error) {
	vs := in
	for _, o := range p {
		args := make([]int64, 0, 4)
		for _, s := range argsOf(o) {
			n, err := parseExpr(string(s))
			if err != nil {
				return mjVal{}, err
			}
			v, err := evalMJ(n, env)
			if err != nil {
				return mjVal{}, err
			}
			if v == nil {
				return mjVal{}, fmt.Errorf("%s: %q is absent, and the file states no default", o.name(), s)
			}
			f, ok := v.(float64)
			if !ok || f != math.Trunc(f) || math.Abs(f) > 1<<40 {
				return mjVal{}, fmt.Errorf("%s: %q is %v, not a whole number", o.name(), s, v)
			}
			args = append(args, int64(f))
		}
		var out mjVal
		var err error
		switch n := o.name(); n {
		case "map":
			out, err = mjMap(o.Map, env, vs)
		case "concat":
			out, err = mjConcat(vs, args[0])
		case "stack":
			out, err = mjStack(vs, args[0])
		default:
			if len(vs) != 1 {
				return mjVal{}, fmt.Errorf("%s takes one tensor, and %d are in hand: stack, concat or map them first", n, len(vs))
			}
			switch n {
			case "reshape":
				out, err = mjReshape(vs[0], args)
			case "transpose":
				out, err = mjTranspose(vs[0], args)
			case "slice":
				out, err = mjSlice(vs[0], args[0], args[1], args[2])
			case "repeat":
				out, err = mjRepeat(vs[0], args[0], args[1])
			case "type":
				out, err = mjRetype(vs[0], o.Type)
			}
		}
		if err != nil {
			return mjVal{}, err
		}
		vs = []mjVal{out}
	}
	if len(vs) != 1 {
		return mjVal{}, fmt.Errorf("%d tensors are left; stack, concat or map them into one", len(vs))
	}
	return vs[0], nil
}

func mjAxis(axis int64, rank int) (int, error) {
	if axis < 0 || axis >= int64(rank) {
		return 0, fmt.Errorf("axis %d of a %d-axis tensor", axis, rank)
	}
	return int(axis), nil
}

func mjReshape(v mjVal, sizes []int64) (mjVal, error) {
	shape := make([]uint64, len(sizes))
	infer, known := -1, uint64(1)
	for i, s := range sizes {
		switch {
		case s == -1 && infer < 0:
			infer = i
		case s <= 0:
			return mjVal{}, fmt.Errorf("reshape %v: a size of %d", sizes, s)
		default:
			shape[i] = uint64(s)
			known *= uint64(s)
		}
	}
	if infer >= 0 {
		if v.count()%known != 0 {
			return mjVal{}, fmt.Errorf("reshape %v of %v: %d elements do not divide by %d", sizes, v.shape, v.count(), known)
		}
		shape[infer] = v.count() / known
	}
	if mjProd(shape) != v.count() {
		return mjVal{}, fmt.Errorf("reshape %v of %v: %d elements into %d", sizes, v.shape, v.count(), mjProd(shape))
	}
	v.shape = shape
	return v, nil
}

func mjTranspose(v mjVal, perm []int64) (mjVal, error) {
	r := len(v.shape)
	if len(perm) != r {
		return mjVal{}, fmt.Errorf("transpose %v of a %d-axis tensor", perm, r)
	}
	seen := make([]bool, r)
	shape := make([]uint64, r)
	for i, a := range perm {
		ax, err := mjAxis(a, r)
		if err != nil || seen[ax] {
			return mjVal{}, fmt.Errorf("transpose %v is not a permutation of %d axes", perm, r)
		}
		seen[ax], shape[i] = true, v.shape[ax]
	}
	es, err := mjElemSize(v.typ)
	if err != nil {
		return mjVal{}, err
	}
	out := mjVal{shape: shape, typ: v.typ}
	if v.dry() {
		return out, nil
	}
	// in strides, slowest first, in elements
	stride := make([]uint64, r)
	s := uint64(1)
	for i := r - 1; i >= 0; i-- {
		stride[i], s = s, s*v.shape[i]
	}
	// The trailing axes the permutation leaves in place move as one run.
	k := r
	for k > 0 && perm[k-1] == int64(k-1) {
		k--
	}
	run := mjProd(v.shape[k:]) * es
	out.data = make([]byte, len(v.data))
	idx := make([]uint64, k)
	for o := uint64(0); o < mjProd(shape[:k]); o++ {
		src := uint64(0)
		for i := range idx {
			src += idx[i] * stride[perm[i]]
		}
		copy(out.data[o*run:(o+1)*run], v.data[src*es:src*es+run])
		for i := k - 1; i >= 0; i-- {
			if idx[i]++; idx[i] < shape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return out, nil
}

// mjSpan is the elements before an axis, and the bytes of one index of it.
func mjSpan(v mjVal, ax int) (outer, inner uint64, err error) {
	es, err := mjElemSize(v.typ)
	if err != nil {
		return 0, 0, err
	}
	return mjProd(v.shape[:ax]), mjProd(v.shape[ax+1:]) * es, nil
}

func mjSlice(v mjVal, axis, start, n int64) (mjVal, error) {
	ax, err := mjAxis(axis, len(v.shape))
	if err != nil {
		return mjVal{}, err
	}
	if start < 0 || n <= 0 || uint64(start+n) > v.shape[ax] {
		return mjVal{}, fmt.Errorf("slice [%d, %d) of an axis of %d", start, start+n, v.shape[ax])
	}
	outer, inner, err := mjSpan(v, ax)
	if err != nil {
		return mjVal{}, err
	}
	out := mjVal{shape: append([]uint64(nil), v.shape...), typ: v.typ}
	out.shape[ax] = uint64(n)
	if v.dry() {
		return out, nil
	}
	row := v.shape[ax] * inner
	for o := uint64(0); o < outer; o++ {
		out.data = append(out.data, v.data[o*row+uint64(start)*inner:o*row+uint64(start+n)*inner]...)
	}
	return out, nil
}

func mjConcat(vs []mjVal, axis int64) (mjVal, error) {
	if len(vs) == 0 {
		return mjVal{}, fmt.Errorf("concat of nothing")
	}
	ax, err := mjAxis(axis, len(vs[0].shape))
	if err != nil {
		return mjVal{}, err
	}
	out := mjVal{shape: append([]uint64(nil), vs[0].shape...), typ: vs[0].typ}
	out.shape[ax] = 0
	for _, v := range vs {
		if v.typ != out.typ || len(v.shape) != len(out.shape) || v.dry() != vs[0].dry() {
			return mjVal{}, fmt.Errorf("concat of %v %v and %v %v", vs[0].typ, vs[0].shape, v.typ, v.shape)
		}
		for i := range v.shape {
			if i != ax && v.shape[i] != vs[0].shape[i] {
				return mjVal{}, fmt.Errorf("concat on axis %d of %v and %v", ax, vs[0].shape, v.shape)
			}
		}
		out.shape[ax] += v.shape[ax]
	}
	if vs[0].dry() {
		return out, nil
	}
	outer, _, err := mjSpan(vs[0], ax)
	if err != nil {
		return mjVal{}, err
	}
	for o := uint64(0); o < outer; o++ {
		for _, v := range vs {
			_, inner, _ := mjSpan(v, ax)
			row := v.shape[ax] * inner
			out.data = append(out.data, v.data[o*row:(o+1)*row]...)
		}
	}
	return out, nil
}

func mjStack(vs []mjVal, axis int64) (mjVal, error) {
	if len(vs) == 0 {
		return mjVal{}, fmt.Errorf("stack of nothing")
	}
	ax, err := mjAxis(axis, len(vs[0].shape)+1)
	if err != nil {
		return mjVal{}, err
	}
	in := make([]mjVal, len(vs))
	for i, v := range vs {
		if len(v.shape) != len(vs[0].shape) || mjProd(v.shape) != mjProd(vs[0].shape) {
			return mjVal{}, fmt.Errorf("stack of %v and %v", vs[0].shape, v.shape)
		}
		in[i] = v
		in[i].shape = append(append(append([]uint64(nil), v.shape[:ax]...), 1), v.shape[ax:]...)
	}
	return mjConcat(in, int64(ax))
}

func mjRepeat(v mjVal, axis, times int64) (mjVal, error) {
	ax, err := mjAxis(axis, len(v.shape))
	if err != nil {
		return mjVal{}, err
	}
	if times <= 0 {
		return mjVal{}, fmt.Errorf("repeat %d times", times)
	}
	outer, inner, err := mjSpan(v, ax)
	if err != nil {
		return mjVal{}, err
	}
	out := mjVal{shape: append([]uint64(nil), v.shape...), typ: v.typ}
	out.shape[ax] *= uint64(times)
	if v.dry() {
		return out, nil
	}
	for o := uint64(0); o < outer; o++ {
		for j := uint64(0); j < v.shape[ax]; j++ {
			at := (o*v.shape[ax] + j) * inner
			for k := int64(0); k < times; k++ {
				out.data = append(out.data, v.data[at:at+inner]...)
			}
		}
	}
	return out, nil
}

func mjRetype(v mjVal, to string) (mjVal, error) {
	t := map[string]jlm.Type{"f32": jlm.TypeF32, "f16": jlm.TypeF16}[to]
	if _, err := mjElemSize(v.typ); err != nil {
		return mjVal{}, err
	}
	out := mjVal{shape: v.shape, typ: t}
	if v.dry() || v.typ == t {
		out.data = v.data
		return out, nil
	}
	vals := mjFloats(v)
	if t == jlm.TypeF32 {
		out.data = f32AsBytes(vals)
		return out, nil
	}
	out.data = make([]byte, 0, 2*len(vals))
	for _, f := range vals {
		out.data = binary.LittleEndian.AppendUint16(out.data, quant.EncodeHalf(f))
	}
	return out, nil
}

func mjFloats(v mjVal) []float32 {
	out := make([]float32, 0, v.count())
	if v.typ == jlm.TypeF16 {
		for i := 0; i+2 <= len(v.data); i += 2 {
			out = append(out, float32(quant.DecodeHalf(binary.LittleEndian.Uint16(v.data[i:]))))
		}
		return out
	}
	for i := 0; i+4 <= len(v.data); i += 4 {
		out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(v.data[i:])))
	}
	return out
}

// mjMapVar is a map's tensor by name, in hand order.
func mjMapVar(name string) (int, bool) {
	i := strings.Index("xyz", name)
	return i, len(name) == 1 && i >= 0
}

// mjMapFuncs are the functions a map may call.
func mjMapFuncs() map[string]func(float64) float64 {
	return map[string]func(float64) float64{"exp": math.Exp, "log": math.Log, "sqrt": math.Sqrt, "tanh": math.Tanh,
		"softplus": softplus}
}

// softplus is torch's: log(1 + e^x), and x itself past 20.
func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

// mjMapCheck refuses at lowering what a map cannot compute.
func mjMapCheck(n *exprNode) error {
	switch n.op {
	case "num", "path":
		return nil
	case "call":
		if _, ok := mjMapFuncs()[n.str]; !ok {
			return fmt.Errorf("%s is not a function a map calls", n.str)
		}
	case "neg", "+", "-", "*", "/", "??":
	default:
		return fmt.Errorf("%q is not arithmetic", n.op)
	}
	for _, a := range n.args {
		if err := mjMapCheck(a); err != nil {
			return err
		}
	}
	return nil
}

// mjCompile turns a map's expression into a function of its tensors' values,
// config paths read once. It computes in float64; the result is rounded to
// float32 once, which for + - * / is float32 arithmetic exactly.
func mjCompile(n *exprNode, env exprEnv, nsrc int) (func([]float64) float64, error) {
	switch n.op {
	case "num":
		v := n.num
		return func([]float64) float64 { return v }, nil
	case "path":
		if i, ok := mjMapVar(n.str); ok {
			if i >= nsrc {
				return nil, fmt.Errorf("%s names tensor %d of %d in hand", n.str, i+1, nsrc)
			}
			return func(xs []float64) float64 { return xs[i] }, nil
		}
		v, err := evalNum(n, env)
		if err != nil {
			return nil, err
		}
		return func([]float64) float64 { return v }, nil
	case "??":
		v, err := evalNum(n, env)
		if err != nil {
			return nil, err
		}
		return func([]float64) float64 { return v }, nil
	case "call":
		f := mjMapFuncs()[n.str]
		if f == nil || len(n.args) != 1 {
			return nil, fmt.Errorf("%s is not a one-argument function a map calls", n.str)
		}
		a, err := mjCompile(n.args[0], env, nsrc)
		if err != nil {
			return nil, err
		}
		return func(xs []float64) float64 { return f(a(xs)) }, nil
	case "neg":
		a, err := mjCompile(n.args[0], env, nsrc)
		if err != nil {
			return nil, err
		}
		return func(xs []float64) float64 { return -a(xs) }, nil
	case "+", "-", "*", "/":
		a, err := mjCompile(n.args[0], env, nsrc)
		if err != nil {
			return nil, err
		}
		b, err := mjCompile(n.args[1], env, nsrc)
		if err != nil {
			return nil, err
		}
		switch n.op {
		case "+":
			return func(xs []float64) float64 { return a(xs) + b(xs) }, nil
		case "-":
			return func(xs []float64) float64 { return a(xs) - b(xs) }, nil
		case "*":
			return func(xs []float64) float64 { return a(xs) * b(xs) }, nil
		}
		return func(xs []float64) float64 { return a(xs) / b(xs) }, nil
	}
	return nil, fmt.Errorf("%q is not arithmetic", n.op)
}

func mjMap(src string, env exprEnv, vs []mjVal) (mjVal, error) {
	n, err := parseExpr(src)
	if err != nil {
		return mjVal{}, err
	}
	f, err := mjCompile(n, env, len(vs))
	if err != nil {
		return mjVal{}, fmt.Errorf("map %q: %w", src, err)
	}
	for _, v := range vs {
		if _, err := mjElemSize(v.typ); err != nil {
			return mjVal{}, err
		}
		if v.count() != vs[0].count() || v.dry() != vs[0].dry() {
			return mjVal{}, fmt.Errorf("map %q over %v and %v", src, vs[0].shape, v.shape)
		}
	}
	out := mjVal{shape: vs[0].shape, typ: jlm.TypeF32}
	if vs[0].dry() {
		return out, nil
	}
	vals := make([][]float32, len(vs))
	for i, v := range vs {
		vals[i] = mjFloats(v)
	}
	xs := make([]float64, len(vs))
	out.data = make([]byte, 0, 4*vs[0].count())
	for j := range vals[0] {
		for i := range vals {
			xs[i] = float64(vals[i][j])
		}
		out.data = binary.LittleEndian.AppendUint32(out.data, math.Float32bits(float32(f(xs))))
	}
	return out, nil
}

// mjTorch is a container tensor's shape slowest axis first.
func mjTorch(e *jlm.Tensor) []uint64 {
	s := make([]uint64, e.NDim)
	for i := range s {
		s[i] = e.Dims[int(e.NDim)-1-i]
	}
	return s
}

// mjSetShape gives a container tensor a primitive's result's shape and type.
func mjSetShape(e *jlm.Tensor, v mjVal) error {
	if len(v.shape) == 0 || len(v.shape) > 4 {
		return fmt.Errorf("convert: %s: a transform's result has %d axes", e.Name, len(v.shape))
	}
	e.NDim, e.Dims, e.Type = uint8(len(v.shape)), [4]uint64{}, v.typ
	for i, d := range v.shape {
		e.Dims[len(v.shape)-1-i] = d
	}
	return nil
}

// mjState is what a lowered model.json's transforms read at conversion: the
// config.json the conversion opened, and each transformed tensor's source
// shape, recorded when its result's shape is stated.
type mjState struct {
	mu  sync.Mutex
	env exprEnv
	src map[string]mjVal
	act string // the activation the file states
}

// hfGatherStep is one model.json gather: the role it makes, its sources by
// block-relative name, and the primitives that make one tensor of them.
type hfGatherStep struct {
	role jlm.Role
	from []mjOperand
	do   mjPipe
}

// mjOperand is one of a gather's inputs: a source by its block-relative name,
// or one tensor composed of sources by primitives of its own.
type mjOperand struct {
	Name string   `json:"-"`
	From []string `json:"from"`
	Do   mjPipe   `json:"do"`
}

func (o *mjOperand) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		*o = mjOperand{Name: s}
		return nil
	}
	type plain mjOperand
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode((*plain)(o)); err != nil {
		return err
	}
	if len(o.From) == 0 || len(o.Do) == 0 {
		return fmt.Errorf("an operand composed of sources names them and its primitives")
	}
	return nil
}

// names are the sources an operand reads, in order.
func (o mjOperand) names() []string {
	if o.Name != "" {
		return []string{o.Name}
	}
	return o.From
}

// mjActivationCover holds the gathered tensors to the stated activation: one
// whose parameters are a per-block role (mjActivationParams) finds them in
// every block, and a block carrying another activation's parameters is
// refused, since the engine runs a block's stored parameters as its activation.
func mjActivationCover(act string, c *Config, out []jlm.Tensor) error {
	have := map[jlm.Role]map[int32]bool{}
	for _, r := range mjActivationParams() {
		have[r] = map[int32]bool{}
	}
	for _, t := range out {
		if h, ok := have[t.Role]; ok {
			h[t.Block] = true
		}
	}
	want := mjActivationParams()[act]
	for a, r := range mjActivationParams() {
		switch {
		case r == want:
			for b := int32(0); b < int32(c.NLayer); b++ {
				if !have[r][b] {
					return fmt.Errorf("convert: the activation is %s, and block %d carries no %v, its parameters", act, b, r)
				}
			}
		case len(have[r]) > 0:
			return fmt.Errorf("convert: the blocks carry %v, the parameters of activation %s, and the activation is %s", r, a, act)
		}
	}
	return nil
}

// gatherBlocks runs each step on every block that carries its first source.
// A source enters as the vector of its values. A source left over -- a second
// without its first -- is left to identify, which refuses it by name.
func gatherBlocks(env exprEnv, prefix string, steps []hfGatherStep, names []string,
	read func(string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error) {
	var out []jlm.Tensor
	used := map[string]bool{}
	for _, g := range steps {
		for _, n := range names {
			rest, ok := strings.CutPrefix(n, prefix)
			if !ok {
				continue
			}
			block, tail, err := leadingIndex(n, rest)
			if err != nil || tail != g.from[0].names()[0] {
				continue
			}
			bp := prefix + itoa(int(block)) + "."
			source := func(f string) (mjVal, error) {
				v, err := read(bp + f)
				if err != nil {
					return mjVal{}, err
				}
				used[bp+f] = true
				return mjVal{shape: []uint64{uint64(len(v))}, typ: jlm.TypeF32, data: f32AsBytes(v)}, nil
			}
			src := make([]mjVal, len(g.from))
			for i, o := range g.from {
				ins := make([]mjVal, len(o.names()))
				for j, f := range o.names() {
					if ins[j], err = source(f); err != nil {
						return nil, nil, err
					}
				}
				if o.Name != "" {
					src[i] = ins[0]
				} else if src[i], err = o.Do.run(env, ins); err != nil {
					return nil, nil, fmt.Errorf("convert: %s: operand %d: %w", n, i, err)
				}
			}
			v, err := g.do.run(env, src)
			if err != nil {
				return nil, nil, fmt.Errorf("convert: %s: %w", n, err)
			}
			e := jlm.Tensor{Role: g.role, Block: block, Index: -1, Name: "blk." + itoa(int(block)) + "." + g.role.String(), Data: v.data}
			if err := mjSetShape(&e, v); err != nil {
				return nil, nil, err
			}
			out = append(out, e)
		}
	}
	return out, used, nil
}
