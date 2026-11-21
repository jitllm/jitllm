package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
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

// mjF64 is a value a float64 gather computes (mjGather.Precision): a type no
// container stores, rounded once to f32 by a type step.
const mjF64 jlm.Type = 255

func mjElemSize(t jlm.Type) (uint64, error) {
	switch t {
	case mjF64:
		return 8, nil
	case jlm.TypeF32:
		return 4, nil
	case jlm.TypeF16, jlm.TypeBF16:
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
	// Trim is [axis, keep]: the first keep indices along the axis, the rest
	// refused unless every byte of it is zero (a release's padding).
	Trim []mjSize `json:"trim"`
	// Type is the storage type: "f32" or "f16".
	Type string `json:"type"`
	// Pick is [axis, start, length, start, length, ...]: those ranges of the
	// axis joined in order.
	Pick []mjSize `json:"pick"`
	// TrimZero, true, cuts the slowest axis's trailing indices whose bytes
	// are all zero, one at least kept: a converter's padding, of any storage
	// type, sized by the bytes it reads.
	TrimZero bool `json:"trim_zero"`
	// Shape is the shape the tensor in hand has, slowest axis first, -1 any
	// size; another is refused. No byte moves.
	Shape []mjSize `json:"shape"`
	// RMS is an epsilon, an expression rounded to f32 as the container's
	// epsilons are: each row (the fastest axis) over its root mean square,
	// x / sqrt(mean(x^2) + eps), accumulated in float64 and rounded once to
	// float32. With a second tensor in hand, a norm's weight, each row is
	// the weight's length and is multiplied by it, x * (w / sqrt(...)).
	RMS string `json:"rms"`
	// Pad is [axis, multiple]: the axis extended with zero indices to a
	// whole multiple.
	Pad []mjSize `json:"pad"`
	// Linear, true, takes the two tensors in hand through a linear layer:
	// the first, [.., k], by the second, its weight as torch's [m][k], to
	// [.., m], each sum accumulated in float64 in k's order.
	Linear bool `json:"linear"`
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
	if o.Trim != nil {
		set = append(set, "trim")
	}
	if o.Type != "" {
		set = append(set, "type")
	}
	if o.TrimZero {
		set = append(set, "trim_zero")
	}
	if o.Pick != nil {
		set = append(set, "pick")
	}
	if o.Shape != nil {
		set = append(set, "shape")
	}
	if o.RMS != "" {
		set = append(set, "rms")
	}
	if o.Pad != nil {
		set = append(set, "pad")
	}
	if o.Linear {
		set = append(set, "linear")
	}
	if len(set) != 1 {
		return strings.Join(set, "+")
	}
	return set[0]
}

// mjPipe is a tensor's primitives in order.
type mjPipe []mjOp

// reshapes says p is reshapes alone, which move no byte.
func (p mjPipe) reshapes() bool {
	for _, o := range p {
		if o.name() != "reshape" {
			return false
		}
	}
	return true
}

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
		case "slice", "repeat", "trim", "pad":
			if want := map[string]int{"slice": 3, "repeat": 2, "trim": 2, "pad": 2}[n]; len(argsOf(o)) != want {
				return fmt.Errorf("%s takes %d values, not %d", n, want, len(argsOf(o)))
			}
		case "type":
			if o.Type != "f32" && o.Type != "f16" && o.Type != "q8" {
				return fmt.Errorf("type %q is not f32, f16 or q8", o.Type)
			}
		case "pick":
			if n := len(o.Pick); n < 3 || n%2 != 1 {
				return fmt.Errorf("pick takes an axis and pairs of start and length, not %d values", n)
			}
		case "rms":
			if _, err := parseExpr(o.RMS); err != nil {
				return fmt.Errorf("rms %q: %w", o.RMS, err)
			}
		case "reshape", "transpose", "concat", "stack", "trim_zero", "shape", "linear":
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
	case o.Trim != nil:
		return o.Trim
	case o.Pick != nil:
		return o.Pick
	case o.Shape != nil:
		return o.Shape
	case o.Pad != nil:
		return o.Pad
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
		case "rms":
			out, err = mjRMS(vs, o.RMS, env)
		case "linear":
			out, err = mjLinear(vs)
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
			case "trim":
				out, err = mjTrim(vs[0], args[0], args[1])
			case "type":
				out, err = mjRetype(vs[0], o.Type)
			case "trim_zero":
				out, err = mjTrimZero(vs[0])
			case "pick":
				out, err = mjPick(vs[0], args)
			case "shape":
				out, err = mjShape(vs[0], args)
			case "pad":
				out, err = mjPad(vs[0], args[0], args[1])
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

func mjRMS(vs []mjVal, src string, env exprEnv) (mjVal, error) {
	if len(vs) != 1 && len(vs) != 2 {
		return mjVal{}, fmt.Errorf("rms takes a tensor and at most its weight, and %d are in hand", len(vs))
	}
	x, err := parseExpr(src)
	if err != nil {
		return mjVal{}, err
	}
	e, err := evalNum(x, env)
	if err != nil {
		return mjVal{}, fmt.Errorf("rms %q: %w", src, err)
	}
	for _, v := range vs {
		if _, err := mjElemSize(v.typ); err != nil {
			return mjVal{}, fmt.Errorf("rms of a %v tensor: take it to f32 first", v.typ)
		}
	}
	v := vs[0]
	if len(v.shape) == 0 || v.shape[len(v.shape)-1] == 0 {
		return mjVal{}, fmt.Errorf("rms of a %v tensor", v.shape)
	}
	n := v.shape[len(v.shape)-1]
	if len(vs) == 2 {
		if n = vs[1].count(); n == 0 || v.count()%n != 0 || vs[1].dry() != v.dry() {
			return mjVal{}, fmt.Errorf("rms of a %v tensor by a weight of %v", v.shape, vs[1].shape)
		}
	}
	out := mjVal{shape: v.shape, typ: mjPrecision(v)}
	if v.dry() {
		return out, nil
	}
	vals := mjFloats64(v)
	var w []float64
	if len(vs) == 2 {
		w = mjFloats64(vs[1])
	}
	eps := float64(float32(e))
	for r := 0; r+int(n) <= len(vals); r += int(n) {
		row := vals[r : r+int(n)]
		var ss float64
		for _, y := range row {
			ss += y * y
		}
		k := 1 / math.Sqrt(ss/float64(n)+eps)
		for j := range row {
			if w != nil {
				row[j] *= k * w[j]
			} else {
				row[j] *= k
			}
		}
	}
	out.data = mjStore(vals, out.typ)
	return out, nil
}

// mjPrecision is the type an arithmetic primitive's result has: float64 where
// one of its tensors is, float32 otherwise.
func mjPrecision(vs ...mjVal) jlm.Type {
	for _, v := range vs {
		if v.typ == mjF64 {
			return mjF64
		}
	}
	return jlm.TypeF32
}

// mjStore is values as the bytes of typ, float32 rounded once.
func mjStore(vals []float64, typ jlm.Type) []byte {
	if typ == mjF64 {
		b := make([]byte, 0, 8*len(vals))
		for _, x := range vals {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(x))
		}
		return b
	}
	b := make([]byte, 0, 4*len(vals))
	for _, x := range vals {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(float32(x)))
	}
	return b
}

// mjLinear is the first tensor in hand, [.., k], through the second, a linear
// layer's weight of m rows of k, as [.., m]: each sum in float64, in k's order.
func mjLinear(vs []mjVal) (mjVal, error) {
	if len(vs) != 2 {
		return mjVal{}, fmt.Errorf("linear takes a tensor and its weight, and %d are in hand", len(vs))
	}
	x, w := vs[0], vs[1]
	for _, v := range vs {
		if _, err := mjElemSize(v.typ); err != nil {
			return mjVal{}, fmt.Errorf("linear of a %v tensor: take it to f32 first", v.typ)
		}
	}
	if len(x.shape) == 0 || x.shape[len(x.shape)-1] == 0 || w.count()%x.shape[len(x.shape)-1] != 0 || x.dry() != w.dry() {
		return mjVal{}, fmt.Errorf("linear of a %v tensor by a weight of %v", x.shape, w.shape)
	}
	k := x.shape[len(x.shape)-1]
	m, n := w.count()/k, x.count()/k
	if n*m > 1<<26 {
		return mjVal{}, fmt.Errorf("linear of a %v tensor by a weight of %v is past %d values", x.shape, w.shape, 1<<26)
	}
	out := mjVal{shape: append(slices.Clone(x.shape[:len(x.shape)-1]), m), typ: mjPrecision(x, w)}
	if x.dry() {
		return out, nil
	}
	xv, wv := mjFloats64(x), mjFloats64(w)
	res := make([]float64, n*m)
	for i := range n {
		row := xv[i*k : (i+1)*k]
		for r := range m {
			wr := wv[r*k : (r+1)*k]
			acc := 0.0
			for j := range row {
				acc += wr[j] * row[j]
			}
			res[i*m+r] = acc
		}
	}
	out.data = mjStore(res, out.typ)
	return out, nil
}

// mjPad extends axis with zero indices to a whole multiple of m.
func mjPad(v mjVal, axis, m int64) (mjVal, error) {
	ax, err := mjAxis(axis, len(v.shape))
	if err != nil {
		return mjVal{}, err
	}
	es, err := mjElemSize(v.typ)
	if err != nil {
		return mjVal{}, err
	}
	if m <= 0 || m > 1<<16 {
		return mjVal{}, fmt.Errorf("pad to a multiple of %d", m)
	}
	n := v.shape[ax]
	p := (n + uint64(m) - 1) / uint64(m) * uint64(m)
	out := mjVal{shape: slices.Clone(v.shape), typ: v.typ}
	out.shape[ax] = p
	if v.dry() || p == n {
		out.data = v.data
		return out, nil
	}
	outer, inner := mjProd(v.shape[:ax]), mjProd(v.shape[ax+1:])*es
	out.data = make([]byte, outer*p*inner)
	for o := range outer {
		copy(out.data[o*p*inner:], v.data[o*n*inner:(o+1)*n*inner])
	}
	return out, nil
}

func mjShape(v mjVal, sizes []int64) (mjVal, error) {
	ok := len(sizes) == len(v.shape)
	for i := 0; ok && i < len(sizes); i++ {
		ok = sizes[i] == -1 || sizes[i] >= 0 && uint64(sizes[i]) == v.shape[i]
	}
	if !ok {
		return mjVal{}, fmt.Errorf("shape %v of a %v tensor", sizes, v.shape)
	}
	return v, nil
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
	if _, err := mjElemSize(v.typ); err != nil && len(v.shape) > 0 && shape[len(shape)-1] != v.shape[len(v.shape)-1] {
		return mjVal{}, fmt.Errorf("reshape %v of a %v tensor %v: its rows are its blocks, and keep their width", sizes, v.typ, v.shape)
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
	axes := make([]int64, r)
	for i, a := range perm {
		ax, err := mjAxis(a, r)
		if err != nil || seen[ax] {
			return mjVal{}, fmt.Errorf("transpose %v is not a permutation of %d axes", perm, r)
		}
		seen[ax], shape[i], axes[i] = true, v.shape[ax], int64(ax)
	}
	out := mjVal{shape: shape, typ: v.typ}
	dims := v.shape
	es, err := mjElemSize(v.typ)
	if err != nil {
		// A packed type moves by whole rows of its blocks: its fastest axis
		// stays, and a row is the element the others permute.
		last := r - 1
		if axes[last] != int64(last) {
			return mjVal{}, fmt.Errorf("a %v tensor moves by whole rows of its blocks, and transpose %v moves its last axis", v.typ, perm)
		}
		if v.dry() {
			return out, nil
		}
		rows := mjProd(v.shape[:last])
		if rows == 0 || uint64(len(v.data))%rows != 0 {
			return mjVal{}, fmt.Errorf("a %v tensor %v of %d bytes is not whole rows", v.typ, v.shape, len(v.data))
		}
		es, dims, axes, shape, r = uint64(len(v.data))/rows, v.shape[:last], axes[:last], shape[:last], last
	}
	if v.dry() {
		return out, nil
	}
	perm = axes
	v = mjVal{shape: dims, typ: v.typ, data: v.data}
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

// mjSpan is v's slabs along axis ax: how many lie outside it, and the bytes
// of each index. A packed type's fastest axis is its blocks, so it is spanned
// only above that axis, a row's bytes read off the data.
func mjSpan(v mjVal, ax int) (outer, inner uint64, err error) {
	if es, err := mjElemSize(v.typ); err == nil {
		return mjProd(v.shape[:ax]), mjProd(v.shape[ax+1:]) * es, nil
	}
	last := len(v.shape) - 1
	rows := mjProd(v.shape[:last])
	if ax >= last || v.dry() || rows == 0 || uint64(len(v.data))%rows != 0 {
		return 0, 0, fmt.Errorf("a %v tensor %v moves by whole rows of its blocks, not along axis %d", v.typ, v.shape, ax)
	}
	return mjProd(v.shape[:ax]), mjProd(v.shape[ax+1:last]) * (uint64(len(v.data)) / rows), nil
}

func mjSlice(v mjVal, axis, start, n int64) (mjVal, error) {
	ax, err := mjAxis(axis, len(v.shape))
	if err != nil {
		return mjVal{}, err
	}
	if start < 0 || n <= 0 || uint64(start+n) > v.shape[ax] {
		return mjVal{}, fmt.Errorf("slice [%d, %d) of an axis of %d", start, start+n, v.shape[ax])
	}
	out := mjVal{shape: append([]uint64(nil), v.shape...), typ: v.typ}
	out.shape[ax] = uint64(n)
	if v.dry() {
		return out, nil
	}
	outer, inner, err := mjSpan(v, ax)
	if err != nil {
		return mjVal{}, err
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

// mjTrim is v's first keep indices along axis, refused where what it drops
// holds any byte but zero.
func mjTrim(v mjVal, axis, keep int64) (mjVal, error) {
	ax, err := mjAxis(axis, len(v.shape))
	if err != nil {
		return mjVal{}, err
	}
	if keep <= 0 || uint64(keep) > v.shape[ax] {
		return mjVal{}, fmt.Errorf("trim to %d of an axis of %d", keep, v.shape[ax])
	}
	if uint64(keep) == v.shape[ax] {
		return v, nil
	}
	outer, inner, err := mjSpan(v, ax)
	if err != nil {
		return mjVal{}, err
	}
	if !v.dry() {
		row := v.shape[ax] * inner
		for o := uint64(0); o < outer; o++ {
			for _, b := range v.data[o*row+uint64(keep)*inner : (o+1)*row] {
				if b != 0 {
					return mjVal{}, fmt.Errorf("trim to %d of %d: the rest is not zero padding", keep, v.shape[ax])
				}
			}
		}
	}
	return mjSlice(v, axis, 0, keep)
}

// mjPick is v's ranges along an axis, args[0], each a start and a length
// after it, joined in order.
func mjPick(v mjVal, args []int64) (mjVal, error) {
	var parts []mjVal
	for i := 1; i+1 < len(args); i += 2 {
		p, err := mjSlice(v, args[0], args[i], args[i+1])
		if err != nil {
			return mjVal{}, err
		}
		parts = append(parts, p)
	}
	return mjConcat(parts, args[0])
}

// mjTrimZero is v without the trailing indices of its slowest axis whose
// bytes are all zero, one at least kept; a slab is v's bytes over the axis,
// so any storage type trims.
func mjTrimZero(v mjVal) (mjVal, error) {
	if v.dry() {
		return mjVal{}, fmt.Errorf("trim_zero is sized by the bytes it reads, and a dry value has none")
	}
	if len(v.shape) == 0 || v.shape[0] == 0 || uint64(len(v.data))%v.shape[0] != 0 {
		return mjVal{}, fmt.Errorf("trim_zero of %d bytes over a shape of %v", len(v.data), v.shape)
	}
	slab := uint64(len(v.data)) / v.shape[0]
	keep := v.shape[0]
	for keep > 1 && !slices.ContainsFunc(v.data[(keep-1)*slab:keep*slab], func(b byte) bool { return b != 0 }) {
		keep--
	}
	out := mjVal{shape: slices.Clone(v.shape), typ: v.typ, data: v.data[:keep*slab]}
	out.shape[0] = keep
	return out, nil
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

// mjRetype stores v as to: f32 or f16 of a float tensor; f32 of a packed one,
// dequantized; q8 of a float one, its rows (the fastest axis) quantized as
// Q8_0, whole blocks of it.
func mjRetype(v mjVal, to string) (mjVal, error) {
	if _, err := mjElemSize(v.typ); err != nil {
		src, ok := sourceType(v.typ)
		if to != "f32" || !ok {
			return mjVal{}, fmt.Errorf("a %v tensor is dequantized to f32, not stored as %s", v.typ, to)
		}
		out := mjVal{shape: v.shape, typ: jlm.TypeF32}
		if v.dry() {
			return out, nil
		}
		vals := make([]float32, v.count())
		if err := quant.Dequant32(src, v.data, vals); err != nil {
			return mjVal{}, err
		}
		out.data = f32AsBytes(vals)
		return out, nil
	}
	if to == "q8" {
		k := uint64(1)
		if len(v.shape) > 0 {
			k = v.shape[len(v.shape)-1]
		}
		if k%q8Elems != 0 {
			return mjVal{}, fmt.Errorf("rows of %d are not whole Q8_0 blocks", k)
		}
		out := mjVal{shape: v.shape, typ: jlm.TypeQ8}
		if !v.dry() {
			out.data = quantizeQ8Rows(mjFloats(v), int(k), int(v.count()/k))
		}
		return out, nil
	}
	t := map[string]jlm.Type{"f32": jlm.TypeF32, "f16": jlm.TypeF16}[to]
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
	if v.typ == mjF64 {
		for _, x := range mjFloats64(v) {
			out = append(out, float32(x))
		}
		return out
	}
	if v.typ == jlm.TypeF16 {
		for i := 0; i+2 <= len(v.data); i += 2 {
			out = append(out, float32(quant.DecodeHalf(binary.LittleEndian.Uint16(v.data[i:]))))
		}
		return out
	}
	if v.typ == jlm.TypeBF16 {
		for i := 0; i+2 <= len(v.data); i += 2 {
			out = append(out, math.Float32frombits(uint32(binary.LittleEndian.Uint16(v.data[i:]))<<16))
		}
		return out
	}
	for i := 0; i+4 <= len(v.data); i += 4 {
		out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(v.data[i:])))
	}
	return out
}

// mjFloats64 is v's values in float64, a float32 one's exactly.
func mjFloats64(v mjVal) []float64 {
	if v.typ != mjF64 {
		f := mjFloats(v)
		out := make([]float64, len(f))
		for i, x := range f {
			out[i] = float64(x)
		}
		return out
	}
	out := make([]float64, 0, v.count())
	for i := 0; i+8 <= len(v.data); i += 8 {
		out = append(out, math.Float64frombits(binary.LittleEndian.Uint64(v.data[i:])))
	}
	return out
}

// mjMapVar is a map's tensor by name, in hand order.
func mjMapVar(name string) (int, bool) {
	i := strings.Index("xyz", name)
	return i, len(name) == 1 && i >= 0
}

// mjReadsTensor reports whether n reads one of a map's tensors.
func mjReadsTensor(n *exprNode) bool {
	if n.op == "path" {
		_, ok := mjMapVar(n.str)
		return ok
	}
	return slices.ContainsFunc(n.args, mjReadsTensor)
}

// mjMapFuncs are the functions a map may call.
func mjMapFuncs() map[string]func(float64) float64 {
	return map[string]func(float64) float64{"exp": math.Exp, "log": math.Log, "sqrt": math.Sqrt, "tanh": math.Tanh,
		"softplus": softplus, "f32": func(x float64) float64 { return float64(float32(x)) }}
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
		// A ?? reads the config: a default for a tensor's element is not one.
		if mjReadsTensor(n) {
			return nil, fmt.Errorf("a ?? reads the config, and %q reads a tensor", exprText(n))
		}
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
	out := mjVal{shape: vs[0].shape, typ: mjPrecision(vs...)}
	if vs[0].dry() {
		return out, nil
	}
	vals := make([][]float64, len(vs))
	for i, v := range vs {
		vals[i] = mjFloats64(v)
	}
	xs := make([]float64, len(vs))
	res := make([]float64, len(vals[0]))
	for j := range vals[0] {
		for i := range vals {
			xs[i] = vals[i][j]
		}
		res[j] = f(xs)
	}
	out.data = mjStore(res, out.typ)
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
	if v.typ == mjF64 {
		return fmt.Errorf("convert: %s: a float64 result is a computation's, and a type step stores it", e.Name)
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

	ternary ternaryForm // how the config says the ternary projections are stored
}

// hfGatherStep is one model.json gather: the role it makes, its sources by
// block-relative name, and the primitives that make one tensor of them.
type hfGatherStep struct {
	role         jlm.Role
	from         []mjOperand
	do           mjPipe
	cond, count  *exprNode
	every, model bool
	name         string
	f64          bool
}

// mjOperand is one of a gather's inputs: a source by its block-relative name,
// one tensor composed of sources by primitives of its own (one source alone
// needs none), or Fill's value at Shape, each a count over the keys. Keep
// reads its sources and leaves them in place, where a gather otherwise
// consumes what it reads.
type mjOperand struct {
	Name  string   `json:"-"`
	From  []string `json:"from"`
	Do    mjPipe   `json:"do"`
	Fill  *float64 `json:"fill"`
	Shape []string `json:"shape"`
	Keep  bool     `json:"keep"`
}

func (o *mjOperand) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		*o = mjOperand{Name: s}
		return nil
	}
	*o = mjOperand{}
	type plain mjOperand
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode((*plain)(o)); err != nil {
		return err
	}
	if o.Fill != nil {
		if len(o.From)+len(o.Do) > 0 || len(o.Shape) == 0 {
			return fmt.Errorf("an operand filled with a value states its shape and reads no source")
		}
		return nil
	}
	if len(o.Shape) > 0 {
		return fmt.Errorf("a shape is a filled operand's")
	}
	if len(o.From) == 0 || len(o.Do) == 0 && len(o.From) != 1 {
		return fmt.Errorf("an operand composed of sources names them and its primitives")
	}
	return nil
}

// filled is a filled operand's tensor, its shape's counts read off env and
// bounded, since a key sizes it (RULE 9).
func (o mjOperand) filled(env exprEnv) (mjVal, error) {
	v := mjVal{typ: jlm.TypeF32}
	n := uint64(1)
	for _, s := range o.Shape {
		x, err := parseExpr(s)
		if err != nil {
			return mjVal{}, err
		}
		c, err := evalNum(x, env)
		if err != nil {
			return mjVal{}, err
		}
		d, err := exprUint(s, c)
		if err != nil || d == 0 {
			return mjVal{}, fmt.Errorf("a filled operand's %s is %v, not a count", s, c)
		}
		if n *= d; n > 1<<24 {
			return mjVal{}, fmt.Errorf("a filled operand of shape %v is past %d values", o.Shape, 1<<24)
		}
		v.shape = append(v.shape, d)
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(*o.Fill)
	}
	v.data = f32AsBytes(vals)
	return v, nil
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

// gatherBlocks runs each step on every block that carries its first source,
// or, where the step says every, on each of the model's nlayer blocks; a
// model step once, on sources named whole. Consecutive every steps run block
// by block, each block's in the statement's order. A step whose condition
// does not hold runs nowhere. A source enters as the vector of its values. A
// source left over -- a second without its first, a step that did not run --
// is left to identify, which refuses it by name.
func gatherBlocks(env exprEnv, prefix string, steps []hfGatherStep, names []string,
	read func(string) ([]float32, error), shape func(string) []uint64, nlayer int) ([]jlm.Tensor, map[string]bool, error) {
	var out []jlm.Tensor
	used := map[string]bool{}
	have := map[string]bool{}
	for _, n := range names {
		have[n] = true
	}
	// prep is step g run on sources under bp (the block's prefix, or none for
	// the model's), false where its condition does not hold.
	prep := func(g hfGatherStep) (func(bp string, block int32) error, bool, error) {
		if g.cond != nil {
			v, err := evalMJ(g.cond, env)
			if err != nil {
				return nil, false, fmt.Errorf("convert: the gather of %v: %w", g.role, err)
			}
			on, ok := v.(bool)
			if !ok {
				return nil, false, fmt.Errorf("convert: the gather of %v: its if is %s, not a boolean", g.role, exprKind(v))
			}
			if !on {
				return nil, false, nil
			}
		}
		count := -1
		if g.count != nil {
			v, err := evalMJ(g.count, env)
			if err != nil {
				return nil, false, fmt.Errorf("convert: the gather of %v: %w", g.role, err)
			}
			n, err := exprUint("count", v)
			if err != nil {
				return nil, false, fmt.Errorf("convert: the gather of %v: %w", g.role, err)
			}
			count = int(n)
		}
		at := func(bp string, block int32) error {
			source := func(f string, keep bool) (mjVal, error) {
				if !have[bp+f] {
					return mjVal{}, fmt.Errorf("convert: the gather of %v reads %s, which the checkpoint does not carry", g.role, bp+f)
				}
				v, err := read(bp + f)
				if err != nil {
					return mjVal{}, err
				}
				if count >= 0 && len(v) != count {
					return mjVal{}, fmt.Errorf("convert: %s holds %d values, want %d", bp+f, len(v), count)
				}
				if !keep {
					used[bp+f] = true
				}
				if g.f64 {
					w := make([]float64, len(v))
					for i, x := range v {
						w[i] = float64(x)
					}
					return mjVal{shape: []uint64{uint64(len(v))}, typ: mjF64, data: mjStore(w, mjF64)}, nil
				}
				return mjVal{shape: []uint64{uint64(len(v))}, typ: jlm.TypeF32, data: f32AsBytes(v)}, nil
			}
			src := make([]mjVal, len(g.from))
			first := g.role.String()
			env := env
			if ns := g.from[0].names(); len(ns) > 0 {
				first = bp + ns[0]
				if shape != nil {
					if s := shape(first); s != nil {
						dims := make([]any, len(s))
						for i, d := range s {
							dims[i] = float64(d)
						}
						outer := env
						env = func(p string) any {
							if p == "shape" {
								return dims
							}
							return outer(p)
						}
					}
				}
			}
			for i, o := range g.from {
				if o.Fill != nil {
					var err error
					if src[i], err = o.filled(env); err != nil {
						return fmt.Errorf("convert: the gather of %v: operand %d: %w", g.role, i, err)
					}
					continue
				}
				ins := make([]mjVal, len(o.names()))
				for j, f := range o.names() {
					var err error
					if ins[j], err = source(f, o.Keep); err != nil {
						return err
					}
				}
				if o.Name != "" {
					src[i] = ins[0]
					continue
				}
				var err error
				if src[i], err = o.Do.run(env, ins); err != nil {
					return fmt.Errorf("convert: %s: operand %d: %w", bp+o.names()[0], i, err)
				}
			}
			v, err := g.do.run(env, src)
			if err != nil {
				return fmt.Errorf("convert: %s: %w", first, err)
			}
			name := "blk." + itoa(int(block)) + "." + g.role.String()
			if g.model {
				name = g.role.String()
			}
			if g.name != "" {
				name = strings.ReplaceAll(g.name, "{block}", bp)
			}
			e := jlm.Tensor{Role: g.role, Block: block, Index: -1, Name: name, Data: v.data}
			if err := mjSetShape(&e, v); err != nil {
				return err
			}
			out = append(out, e)
			return nil
		}
		return at, true, nil
	}
	var every []func(string, int32) error
	flush := func() error {
		for b := range nlayer {
			for _, at := range every {
				if err := at(prefix+itoa(b)+".", int32(b)); err != nil {
					return err
				}
			}
		}
		every = every[:0]
		return nil
	}
	for _, g := range steps {
		at, on, err := prep(g)
		if err != nil {
			return nil, nil, err
		}
		if !on {
			continue
		}
		if g.every {
			every = append(every, at)
			continue
		}
		if err := flush(); err != nil {
			return nil, nil, err
		}
		if g.model {
			if err := at("", jlm.DenseBlock); err != nil {
				return nil, nil, err
			}
			continue
		}
		for _, n := range names {
			rest, ok := strings.CutPrefix(n, prefix)
			if !ok {
				continue
			}
			block, tail, err := leadingIndex(n, rest)
			if err != nil || tail != g.from[0].names()[0] {
				continue
			}
			if err := at(prefix+itoa(int(block))+".", block); err != nil {
				return nil, nil, err
			}
		}
	}
	if err := flush(); err != nil {
		return nil, nil, err
	}
	return out, used, nil
}
