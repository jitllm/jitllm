package model

import (
	"cmp"
	"crypto/sha256"
	"flag"
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/paritytest/maintier"
)

var (
	parityProfiles = flag.String("parity-profiles", "cuda-sm70",
		"the recorder profiles TestComposerParity compares on")
	parityModel  = flag.String("parity-model", "", "TestComposerParity on this fixture alone, printing what it misses")
	parityKernel = flag.String("parity-kernel", "", "with -parity-model, the main kernel name prefix whose missing instructions to print")
)

// TestComposerParity holds what the composer emits to what main emits, with
// no GPU: every fixture opened twice on a device that runs nothing and
// records what is compiled for it (recorder) -- once on main's tier
// (paritytest/maintier, main's own code), once on the composer's -- a prompt
// chunk and a decode step asked of each, and every instruction of main's
// kernels looked for among the composer's formed kernels by its dataflow:
// the instruction and, recursively, everything it reads, so the order of
// independent instructions does not count. It reports per profile the share
// of main's instructions matched, the models whose every instruction is, and
// main's kernels by instructions missing. A measurement, not a gate.
func TestComposerParity(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	if len(paths) == 0 {
		t.Fatalf("no synth-*.jlm under %s (set JITLLM_MODELS)", testmodels.Dir())
	}
	for _, pn := range strings.Split(*parityProfiles, ",") {
		prof, ok := recorder.Profiles[pn]
		if !ok {
			t.Fatalf("no recorder profile %q", pn)
		}
		var total, same, models, identical int
		missing := map[string]int{}
		for _, p := range paths {
			name := strings.TrimSuffix(filepath.Base(p), ".jlm")
			if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
				continue
			}
			if *parityModel != "" && name != *parityModel {
				continue
			}
			mk, err := parityKernels(t, prof, p, true)
			if err != nil {
				t.Logf("%s on %s, main: %v", name, pn, err)
			}
			if len(mk) == 0 {
				continue
			}
			ck, err := parityKernels(t, prof, p, false)
			if err != nil {
				t.Logf("%s on %s, composer: %v", name, pn, err)
			}
			in := interner{}
			have := map[[16]byte]int{}
			for _, k := range ck {
				if strings.HasPrefix(k.Comment, "composed") {
					for _, h := range dataflow(k, in) {
						have[h]++
					}
				}
			}
			mt, ms := 0, 0
			for _, k := range mk {
				prefix, _, _ := strings.Cut(k.Name, "_")
				for i, h := range dataflow(k, in) {
					if k.Ops[i].Kind == ir.OpParam {
						continue
					}
					mt++
					if have[h] > 0 {
						have[h]--
						ms++
						continue
					}
					missing[prefix]++
					if *parityModel != "" && strings.HasPrefix(k.Name, *parityKernel) && *parityKernel != "" {
						t.Logf("%s missing %%%d: %s\n        %s", k.Name, i, opLine(k, i), forms(k)[i])
					}
				}
			}
			if *parityModel != "" && *parityKernel != "" {
				for _, k := range ck {
					if strings.HasPrefix(k.Comment, "composed") {
						for i, f := range forms(k) {
							t.Logf("composed %%%d: %s", i, f)
						}
					}
				}
			}
			total, same, models = total+mt, same+ms, models+1
			if ms == mt {
				identical++
			}
		}
		t.Logf("%-16s main's instructions %d, the composer's dataflow-identical %d (%.1f%%), models identical %d/%d",
			pn, total, same, 100*float64(same)/float64(max(total, 1)), identical, models)
		var names []string
		for n := range missing {
			names = append(names, n)
		}
		slices.SortFunc(names, func(a, b string) int { return cmp.Compare(missing[b], missing[a]) })
		for _, n := range names[:min(10, len(names))] {
			t.Logf("    %-22s %9d missing", n, missing[n])
		}
	}
}

// parityKernels is every kernel compiled for the model at path on a
// recording device standing in for prof, on main's tier or the composer's,
// over a prompt chunk and a decode step: main's launched, the composer's
// formed. Launches run nothing.
func parityKernels(t *testing.T, prof recorder.Profile, path string, main bool) ([]*ir.Kernel, error) {
	m, err := Open(path, noTune)
	if err != nil {
		return nil, err
	}
	defer m.Close()
	if m.IsEncoder() {
		return nil, nil
	}
	rec := recorder.New(prof, 1<<40)
	var g nn.Device
	if main {
		mg, err := maintier.New([]maintier.Slot{{Dev: rec, Bytes: 1 << 40}},
			maintier.WithConfig(func(c *maintier.Config) { c.NoGraph = true }))
		if err != nil {
			return nil, err
		}
		defer mg.Close()
		g = mg
	} else {
		cg, err := tier.New([]tier.Slot{{Dev: rec, Bytes: 1 << 40}}, append(testTierOpts(t),
			tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))...)
		if err != nil {
			return nil, err
		}
		defer cg.Close()
		g = cg
	}
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		return nil, err
	}
	lg, err := s.Prefill(m.Vocab.Encode("Hello there", true))
	tok := int32(1)
	if err == nil {
		tok = Greedy(lg)
	}
	if _, ferr := s.Forward(tok); err == nil {
		err = ferr
	}
	// Main's kernels are what it launched (its tuners compile alternatives
	// they never run); the composer's what it formed, launched or not (a
	// step stops at the first op it cannot form).
	var ks []*ir.Kernel
	if main {
		for _, e := range rec.Launched() {
			ks = append(ks, e.K)
		}
		return ks, err
	}
	for _, e := range rec.Compiled(func(k *ir.Kernel) bool { return strings.HasPrefix(k.Comment, "composed") }) {
		ks = append(ks, e.K)
	}
	return ks, err
}

// dataflow is a digest of each of k's instructions by what it computes:
// its kind, type, tile and loop depth, and the digests of what it reads,
// with integer arithmetic canonical -- an index is an affine form over the
// thread's id, its workgroup's, the group width and each loop's iteration
// (a phi stepped by a constant is its start plus the step times the
// iteration), a load's or store's immediate folded into its address -- so
// two kernels agree when they read the same elements, compute the same
// floats and write the same places, whatever index arithmetic got them
// there and in whatever order. A float loop-carried value enters by its
// kind alone. A parameter is its element type.
func dataflow(k *ir.Kernel, in interner) [][16]byte {
	c := newCanon(k, in)
	out := make([][16]byte, len(k.Ops))
	for i := range k.Ops {
		copy(out[i][:], hashOf(c.form(i)))
	}
	return out
}

// affine is c + sum coef*atom over atoms named by digests (a product of
// atoms is one atom).
type affine struct {
	c     int64
	terms map[string]int64
}

func (a affine) add(b affine, sign int64) affine {
	o := affine{c: a.c + sign*b.c, terms: map[string]int64{}}
	for t, v := range a.terms {
		o.terms[t] += v
	}
	for t, v := range b.terms {
		o.terms[t] += sign * v
	}
	for t, v := range o.terms {
		if v == 0 {
			delete(o.terms, t)
		}
	}
	return o
}

func (a affine) scale(m int64) affine {
	o := affine{c: a.c * m, terms: map[string]int64{}}
	if m != 0 {
		for t, v := range a.terms {
			o.terms[t] = v * m
		}
	}
	return o
}

func (a affine) isConst() bool { return len(a.terms) == 0 }

func (a affine) String() string {
	keys := make([]string, 0, len(a.terms))
	for t := range a.terms {
		keys = append(keys, t)
	}
	slices.Sort(keys)
	var b strings.Builder
	fmt.Fprintf(&b, "%d", a.c)
	for _, t := range keys {
		fmt.Fprintf(&b, "+%d*%s", a.terms[t], t)
	}
	return b.String()
}

func atom(name string, parts ...string) affine {
	if len(parts) > 0 {
		name += "(" + strings.Join(parts, ",") + ")"
	}
	return affine{terms: map[string]int64{name: 1}}
}

// interner names each distinct canonical form by a short id, so a form built
// from its operands' names stays short however deep they nest. One interner
// serves every kernel compared, so equal forms get equal names.
type interner map[string]int

func (in interner) name(key string) string {
	id, ok := in[key]
	if !ok {
		id = len(in)
		in[key] = id
	}
	return "v" + strconv.Itoa(id)
}

type canon struct {
	in    interner
	k     *ir.Kernel
	depth []int
	loop  []int // the loop op each op sits in, -1 outside
	aff   map[int]*affine
	str   map[int]string
	busy  map[int]bool
}

func newCanon(k *ir.Kernel, in interner) *canon {
	c := &canon{in: in, k: k, depth: make([]int, len(k.Ops)), loop: make([]int, len(k.Ops)),
		aff: map[int]*affine{}, str: map[int]string{}, busy: map[int]bool{}}
	var stack []int
	for i, o := range k.Ops {
		if o.Kind == ir.OpEndLoop && len(stack) > 0 {
			stack = stack[:len(stack)-1]
		}
		c.depth[i] = len(stack)
		c.loop[i] = -1
		if len(stack) > 0 {
			c.loop[i] = stack[len(stack)-1]
		}
		if o.Kind == ir.OpLoop {
			stack = append(stack, i)
		}
	}
	return c
}

func isInt(t ir.Type) bool { return t == ir.U32 || t == ir.I32 }

// index is op i as an affine form, false where it is not integer
// arithmetic the form states (a division, a load): its digest is an atom.
func (c *canon) index(i int) (affine, bool) {
	if a, ok := c.aff[i]; ok {
		return *a, true
	}
	o := c.k.Ops[i]
	if !isInt(o.Type) || c.busy[i] {
		return affine{}, false
	}
	c.busy[i] = true
	defer delete(c.busy, i)
	arg := func(j int) (affine, bool) {
		a := o.Args[j]
		if a == 0 {
			return affine{}, false
		}
		return c.index(int(a) - 1)
	}
	// opaque is op i as an atom named by its form: an integer the affine
	// form does not state (a division, a minimum, a load) inside one it does.
	opaque := func() (affine, bool) {
		delete(c.busy, i)
		r := atom(c.structForm(i))
		c.busy[i] = true
		return r, true
	}
	var r affine
	ok := true
	switch o.Kind {
	case ir.OpConst:
		r = affine{c: o.Imm}
	case ir.OpTID:
		r = atom("tid")
	case ir.OpCTAID:
		r = atom("ctaid")
	case ir.OpNTID:
		r = atom("ntid")
	case ir.OpAdd, ir.OpSub:
		x, ok1 := arg(0)
		y, ok2 := arg(1)
		if ok = ok1 && ok2; ok {
			sign := int64(1)
			if o.Kind == ir.OpSub {
				sign = -1
			}
			r = x.add(y, sign)
		}
	case ir.OpMul:
		x, ok1 := arg(0)
		y, ok2 := arg(1)
		switch {
		case ok1 && ok2 && y.isConst():
			r = x.scale(y.c)
		case ok1 && ok2 && x.isConst():
			r = y.scale(x.c)
		case ok1 && ok2 && len(x.terms) == 1 && len(y.terms) == 1 && x.c == 0 && y.c == 0:
			// A product of two atoms is an atom of the pair, coefficients
			// multiplied.
			var tx, ty string
			var cx, cy int64
			for t, v := range x.terms {
				tx, cx = t, v
			}
			for t, v := range y.terms {
				ty, cy = t, v
			}
			if tx > ty {
				tx, ty = ty, tx
			}
			r = atom("prod", tx, ty).scale(cx * cy)
		default:
			ok = false
		}
	case ir.OpDiv, ir.OpRem:
		// By a constant past the dividend's bound, a quotient is zero and a
		// remainder the dividend.
		y, ok2 := arg(1)
		if hi, okh := c.bound(int(o.Args[0]) - 1); ok2 && y.isConst() && okh && hi < uint64(y.c) && y.c > 0 {
			if o.Kind == ir.OpDiv {
				r = affine{}
			} else {
				r, ok = arg(0)
			}
			break
		}
		ok = false
	case ir.OpShl:
		x, ok1 := arg(0)
		y, ok2 := arg(1)
		if ok = ok1 && ok2 && y.isConst() && y.c >= 0 && y.c < 32; ok {
			r = x.scale(1 << y.c)
		}
	case ir.OpPhi:
		// An induction: phi(start, phi + step) is start + step*iteration.
		start, ok1 := arg(0)
		nx := int(o.Args[1]) - 1
		ok = false
		if ok1 && nx > i {
			n := c.k.Ops[nx]
			if n.Kind == ir.OpAdd {
				for side := 0; side < 2; side++ {
					if int(n.Args[side])-1 != i {
						continue
					}
					if st, ok2 := c.index(int(n.Args[1-side]) - 1); ok2 && st.isConst() {
						lp := c.loop[i]
						trip := int64(0)
						if lp >= 0 {
							trip = c.k.Ops[lp].Imm
						}
						it := atom(fmt.Sprintf("iter d%d t%d", c.depth[i], trip))
						r, ok = start.add(it.scale(st.c), 1), true
					}
				}
			}
		}
	default:
		ok = false
	}
	if !ok {
		if o.Kind == ir.OpPhi {
			return affine{}, false // a carried value that is no induction
		}
		r, _ = opaque()
	}
	c.aff[i] = &r
	return r, true
}

// form is op i's canonical form, as text.
func (c *canon) form(i int) string {
	if f, ok := c.str[i]; ok {
		return f
	}
	if a, ok := c.index(i); ok {
		f := c.in.name(a.String())
		c.str[i] = f
		return f
	}
	return c.structForm(i)
}

// structForm is op i's form by its kind and its operands' forms, not as an
// affine value.
func (c *canon) structForm(i int) string {
	if f, ok := c.str[-1-i]; ok {
		return f
	}
	o := c.k.Ops[i]
	if c.busy[i] {
		return fmt.Sprintf("back(%v)", o.Kind)
	}
	c.busy[i] = true
	defer delete(c.busy, i)
	argF := func(a ir.Value) string {
		if a == 0 {
			return "_"
		}
		j := int(a) - 1
		if j >= i {
			return fmt.Sprintf("back(%v)", c.k.Ops[j].Kind)
		}
		return c.form(j)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%v.%v@%d", o.Kind, o.Type, c.depth[i])
	if o.Tile != nil {
		fmt.Fprintf(&b, "%+v", *o.Tile)
	}
	switch o.Kind {
	case ir.OpParam:
		if int(o.Imm) < len(c.k.Params) {
			p := c.k.Params[o.Imm]
			fmt.Fprintf(&b, "(%v)", p.Elem)
		}
	case ir.OpLoad, ir.OpStore, ir.OpLoadV, ir.OpStoreV, ir.OpPrefetch:
		// The buffer, and the address with the immediate folded in.
		fmt.Fprintf(&b, "(%s,", argF(o.Args[0]))
		if a, ok := c.index(int(o.Args[1]) - 1); ok && o.Args[1] != 0 {
			b.WriteString("[" + a.add(affine{c: o.Imm}, 1).String() + "]")
		} else {
			fmt.Fprintf(&b, "[%s+%d]", argF(o.Args[1]), o.Imm)
		}
		for _, a := range append(o.Args[2:], o.Vargs...) {
			b.WriteString("," + argF(a))
		}
		b.WriteString(")")
	default:
		fmt.Fprintf(&b, "#%d(", o.Imm)
		for _, a := range append(o.Args[:], o.Vargs...) {
			b.WriteString(argF(a) + ",")
		}
		b.WriteString(")")
	}
	f := c.in.name(b.String())
	c.str[-1-i] = f
	return f
}

func hashOf(s string) []byte { h := sha256.Sum256([]byte(s)); return h[:16] }

// forms is each of k's instructions' canonical form.
func forms(k *ir.Kernel) []string {
	c := newCanon(k, interner{})
	out := make([]string, len(k.Ops))
	for i := range k.Ops {
		out[i] = c.form(i)
	}
	return out
}

// opLine is instruction i of k as recorder.Text prints it.
func opLine(k *ir.Kernel, i int) string {
	lines := strings.Split(recorder.Text(k), "\n")
	pre := fmt.Sprintf("%%%d = ", i)
	for _, l := range lines {
		if strings.HasPrefix(l, pre) {
			return l
		}
	}
	return "?"
}

// bound is an upper bound on integer op i's value, false where none is
// known (a thread id, a load, a carried value).
func (c *canon) bound(i int) (uint64, bool) {
	o := c.k.Ops[i]
	arg := func(j int) (uint64, bool) {
		if o.Args[j] == 0 {
			return 0, false
		}
		return c.bound(int(o.Args[j]) - 1)
	}
	switch o.Kind {
	case ir.OpConst:
		if o.Imm >= 0 {
			return uint64(o.Imm), true
		}
	case ir.OpMin:
		x, okx := arg(0)
		y, oky := arg(1)
		switch {
		case okx && oky:
			return min(x, y), true
		case okx:
			return x, true
		case oky:
			return y, true
		}
	case ir.OpAdd:
		x, okx := arg(0)
		y, oky := arg(1)
		if okx && oky {
			return x + y, true
		}
	case ir.OpMul:
		x, okx := arg(0)
		y, oky := arg(1)
		if okx && oky {
			return x * y, true
		}
	case ir.OpDiv:
		x, okx := arg(0)
		if y, oky := arg(1); okx && oky && y > 0 && c.k.Ops[int(o.Args[1])-1].Kind == ir.OpConst {
			return x / y, true
		}
	case ir.OpRem:
		if y, oky := arg(1); oky && y > 0 && c.k.Ops[int(o.Args[1])-1].Kind == ir.OpConst {
			if x, okx := arg(0); okx && x < y {
				return x, true
			}
			return y - 1, true
		}
	case ir.OpAnd:
		if y, oky := arg(1); oky && c.k.Ops[int(o.Args[1])-1].Kind == ir.OpConst {
			return y, true
		}
	case ir.OpShr:
		x, okx := arg(0)
		if y, oky := arg(1); okx && oky && c.k.Ops[int(o.Args[1])-1].Kind == ir.OpConst && y < 64 {
			return x >> y, true
		}
	}
	return 0, false
}
