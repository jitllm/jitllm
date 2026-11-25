package gpu

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestAKernelReadingXWhereItWritesIsNotInPlace forms a block with no MLP,
// x plus the projection of a row of the mixer's width, at mamba-130m's
// widths (768 over 1536), at a fixture's (64 over 128) and at 80, whose rows
// fill no workgroup and split into planes, and holds
// partForm.inPlace to the rule: a form whose kernel writing the output
// reads x is not written over x, though it reads x only at the elements it
// writes -- the launch clamps its surplus threads to the last element and
// each lane storing an element reads it, some after another's store, so
// the residual is added twice (mamba-130m on the V100 parted from its float
// reference by 3.3 times the tier's distance and changed under poisoned
// buffers). A form reading x before its first write is.
func TestAKernelReadingXWhereItWritesIsNotInPlace(t *testing.T) {
	over, before := 0, 0
	for _, sh := range [][2]int{{768, 1536}, {64, 128}, {2048, 512}, {80, 512}} {
		for _, ty := range []jlm.Type{jlm.TypeF32, jlm.TypeQ8} {
			if ty == jlm.TypeQ8 && sh[0]%32 != 0 {
				continue
			}
			for _, rows := range []int{1, 3} {
				name := fmt.Sprintf("%dx%d %v rows %d", sh[0], sh[1], ty, rows)
				b := graph.New()
				x := b.In("x", sh[0])
				mix := b.Emit(graph.OpMap, []graph.ValueID{b.MatVec(x, jlm.RoleAttnV, sh[1])}, []graph.Result{{Name: "mix", Width: sh[1]}}, nil,
					graph.Attrs{Fn: graph.MapExp}, nil, nil)[0]
				b.Out(b.Add(x, b.MatVec(mix, jlm.RoleAttnOut, sh[0])))
				_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnOut: ty, jlm.RoleAttnV: ty},
					Step:   composer.Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32},
					Device: composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32}})
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				f := &partForm{codes: make([]partCode, len(codes))}
				lastRead, firstWrite, shared := -1, len(codes), false
				for ci, c := range codes {
					var args []partArg
					reads, writes := false, false
					for _, bd := range c.Binds {
						a := partArg{kind: partOwn, read: bd.Read, writes: bd.Writes}
						if o := bd.Operand; o.Kind == composer.OperandValue && o.Value == b.G.In[0] {
							a.kind, reads = partIn, reads || bd.Read
						} else if o.Kind == composer.OperandValue && o.Value == b.G.Out[0] {
							a.kind, writes = partOut, writes || bd.Writes
						}
						args = append(args, a)
					}
					if reads {
						lastRead = ci
					}
					if writes {
						firstWrite = min(firstWrite, ci)
					}
					shared = shared || reads && writes
					f.args = append(f.args, args)
				}
				got := f.inPlace(0, 0, 0)
				switch {
				case shared && got:
					t.Fatalf("%s: a kernel reads x and writes the output, and the form is written over x", name)
				case got != (lastRead < firstWrite):
					t.Fatalf("%s: x last read by kernel %d, the output first written by %d, and the form written over x: %v", name, lastRead, firstWrite, got)
				case shared:
					over++
				case got:
					before++
				}
			}
		}
	}
	if over == 0 || before == 0 {
		t.Fatalf("%d forms read x where they write, %d before: the gate sees one side only", over, before)
	}
	t.Logf("%d forms read x in the kernel writing the output and are refused, %d read it before and are written over it", over, before)
}

// TestABandedFormWritesOverXAtItsBand forms a Llama-3.1-8B-shaped block at
// a ragged step's capacity of 512 rows on a device of a V100's CUDA
// statement -- a form of bands, the vector form's calls of up to 2, 4 and
// 8 rows and the matrix unit's past them -- and holds partForm.apartAt to
// the codes a call launches: a call of 1, 2, 4 or 8 rows, whose band reads
// x before it writes the output, is written over x with no copy after,
// though over every code the form is apart (a later band reads x after an
// earlier one writes the output), and a call of the matrix unit's rows is
// apart where its own codes are.
func TestABandedFormWritesOverXAtItsBand(t *testing.T) {
	const embd, heads, kv, dim, ffn = 4096, 32, 8, 128, 14336
	b := graph.New()
	x := b.In("x", embd)
	n := b.Norm(x, jlm.RoleAttnNorm, 1e-5)
	q := b.MatVec(n, jlm.RoleAttnQ, heads*dim)
	k := b.MatVec(n, jlm.RoleAttnK, kv*dim)
	v := b.MatVec(n, jlm.RoleAttnV, kv*dim)
	rope := graph.Attrs{HeadDim: dim, NRot: dim, Layout: graph.RopeNeox}
	qr := b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "qr", Width: heads * dim}}, nil, rope, nil, nil)[0]
	kr := b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "kr", Width: kv * dim}}, nil, rope, nil, nil)[0]
	a := b.Attend(qr, b.KVAppend(kr, v, 0), graph.Attrs{NHead: heads, NKVHead: kv, HeadDim: dim, Scale: 0.125})
	x2 := b.Add(x, b.MatVec(a, jlm.RoleAttnOut, embd))
	m := b.Norm(x2, jlm.RoleFFNNorm, 1e-5)
	act := b.Emit(graph.OpMap, []graph.ValueID{b.MatVec(m, jlm.RoleFFNGate, ffn)}, []graph.Result{{Name: "act", Width: ffn}}, nil,
		graph.Attrs{Fn: graph.MapExp}, nil, nil)[0]
	b.Out(b.Add(x2, b.MatVec(act, jlm.RoleFFNDown, embd)))
	formats := map[jlm.Role]jlm.Type{jlm.RoleAttnNorm: jlm.TypeF32, jlm.RoleFFNNorm: jlm.TypeF32, jlm.RoleFFNDown: jlm.TypeQ6S}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate} {
		formats[r] = jlm.TypeQ4S
	}
	d := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32,
		Matrix: [3]int{16, 16, 16}, Fragment: ir.MMAVolta(), HalfPairs: true, VectorLoads: true, Shared: 96 << 10, Buffers: 32}
	_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: formats,
		Step: composer.Step{Rows: 512, Page: 64, Keys: 1024, ActWindow: 32}, Device: d,
		History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}})
	if err != nil {
		t.Fatal(err)
	}
	f := &partForm{}
	bands := map[composer.Band]bool{}
	for _, c := range codes {
		var args []partArg
		for _, bd := range c.Binds {
			a := partArg{kind: partOwn, read: bd.Read, writes: bd.Writes}
			if o := bd.Operand; o.Kind == composer.OperandValue && o.Value == b.G.In[0] {
				a.kind = partIn
			} else if o.Kind == composer.OperandValue && o.Value == b.G.Out[0] {
				a.kind = partOut
			}
			args = append(args, a)
		}
		f.codes = append(f.codes, partCode{geom: composer.Code{Band: c.Band, Cut: c.Cut}})
		f.args = append(f.args, args)
		bands[c.Band] = true
	}
	if len(bands) < 3 || !f.apart(1) {
		t.Fatalf("the form holds %d bands and is apart over every code %v: want bands, a later one reading x after an earlier writes the output", len(bands), f.apart(1))
	}
	for _, rows := range []int{1, 2, 3, 4, 5, 8} {
		launched := 0
		for i := range f.codes {
			if f.codes[i].geom.Runs(rows, 0) {
				launched++
			}
		}
		if launched == 0 || launched == len(f.codes) {
			t.Fatalf("a call of %d rows launches %d of the form's %d codes: its band is not selected", rows, launched, len(f.codes))
		}
		if f.apartAt(1, rows) {
			t.Errorf("a call of %d rows, %d codes, is written apart and copied over x: its band reads x before it writes the output", rows, launched)
		}
	}
	matrix := 0
	for _, rows := range []int{64, 512} {
		in := f.inPlace(0, 0, rows)
		lastRead, firstWrite := -1, len(f.codes)
		for i, args := range f.args {
			if !f.codes[i].geom.Runs(rows, 0) {
				continue
			}
			for _, a := range args {
				if a.kind == partIn && a.read {
					lastRead = i
				}
				if a.kind == partOut && a.writes {
					firstWrite = min(firstWrite, i)
				}
			}
		}
		if in != (lastRead < firstWrite) {
			t.Errorf("a call of %d rows: x last read by code %d, the output first written by %d, written over x %v", rows, lastRead, firstWrite, in)
		}
		if !in {
			matrix++
		}
	}
	t.Logf("%d bands; calls of 1 to 8 rows written over x, %d of the matrix unit's calls apart", len(bands), matrix)
}
