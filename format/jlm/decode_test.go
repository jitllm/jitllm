package jlm

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// gv is a value of the Go evaluator: an integer's bits or a float.
type gv struct {
	u uint32
	f float32
}

// planes evaluates the reading rules over a tensor's three word arrays.
type planes struct{ qs, d, sc []byte }

func (planes) U(x uint32) gv              { return gv{u: x} }
func (planes) Add(a, b gv) gv             { return gv{u: a.u + b.u} }
func (planes) Sub(a, b gv) gv             { return gv{u: a.u - b.u} }
func (planes) Mul(a, b gv) gv             { return gv{u: a.u * b.u} }
func (planes) Div(a, b gv) gv             { return gv{u: a.u / b.u} }
func (planes) Rem(a, b gv) gv             { return gv{u: a.u % b.u} }
func (planes) Min(a, b gv) gv             { return gv{u: min(a.u, b.u)} }
func (planes) Shr(a, b gv) gv             { return gv{u: a.u >> b.u} }
func (planes) Shl(a, b gv) gv             { return gv{u: a.u << b.u} }
func (planes) And(a, b gv) gv             { return gv{u: a.u & b.u} }
func (planes) Xor(a, b gv) gv             { return gv{u: a.u ^ b.u} }
func (planes) Code(t *[16]uint8, q gv) gv { return gv{u: uint32(t[q.u])} }
func (planes) Half(a gv) gv               { return gv{f: float32(quant.DecodeHalf(uint16(a.u)))} }
func (planes) F32(a gv) gv                { return gv{f: math.Float32frombits(a.u)} }
func (planes) Int(a gv) gv                { return gv{f: float32(int32(a.u))} }
func (planes) Float(a gv) gv              { return gv{f: float32(a.u)} }
func (planes) FConst(x float32) gv        { return gv{f: x} }
func (planes) FMul(a, b gv) gv            { return gv{f: a.f * b.f} }
func (planes) FSub(a, b gv) gv            { return gv{f: a.f - b.f} }
func (planes) FAdd(a, b gv) gv            { return gv{f: a.f + b.f} }

func (p planes) Word(pl Plane, i gv) gv {
	b := [...][]byte{p.qs, p.d, p.sc}[pl]
	return gv{u: binary.LittleEndian.Uint32(b[4*i.u:])}
}

// TestDecodeMatchesThePacker holds DecodeElement, evaluated in Go, to the
// packer's reference reader (oracle.RowPacked) on every element of every row
// of random tensors of every packed type: random payload and scale words,
// super-scales drawn finite, an odd row count so two rows a d word leave a
// half alone.
func TestDecodeMatchesThePacker(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	half := func() uint32 {
		// A binary16 of exponent 8..22: finite, neither tiny nor huge.
		return uint32(rng.IntN(2))<<15 | uint32(8+rng.IntN(15))<<10 | uint32(rng.IntN(1024))
	}
	checked := 0
	for c := TypeNone + 1; c.Valid(); c++ {
		l, ok := Layout(c)
		q, packed := packerOf(c)
		if !ok || !packed {
			continue
		}
		const rows = 5
		k := 2 * l.Sub * l.SuperSub
		k = max(k, q.Elems())
		k = (k + q.Elems() - 1) / q.Elems() * q.Elems()
		nq, nd, nsc, err := kernels.PackedWords(q, rows, k)
		if err != nil {
			t.Fatalf("%s: %v", c, err)
		}
		if a, b, cc := l.Words(rows, k); a != nq || b != nd || cc != nsc {
			t.Fatalf("%s: Words says %d/%d/%d, the packer %d/%d/%d", c, a, b, cc, nq, nd, nsc)
		}
		p := planes{qs: make([]byte, 4*nq), d: make([]byte, 4*nd), sc: make([]byte, 4*nsc)}
		for i := 0; i < nq; i++ {
			binary.LittleEndian.PutUint32(p.qs[4*i:], rng.Uint32())
		}
		for i := 0; i < nsc; i++ {
			binary.LittleEndian.PutUint32(p.sc[4*i:], rng.Uint32())
		}
		for i := 0; i < nd; i++ {
			var w uint32
			switch {
			case l.E8M0:
				for s := 0; s < 4; s++ {
					w |= uint32(110+rng.IntN(30)) << (8 * s)
				}
			case l.D32:
				w = math.Float32bits(float32(rng.NormFloat64()))
			default:
				w = half() | half()<<16
			}
			binary.LittleEndian.PutUint32(p.d[4*i:], w)
		}
		want := make([]float32, k)
		for r := range rows {
			if err := oracle.RowPacked(want, q, p.qs, p.d, p.sc, r, rows, k); err != nil {
				t.Fatalf("%s: %v", c, err)
			}
			for e := range k {
				got := DecodeElement[gv](p, l, rows, gv{u: uint32(r)}, gv{u: uint32(e)}).f
				if got != want[e] {
					t.Fatalf("%s: row %d element %d decodes to %v, the packer's reader %v", c, r, e, got, want[e])
				}
				checked++
			}
		}
	}
	if checked == 0 {
		t.Fatal("no packed type was decoded -- this gate proved nothing")
	}
	t.Logf("%d elements of every packed type decode as the packer's reader does", checked)
}

// TestDecodeVerbatimMatchesTheDecoders holds DecodeVerbatim to the quant
// package's own decoders of f32, f16 and bf16, element by element.
func TestDecodeVerbatimMatchesTheDecoders(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const rows, k = 3, 7
	for _, c := range []Type{TypeF32, TypeF16, TypeBF16} {
		width := 2
		if c == TypeF32 {
			width = 4
		}
		raw := make([]byte, (rows*k*width+3)/4*4)
		for i := 0; i < rows*k; i++ {
			v := float32(rng.NormFloat64())
			switch c {
			case TypeF32:
				binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
			case TypeF16:
				binary.LittleEndian.PutUint16(raw[2*i:], quant.EncodeHalf(v))
			default:
				binary.LittleEndian.PutUint16(raw[2*i:], uint16(math.Float32bits(v)>>16))
			}
		}
		src, _ := SourceType(c)
		want := make([]float32, rows*k)
		if err := quant.Dequant32(src, raw[:rows*k*width], want); err != nil {
			t.Fatalf("%s: %v", c, err)
		}
		p := planes{qs: raw}
		for r := range rows {
			for e := range k {
				if got := DecodeVerbatim[gv](p, c, k, gv{u: uint32(r)}, gv{u: uint32(e)}).f; got != want[r*k+e] {
					t.Fatalf("%s: element %d of row %d decodes to %v, the decoder %v", c, e, r, got, want[r*k+e])
				}
			}
		}
	}
}
