package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// KVConvForms is one of a history's conversions: rows of FW floats into rows
// of HW words as the history's pages hold them (Narrow), and back (Widen).
type KVConvForms struct {
	FW, HW        int
	Narrow, Widen Part
}

// KVConv is a history's conversions to and from the floats it leaves and
// returns as, on the CPU device, composed once at the instance's placement:
// a binary16 history's (any head width) or a q8 history's, one a head width
// its layers hold. f and h stage the rows a call's floats end inside of.
type KVConv struct {
	conv    []KVConvForms
	half    bool
	dev     *Device
	f       []float32
	h       []uint32
	call    [4]uint32
	zero    []uint32
	recRows [3]uint32
	bind    Call
	// Ran counts the conversions served.
	Ran int
}

// NewKVConv is the conversions conv on dev; half says they are binary16's,
// one for every head width.
func NewKVConv(dev *Device, conv []KVConvForms, half bool) *KVConv {
	k := &KVConv{conv: conv, half: half, dev: dev}
	nf, nh := 0, 0
	for i := range k.conv {
		c := &k.conv[i]
		u := 1
		for _, p := range []*Part{&c.Narrow, &c.Widen} {
			if p.F != nil {
				u = max(u, p.F.Unit, p.F.Rows)
			}
		}
		nf, nh = max(nf, u*c.FW), max(nh, u*c.HW)
	}
	k.f, k.h = make([]float32, nf), make([]uint32, nh)
	k.zero = make([]uint32, 4)
	return k
}

// Convert narrows the floats f of head rows hd wide into the history's bytes
// h, or widens h into f; false where no conversion serves them.
func (k *KVConv) Convert(narrow bool, hd int, f []float32, h []byte) bool {
	for i := range k.conv {
		if c := &k.conv[i]; k.half || c.FW == hd {
			return k.convert(c, narrow, f, h)
		}
	}
	return false
}

// convert runs conversion c over f and h in calls of whole units, the rows
// past them staged in one call of a unit.
func (k *KVConv) convert(c *KVConvForms, narrow bool, f []float32, h []byte) bool {
	p := &c.Widen
	if narrow {
		p = &c.Narrow
	}
	form, why := p.Form()
	if why != nn.DeclineServes || len(h)*c.FW != len(f)*4*c.HW {
		return false
	}
	run := func(fp, hp unsafe.Pointer, n int) {
		b := &k.bind
		*b = Call{Own: k.dev.Own(form), N: n}
		b.IO[0], b.IO[1] = hp, fp
		if narrow {
			b.IO[0], b.IO[1] = fp, hp
		}
		k.call[0] = uint32(n)
		k.recRows = [3]uint32{1, 0, 0}
		b.At[SrcCall], b.At[SrcZero] = unsafe.Pointer(&k.call[0]), unsafe.Pointer(&k.zero[0])
		b.At[SrcRecRows] = unsafe.Pointer(&k.recRows[0])
		for i := range form.Codes {
			k.dev.Call(b, &form.Codes[i])
		}
	}
	u := max(form.Unit, 1)
	whole := len(f) / c.FW / u * u
	for r0 := 0; r0 < whole; {
		n := min(form.Rows/u*u, whole-r0)
		run(unsafe.Pointer(&f[r0*c.FW]), unsafe.Pointer(&h[r0*c.HW*4]), n)
		r0 += n
	}
	if at := whole * c.FW; at < len(f) {
		hb := at * 4 * c.HW / c.FW
		sf, sh := k.f[:u*c.FW], unsafe.Slice((*byte)(unsafe.Pointer(&k.h[0])), u*c.HW*4)
		if narrow {
			clear(sf[copy(sf, f[at:]):])
		} else {
			clear(sh[copy(sh, h[hb:]):])
		}
		run(unsafe.Pointer(&sf[0]), unsafe.Pointer(&sh[0]), u)
		if narrow {
			copy(h[hb:], sh)
		} else {
			copy(f[at:], sf)
		}
	}
	k.Ran++
	return true
}
