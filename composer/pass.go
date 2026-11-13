package composer

import (
	"fmt"
	"time"

	"github.com/jitllm/jitllm/composer/x86"
)

// Pass is one optimization: IR in, rewritten IR out. It decides from the
// IR's shape and structure and the device, never from names.
type Pass interface {
	Name() string
	Apply(ks []*Kernel, d Device) ([]*Kernel, error)
}

// Profile is a device's choices: what its parts' decompositions carry, and
// its passes in order, any of them turned off by name.
type Profile struct {
	Int8 *Int8Form
	// ScanLoops steps a scan's state row in memory, its components loops
	// (generateScanLoops), where a device's element is one thread.
	ScanLoops bool
	Passes    []Pass
	Off       map[string]bool
	// Narrow, where set, is the passes of a device whose stated ISA holds
	// Floor's bits and lacks any of Wide's: on x86, a host at the SSE tier
	// without AVX2, FMA and F16C forms its kernels four floats a vector. An
	// arm64 states neither.
	Narrow      []Pass
	Wide, Floor uint64
}

// on is the profile p forms by on device d.
func (p Profile) on(d Device) Profile {
	if p.narrows(d) {
		p.Passes, p.Narrow = p.Narrow, nil
	}
	return p
}

// narrows reports whether p forms device d's kernels from its narrow passes.
func (p Profile) narrows(d Device) bool {
	return p.Narrow != nil && d.ISA&p.Floor == p.Floor && d.ISA&p.Wide != p.Wide
}

// NarrowsCPU reports whether the composer forms CPU device d's kernels from
// its narrow passes: an x86 host without AVX2, FMA and F16C, whose binary16
// conversions are software. An arm64 states neither set and never does.
func NarrowsCPU(d Device) bool { return cpuProfile().narrows(d) }

// Int8Form is the int8 form a device keeps of a block's contraction inputs
// produced by a row's op: windows of Window by an exact scale amax/127,
// codes rounded half away from zero packed Word to a word, the codes' sums
// a Sums: the scale and its inverse divisions where the device's are exact
// (Device.ExactDiv), made exact in integers where they are not.
// UnsignedDot is the instruction sets (Device.ISA's bits) under which the
// device's byte dot reads one operand's bytes unsigned and sums them
// without saturating (OpDot4U), 0 for none: a signed weight's dot is there
// its bytes offset by 128, less 128 times the input codes' sum.
type Int8Form struct {
	Window, Word, Sums int
	UnsignedDot        uint64
	// Slot, where set, is the elements one scale word covers under a wider
	// window, the window's scale stored once a slot (Context.slot).
	Slot int
	// SumCodes runs the codes map a Sums of a window a row and sums the
	// Sums straight from the codes, with no word sums between.
	SumCodes bool
}

// run applies p's passes to ks in order.
func (p Profile) run(ks []*Kernel, d Device) ([]*Kernel, error) { return p.counted(ks, d, nil) }

// counted is run with each pass's time counted into c.
func (p Profile) counted(ks []*Kernel, d Device, c *formCache) ([]*Kernel, error) {
	if p.Off[Vectorize{}.Name()] {
		// No loop across lanes: the form the registers' refusal leaves.
		for _, k := range ks {
			k.Unvector = true
		}
	}
	// Every pass reads which words a vector register holds (VectorWords),
	// so an unvectored kernel's general-register words are its from the
	// first pass, not only the binding's.
	for _, k := range ks {
		k.BitsGPR = k.Unvector && d.ISA&uint64(x86.ISASSE2) != 0
	}
	for _, ps := range p.Passes {
		if p.Off[ps.Name()] {
			continue
		}
		t0 := time.Now()
		out, err := ps.Apply(ks, d)
		c.clock(ps.Name(), time.Since(t0))
		if err != nil {
			return nil, fmt.Errorf("jit: pass %s: %w", ps.Name(), err)
		}
		ks = out
	}
	return ks, nil
}

// eachKernel applies f to every kernel, recording the pass on those f
// changes.
func eachKernel(name string, ks []*Kernel, f func(k *Kernel) ([]*Kernel, bool, error)) ([]*Kernel, error) {
	var out []*Kernel
	for _, k := range ks {
		ns, changed, err := f(k)
		if err != nil {
			return nil, err
		}
		if !changed {
			out = append(out, k)
			continue
		}
		for _, n := range ns {
			n.Passes = append(append([]string(nil), k.Passes...), name)
			n.compact()
		}
		out = append(out, ns...)
	}
	return out, nil
}
