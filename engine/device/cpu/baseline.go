package cpu

import "fmt"

// ErrNoVNNI is Map refusing a kernel that contains VPDPBUSD on a host without
// AVX-VNNI. It is a decline: the builders in nn skip the kernel (the packed
// family has a pre-VNNI sequence to use instead) and a mandatory kernel's
// mustMap names it.
const ErrNoVNNI jitError = "cpu: this kernel contains VPDPBUSD and this host has no AVX-VNNI"

// jitError is a sentinel error of this package: a constant, matched with
// errors.Is like any other.
type jitError string

func (e jitError) Error() string { return string(e) }

// ErrISA is what every ISAError matches under errors.Is: Map refusing a kernel
// that needs extensions this host does not have.
const ErrISA jitError = "cpu: this kernel needs instructions this host does not have"

// ISAError is Map's refusal of a kernel whose ISA this host does not cover. It
// names the missing extensions, so the message reads as a property of the host
// rather than a bug in the engine.
type ISAError struct {
	Need     ISA  // what the kernel requires
	Missing  ISA  // Need minus what this host has
	Declared bool // whether Need came from the kernel's own declaration
	detail   string
}

func (e *ISAError) Error() string {
	where := "an undeclared (AVX2-tier: VEX-encoded, FMA, F16C) kernel"
	if e.Declared {
		where = "a kernel declared " + e.Need.String()
	}
	return fmt.Sprintf("cpu: %s cannot run here: this host lacks %s", where, e.detail)
}

// Is makes errors.Is(err, ErrISA) true for every ISAError.
func (e *ISAError) Is(target error) bool { return target == ErrISA }

// isaCheck refuses code whose ISA f does not cover. It is the amd64 half of
// Host.gate, factored out so the refusal text is built in one place.
func isaCheck(f Features, code []byte) error {
	need, declared := KernelISA(code)
	if miss := need &^ f.ISA(); miss != 0 {
		return &ISAError{Need: need, Missing: miss, Declared: declared, detail: missingNames(f, need)}
	}
	return nil
}
