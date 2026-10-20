// Package recorder is a device that runs nothing and records what is compiled
// and launched on it. It stands in for a real device by a Profile: the target
// its kernels are lowered for (ptx at an sm, spirv, msl) through the same
// lowerer the real backend uses, so it refuses exactly what that lowering
// refuses; the subgroup width it guarantees; whether it holds cooperative
// matrices; how many workgroups it keeps resident for a grid barrier; and
// whether its memory is the host's. Buffers are host memory and every launch
// is a no-op, so a test opens a tier on it to see, with no GPU, the
// instructions the engine emits for a model on that device
// (model.TestDumpEmittedKernels).
package recorder

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"sync"

	"github.com/jitllm/jitllm/jit/gpu/amdgpu"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/msl"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
	"github.com/jitllm/jitllm/jit/gpu/spirv"
)

// Profile is the device a recorder stands in for.
type Profile struct {
	Name string // the profile's name, as Profiles keys it
	API  string // "ptx", "spirv", "msl" or "amdgcn": the target kernels are lowered for
	ISA  string // the ptx target ("sm_70") or the gfx arch ("gfx942"); empty elsewhere
	Arch ir.ISA // the instruction set as an id
	// Lanes is the one subgroup width the device guarantees, 0 for none.
	Lanes int
	// Tiles says a spirv device enabled cooperative matrices; ptx and msl
	// answer from their lowerers.
	Tiles bool
	// Cores and ThreadsPerCore bound the workgroups resident at once, which a
	// grid barrier needs.
	Cores, ThreadsPerCore int
	// MaxGroup is the largest workgroup a pipeline takes.
	MaxGroup int
	Unified  bool
	// MaxBuffer is the largest buffer one binding addresses, 0 for no limit.
	MaxBuffer uint64
}

// Profiles are the devices the engine has run on, by name.
var Profiles = map[string]Profile{
	// NVIDIA, by compute capability: SMs and resident threads an SM.
	"cuda-sm70": {Name: "cuda-sm70", API: "ptx", ISA: "sm_70", Arch: ir.PTX(70), Lanes: ir.SubgroupLanes, Cores: 80, ThreadsPerCore: 2048, MaxGroup: 1024},  // Tesla V100
	"cuda-sm75": {Name: "cuda-sm75", API: "ptx", ISA: "sm_75", Arch: ir.PTX(75), Lanes: ir.SubgroupLanes, Cores: 40, ThreadsPerCore: 1024, MaxGroup: 1024},  // Tesla T4
	"cuda-sm80": {Name: "cuda-sm80", API: "ptx", ISA: "sm_80", Arch: ir.PTX(80), Lanes: ir.SubgroupLanes, Cores: 108, ThreadsPerCore: 2048, MaxGroup: 1024}, // A100
	"cuda-sm86": {Name: "cuda-sm86", API: "ptx", ISA: "sm_86", Arch: ir.PTX(86), Lanes: ir.SubgroupLanes, Cores: 20, ThreadsPerCore: 1536, MaxGroup: 1024},  // RTX 3050 Ti Laptop
	"cuda-sm89": {Name: "cuda-sm89", API: "ptx", ISA: "sm_89", Arch: ir.PTX(89), Lanes: ir.SubgroupLanes, Cores: 58, ThreadsPerCore: 1536, MaxGroup: 1024},  // L4
	"cuda-sm90": {Name: "cuda-sm90", API: "ptx", ISA: "sm_90", Arch: ir.PTX(90), Lanes: ir.SubgroupLanes, Cores: 132, ThreadsPerCore: 2048, MaxGroup: 1024}, // H100
	// AMD through HIP, by gfx arch: compute units and resident threads a CU.
	"hip-gfx942":  {Name: "hip-gfx942", API: "amdgcn", ISA: "gfx942", Arch: ir.GFX(942), Lanes: ir.SubgroupLanes, Cores: 304, ThreadsPerCore: 2048, MaxGroup: 1024},   // MI300X
	"hip-gfx1030": {Name: "hip-gfx1030", API: "amdgcn", ISA: "gfx1030", Arch: ir.GFX(1030), Lanes: ir.SubgroupLanes, Cores: 80, ThreadsPerCore: 2048, MaxGroup: 1024}, // RX 6900
	"hip-gfx1100": {Name: "hip-gfx1100", API: "amdgcn", ISA: "gfx1100", Arch: ir.GFX(1100), Lanes: ir.SubgroupLanes, Cores: 96, ThreadsPerCore: 2048, MaxGroup: 1024}, // RX 7900
	"hip-gfx1201": {Name: "hip-gfx1201", API: "amdgcn", ISA: "gfx1201", Arch: ir.GFX(1201), Lanes: ir.SubgroupLanes, Cores: 64, ThreadsPerCore: 2048, MaxGroup: 1024}, // RX 9070
	// Vulkan: a discrete card with 32-lane subgroups and no cooperative
	// matrices; Intel's Iris Xe, integrated; lavapipe, on the CPU, which
	// promises no subgroup width.
	"vulkan":          {Name: "vulkan", API: "spirv", Arch: ir.SPIRV(), Lanes: ir.SubgroupLanes, Cores: 20, ThreadsPerCore: 1024, MaxGroup: 1024, MaxBuffer: 1 << 32},
	"vulkan-irisxe":   {Name: "vulkan-irisxe", API: "spirv", Arch: ir.SPIRV(), Lanes: ir.SubgroupLanes, Cores: 96, ThreadsPerCore: 448, MaxGroup: 1024, Unified: true, MaxBuffer: 1 << 32},
	"vulkan-lavapipe": {Name: "vulkan-lavapipe", API: "spirv", Arch: ir.SPIRV(), Cores: 6, ThreadsPerCore: 1024, MaxGroup: 1024, Unified: true, MaxBuffer: 1 << 31},
	// Apple M4: 10 GPU cores, unified memory.
	"metal-m4": {Name: "metal-m4", API: "msl", Arch: ir.MSL(), Lanes: ir.SubgroupLanes, Cores: 10, ThreadsPerCore: 1024, MaxGroup: 1024, Unified: true, MaxBuffer: 1 << 34},
}

// Device records the kernels compiled and launched on it.
type Device struct {
	p     Profile
	total uint64
	mu    sync.Mutex
	irs   []*ir.Kernel
	code  []string // code[i]: irs[i] lowered for the profile's target
	ran   []bool   // ran[i]: irs[i] was launched
	order []int    // the kernels in the order first launched
	used  uint64
}

// New is a recording device standing in for p, holding total bytes.
func New(p Profile, total uint64) *Device { return &Device{p: p, total: total} }

// Emitted is a launched kernel and its instructions as lowered for the
// device: ptx, amdgcn or msl source, or the IR ops (Text) the spirv emitter
// encodes. The kernel's own name in it is replaced by "K", so two kernels with
// the same instructions read alike whoever named them.
type Emitted struct {
	K    *ir.Kernel
	Code string
}

// Launched is every kernel launched on d and its instructions, in the order
// first launched.
func (d *Device) Launched() []Emitted {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := make([]Emitted, 0, len(d.order))
	for _, i := range d.order {
		out = append(out, Emitted{K: d.irs[i], Code: d.code[i]})
	}
	return out
}

// Compiled is every kernel compiled for d whose IR keep accepts, launched
// or not, in the order compiled.
func (d *Device) Compiled(keep func(*ir.Kernel) bool) []Emitted {
	d.mu.Lock()
	defer d.mu.Unlock()
	var out []Emitted
	for i, k := range d.irs {
		if keep(k) {
			out = append(out, Emitted{K: k, Code: d.code[i]})
		}
	}
	return out
}

// IR is every kernel compiled for d, in order.
func (d *Device) IR() []*ir.Kernel {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]*ir.Kernel(nil), d.irs...)
}

func (d *Device) Name() string { return "recorder " + d.p.Name }
func (d *Device) API() string  { return d.p.API }
func (d *Device) ISA() string  { return d.p.ISA }

// Arch is the profile's instruction set.
func (d *Device) Arch() ir.ISA        { return d.p.Arch }
func (d *Device) Slots() int          { return 0 }
func (d *Device) Close()              {}
func (d *Device) UnifiedMemory() bool { return d.p.Unified }

// MaxBuffer is the profile's binding limit, or the device's size when it
// states none.
func (d *Device) MaxBuffer() uint64 {
	if d.p.MaxBuffer == 0 {
		return d.total
	}
	return d.p.MaxBuffer
}

func (d *Device) GuaranteedLanes(w int) (bool, string) {
	if w <= 0 || w == d.p.Lanes {
		return true, ""
	}
	return false, fmt.Sprintf("%s guarantees %d lanes, not %d", d.p.Name, d.p.Lanes, w)
}

func (d *Device) GridLimit(k backend.Kernel) (int, error) {
	src := k.(kern).src
	if !src.GridSync() {
		return 0, fmt.Errorf("recorder: %s has no grid barrier", src.Name)
	}
	g := max(src.Group[0]*max(src.Group[1], 1)*max(src.Group[2], 1), 1)
	return d.p.Cores * max(d.p.ThreadsPerCore/g, 1), nil
}

func (d *Device) Mem() (free, total uint64, err error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.total - d.used, d.total, nil
}

// Compile lowers k for the profile's target as the real backend does, and
// records it.
func (d *Device) Compile(k *ir.Kernel) (backend.Kernel, error) {
	if g := k.Group[0] * max(k.Group[1], 1) * max(k.Group[2], 1); g > d.p.MaxGroup {
		return nil, fmt.Errorf("recorder: %s wants a %d-thread workgroup and %s allows %d", k.Name, g, d.p.Name, d.p.MaxGroup)
	}
	if k.Lanes != 0 {
		if ok, why := d.GuaranteedLanes(k.Lanes); !ok {
			return nil, fmt.Errorf("recorder: %s: %s", k.Name, why)
		}
	}
	var err error
	var code string
	switch d.p.API {
	case "ptx":
		code, err = ptx.Lower(k, d.p.ISA)
	case "spirv":
		if ir.UsesTiles(k) && !d.p.Tiles {
			return nil, fmt.Errorf("spirv: %s uses cooperative-matrix tiles and %s has none", k.Name, d.p.Name)
		}
		if _, err = spirv.Emit(k); err == nil {
			code = Text(k)
		}
	case "msl":
		code, err = msl.Emit(k)
	case "amdgcn":
		var tg amdgpu.Target
		if tg, err = amdgpu.TargetFor(d.p.ISA); err == nil {
			code, err = amdgpu.Lower(k, tg)
		}
	default:
		err = fmt.Errorf("recorder: no lowering for %q", d.p.API)
	}
	if err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.irs = append(d.irs, k)
	d.code = append(d.code, strings.ReplaceAll(code, k.Name, "K"))
	d.ran = append(d.ran, false)
	return kern{d: d, i: len(d.irs) - 1, src: k}, nil
}

func (d *Device) Alloc(n int) (backend.Buf, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.used+uint64(n) > d.total {
		return nil, fmt.Errorf("recorder: %d bytes past the %d the device holds", n, d.total)
	}
	d.used += uint64(n)
	return &buf{d: d, b: make([]byte, n)}, nil
}

func (d *Device) Copy(dst backend.Buf, dstOff int, src backend.Buf, srcOff, n int) error {
	copy(dst.(*buf).b[dstOff:dstOff+n], src.(*buf).b[srcOff:srcOff+n])
	return nil
}

func (d *Device) Session(f func(backend.Session)) { f(session{}) }

// kern is a compiled kernel: launching it marks it run.
type kern struct {
	d   *Device
	i   int
	src *ir.Kernel
}

func (k kern) IR() *ir.Kernel { return k.src }

func (k kern) Launch(int, int, ...backend.Buf) error {
	k.d.mu.Lock()
	defer k.d.mu.Unlock()
	if !k.d.ran[k.i] {
		k.d.ran[k.i] = true
		k.d.order = append(k.d.order, k.i)
	}
	return nil
}

func (kern) Close() {}

type buf struct {
	d *Device
	b []byte
}

func (b *buf) Write(p []byte) error            { copy(b.b, p); return nil }
func (b *buf) WriteAt(off int, p []byte) error { copy(b.b[off:], p); return nil }
func (b *buf) Read(p []byte) error             { copy(p, b.b); return nil }
func (b *buf) Free() {
	b.d.mu.Lock()
	defer b.d.mu.Unlock()
	b.d.used -= uint64(len(b.b))
	b.b = nil
}

type session struct{}

func (session) Write(b backend.Buf, p []byte) error            { return b.Write(p) }
func (session) WriteAt(b backend.Buf, off int, p []byte) error { return b.WriteAt(off, p) }
func (session) Read(b backend.Buf, p []byte) error             { return b.Read(p) }
func (session) Launch(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	return k.Launch(groups, width, bufs...)
}
func (session) Sync() error { return nil }

// Hash is a kernel's instructions as a short digest: its workgroup, lanes,
// parameter types and every op -- not its name or its parameters' names, so
// two kernels emitting the same instructions hash alike whoever named them.
func Hash(k *ir.Kernel) string {
	h := sha256.New()
	fmt.Fprintf(h, "%v|%d|", k.Group, k.Lanes)
	for _, p := range k.Params {
		fmt.Fprintf(h, "%d/%d/%v,", p.Elem, p.Tile, p.IsTile)
	}
	fmt.Fprintln(h)
	for _, o := range k.Ops {
		fmt.Fprintf(h, "%d %d %v %d %v", o.Kind, o.Type, o.Args, o.Imm, o.Vargs)
		if o.Tile != nil {
			fmt.Fprintf(h, " %+v", *o.Tile)
		}
		fmt.Fprintln(h)
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:12])
}

// Text is a kernel's instructions, one op a line ("%i = kind type args imm"),
// after a header of its workgroup, lanes and parameter types: what Hash
// digests, to read or diff.
func Text(k *ir.Kernel) string {
	var b strings.Builder
	fmt.Fprintf(&b, "kernel %s group %v lanes %d\n", k.Name, k.Group, k.Lanes)
	for i, p := range k.Params {
		fmt.Fprintf(&b, "param %d %s %v tile=%v\n", i, p.Name, p.Elem, p.IsTile)
	}
	for i, o := range k.Ops {
		fmt.Fprintf(&b, "%%%d = %v %v %v %d", i, o.Kind, o.Type, o.Args, o.Imm)
		if o.Vargs != nil {
			fmt.Fprintf(&b, " %v", o.Vargs)
		}
		if o.Tile != nil {
			fmt.Fprintf(&b, " %+v", *o.Tile)
		}
		b.WriteByte('\n')
	}
	return b.String()
}
