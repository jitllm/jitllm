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

	"github.com/jitllm/jitllm/jit_old/gpu/amdgpu"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/msl"
	"github.com/jitllm/jitllm/jit_old/gpu/ptx"
	"github.com/jitllm/jitllm/jit_old/gpu/spirv"
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
	low   *Lowerings
	// events is every launch and host access in order, for LiveLaunched.
	events []event
}

// event is a launch of kernel k over bufs, or (k < 0) a host read or write
// of bufs[0].
type event struct {
	k     int
	write bool
	bufs  []*buf
}

// New is a recording device standing in for p, holding total bytes.
func New(p Profile, total uint64) *Device { return &Device{p: p, total: total} }

// Lowerings are kernels lowered for one profile, shared by the devices that
// stand in for it: a kernel two devices are given (one formed kernel shared
// between tiers) is lowered once.
type Lowerings struct {
	mu sync.Mutex
	m  map[*ir.Kernel]lowered
}

type lowered struct {
	code string
	err  error
}

// NewLowerings is an empty Lowerings.
func NewLowerings() *Lowerings { return &Lowerings{m: map[*ir.Kernel]lowered{}} }

// NewSharing is New lowering through l, every device sharing l standing in
// for the same profile.
func NewSharing(p Profile, total uint64, l *Lowerings) *Device {
	return &Device{p: p, total: total, low: l}
}

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
	code, err := d.lowered(k)
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

// lowered is k lowered for the profile, through the shared lowerings where
// the device has them.
func (d *Device) lowered(k *ir.Kernel) (string, error) {
	if d.low == nil {
		return d.lower(k)
	}
	d.low.mu.Lock()
	l, ok := d.low.m[k]
	d.low.mu.Unlock()
	if ok {
		return l.code, l.err
	}
	code, err := d.lower(k)
	d.low.mu.Lock()
	d.low.m[k] = lowered{code: code, err: err}
	d.low.mu.Unlock()
	return code, err
}

// lower is k as the profile's target takes it.
func (d *Device) lower(k *ir.Kernel) (code string, err error) {
	switch d.p.API {
	case "ptx":
		code, err = ptx.Lower(k, d.p.ISA)
	case "spirv":
		if ir.UsesTiles(k) && !d.p.Tiles {
			return "", fmt.Errorf("spirv: %s uses cooperative-matrix tiles and %s has none", k.Name, d.p.Name)
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
	return code, err
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

func (k kern) Launch(_, _ int, bufs ...backend.Buf) error {
	k.d.mu.Lock()
	defer k.d.mu.Unlock()
	if !k.d.ran[k.i] {
		k.d.ran[k.i] = true
		k.d.order = append(k.d.order, k.i)
	}
	bs := make([]*buf, len(bufs))
	for i, b := range bufs {
		bs[i], _ = b.(*buf)
	}
	k.d.events = append(k.d.events, event{k: k.i, bufs: bs})
	return nil
}

// host notes a host access to b.
func (b *buf) host(write bool) {
	b.d.mu.Lock()
	b.d.events = append(b.d.events, event{k: -1, write: write, bufs: []*buf{b}})
	b.d.mu.Unlock()
}

// LiveLaunched is Launched without the kernels every launch of which was
// dead: each buffer it wrote written again -- by a later launch that stores
// at places no loaded value moves (not a cache row at a table's offset) and
// does not read it, or by the host -- before any launch or the host read it. What is read at
// the end of the recording -- a cache, a state -- counts as read, unless
// every earlier launch of the same kernel was dead: the recording's end is
// one more step of the same loop, and a launch whose output each step
// overwrote unread is dead at the last step too.
func (d *Device) LiveLaunched() []Emitted {
	d.mu.Lock()
	defer d.mu.Unlock()
	reads, writes, fixed := make([][]bool, len(d.irs)), make([][]bool, len(d.irs)), make([][]bool, len(d.irs))
	for i, k := range d.irs {
		reads[i], writes[i], fixed[i] = paramUse(k)
	}
	need := map[*buf]bool{}
	needed := func(b *buf) bool {
		v, ok := need[b]
		return !ok || v
	}
	live, atEnd, deadOnce := make([]bool, len(d.irs)), make([]bool, len(d.irs)), make([]bool, len(d.irs))
	for e := len(d.events) - 1; e >= 0; e-- {
		ev := d.events[e]
		if ev.k < 0 {
			need[ev.bufs[0]] = !ev.write
			continue
		}
		r, w := reads[ev.k], writes[ev.k]
		alive, read := false, false
		for j, b := range ev.bufs {
			if b == nil || j >= len(w) || !w[j] {
				continue
			}
			if needed(b) {
				alive = true
				if _, ok := need[b]; ok {
					read = true
				}
			}
		}
		if !alive {
			deadOnce[ev.k] = true
			continue
		}
		if read {
			live[ev.k] = true
		} else {
			atEnd[ev.k] = true
		}
		for j, b := range ev.bufs {
			if b == nil || j >= len(w) {
				continue
			}
			if w[j] && !r[j] && fixed[ev.k][j] {
				need[b] = false
			}
		}
		for j, b := range ev.bufs {
			if b != nil && j < len(r) && r[j] {
				need[b] = true
			}
		}
	}
	out := make([]Emitted, 0, len(d.order))
	for _, i := range d.order {
		if live[i] || atEnd[i] && !deadOnce[i] {
			out = append(out, Emitted{K: d.irs[i], Code: d.code[i]})
		}
	}
	return out
}

// paramUse is which of k's parameters it reads, which it writes, and
// which it writes only at places no loaded value moves.
func paramUse(k *ir.Kernel) (reads, writes, fixed []bool) {
	reads, writes, fixed = make([]bool, len(k.Params)), make([]bool, len(k.Params)), make([]bool, len(k.Params))
	for i := range fixed {
		fixed[i] = true
	}
	loaded := make([]int8, len(k.Ops)+1) // 0 unknown, 1 reads a load, -1 does not
	var readsLoad func(v ir.Value) bool
	readsLoad = func(v ir.Value) bool {
		if v <= 0 || int(v) > len(k.Ops) {
			return false
		}
		if loaded[v] != 0 {
			return loaded[v] > 0
		}
		loaded[v] = -1
		o := k.Ops[v-1]
		x := o.Kind == ir.OpLoad || o.Kind == ir.OpLoadV || o.Kind == ir.OpLoadVGet || o.Kind == ir.OpPhi
		for _, a := range o.Args {
			x = x || readsLoad(a)
		}
		if x {
			loaded[v] = 1
		}
		return x
	}
	param := func(v ir.Value) int {
		if v <= 0 || int(v) > len(k.Ops) || k.Ops[v-1].Kind != ir.OpParam {
			return -1
		}
		return int(k.Ops[v-1].Imm)
	}
	for _, o := range k.Ops {
		switch o.Kind {
		case ir.OpLoad, ir.OpLoadV, ir.OpTileLoad:
			if p := param(o.Args[0]); p >= 0 {
				reads[p] = true
			}
		case ir.OpStore, ir.OpStoreV, ir.OpTileStore:
			if p := param(o.Args[0]); p >= 0 {
				writes[p] = true
				if readsLoad(o.Args[1]) {
					fixed[p] = false
				}
			}
		}
	}
	return reads, writes, fixed
}

func (kern) Close() {}

type buf struct {
	d *Device
	b []byte
}

func (b *buf) Write(p []byte) error            { b.host(true); copy(b.b, p); return nil }
func (b *buf) WriteAt(off int, p []byte) error { b.host(false); copy(b.b[off:], p); return nil }
func (b *buf) Read(p []byte) error             { b.host(false); copy(p, b.b); return nil }
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
