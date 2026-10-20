// Package recorder is a device that runs nothing and records what is compiled
// for it: host memory for buffers, every kernel's IR kept, every launch a no-op.
// A test opens a tier on it to see the kernels the composer emits for a model
// without a GPU (model.TestComposerEmitsEveryModel).
package recorder

import (
	"crypto/sha256"
	"fmt"
	"sync"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Device records the kernels compiled for it.
type Device struct {
	api   string
	total uint64
	mu    sync.Mutex
	irs   []*ir.Kernel
	used  uint64
}

// New is a recording device of total bytes whose kernels are written for
// api's target ("ptx", "spirv", "msl").
func New(api string, total uint64) *Device { return &Device{api: api, total: total} }

// IR is every kernel compiled for d, in order.
func (d *Device) IR() []*ir.Kernel {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]*ir.Kernel(nil), d.irs...)
}

func (d *Device) Name() string { return "recorder" }
func (d *Device) API() string  { return d.api }
func (d *Device) Slots() int   { return 0 }
func (d *Device) Close()       {}

func (d *Device) Mem() (free, total uint64, err error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.total - d.used, d.total, nil
}

func (d *Device) Compile(k *ir.Kernel) (backend.Kernel, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.irs = append(d.irs, k)
	return kern{}, nil
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

type kern struct{}

func (kern) Launch(int, int, ...backend.Buf) error { return nil }
func (kern) Close()                                {}

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
func (session) Launch(backend.Kernel, int, int, ...backend.Buf) error {
	return nil
}
func (session) Sync() error { return nil }

// Hash is a kernel's IR as a short digest: its name, group, lanes, parameters
// and every op.
func Hash(k *ir.Kernel) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%v|%d|%v\n", k.Name, k.Group, k.Lanes, k.Params)
	for _, o := range k.Ops {
		fmt.Fprintf(h, "%d %d %v %d %v", o.Kind, o.Type, o.Args, o.Imm, o.Vargs)
		if o.Tile != nil {
			fmt.Fprintf(h, " %+v", *o.Tile)
		}
		fmt.Fprintln(h)
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:12])
}
