package cpudev

import (
	"fmt"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// Device is the CPU device's launches: its worker pool, a call block of
// words for each worker, and the counters WithProfile arms (Prof).
type Device struct {
	workers *cpu.Pool
	args    []cpu.Args
	// part is the workers the launch running sizes its calls for, and ends
	// its phase's kernels' calls' running totals (group).
	part int
	ends []int
	// t0 and maxStart are the profiled launch's start and its latest call's.
	t0       time.Time
	maxStart int64
	run      phase
	Prof     bool
	// own is each form's own buffers, by form: runs of one form follow
	// each other, so they share them; mapped is those held off the heap.
	own    map[*Form][][]uint32
	mapped [][]byte
}

// NewDevice is the CPU device launching on workers.
func NewDevice(workers *cpu.Pool) *Device {
	d := &Device{workers: workers, args: argBlocks(workers.Max())}
	d.run.fn = d.run.region
	d.run.gfn = d.run.groupRegion
	d.run.cfn = d.run.chainRegion
	return d
}

// Call is what a launch of a form's kernels binds, by address: the rows'
// residual streams (X, a stream Stride apart) and their count N, the form's
// places is a caller's IO across its calls, grown to the most places a
// form of its binds, so a warm call allocates nothing.
type places []unsafe.Pointer

// of is n places, cleared.
func (p *places) of(n int) []unsafe.Pointer {
	if cap(*p) < n {
		*p = make([]unsafe.Pointer, n)
	}
	s := (*p)[:n]
	clear(s)
	return s
}

// own buffers, the places of a graph apart from its residual (IO), its
// weights in the form's order (Form.Weights), its rotary tables and planes
// by instance, and each other source's address (At, by Src).
type Call struct {
	Own     [][]uint32
	X       []float32
	Stride  int
	N       int
	IO      []unsafe.Pointer
	Weights []unsafe.Pointer
	Rope    [2]unsafe.Pointer
	Planes  [2]unsafe.Pointer
	At      [NumSrc]unsafe.Pointer
}

// phase is the kernel a pool region calls, its calls the region's range.
type phase struct {
	d    *Device
	b    *Call
	c    *Code
	live int
	fn   func(worker, lo, hi int)
	// g is a phase's kernels (group), gfn the region that runs them;
	// cfn runs them unit by unit, units units (chain).
	g     []Code
	gfn   func(worker, lo, hi int)
	cfn   func(worker, lo, hi int)
	units int
}

// Call runs one kernel of a form: once, or over its calls on the pool.
func (d *Device) Call(b *Call, c *Code) {
	if d.Prof {
		t0 := time.Now()
		d.t0, d.maxStart = t0, 0
		d.launch(b, c)
		c.nanos += int64(time.Since(t0))
		c.runs++
		c.pmaxNs += d.maxStart
		return
	}
	d.launch(b, c)
}

func (d *Device) launch(b *Call, c *Code) {
	n, live := c.n, 0
	if c.live.Live > 0 {
		live = c.live.LiveUnitsOf(b.N)
		n = live * c.inner
	}
	if !c.tiles {
		live = 0
	}
	d.part = d.workers.Workers()
	if c.serial {
		d.part = 1
	}
	if c.share.Share > 0 {
		_, n = c.share.ShareRows(d.part)
	}
	if n <= 1 || c.serial || d.part <= 1 {
		a := &d.args[0]
		for i := range max(n, 1) {
			d.words(a, b, c, c.callAt(i, live))
			c.code.Call(a)
		}
		return
	}
	r := &d.run
	r.d, r.b, r.c, r.live = d, b, c, live
	d.workers.DoLabeled(c.label, n, 1, r.fn)
}

// callAt is the call the i-th of a kernel's calls runs: in order, or for a
// kernel tiling a weight over live units of rows (tiles), a tile's every
// unit before the next tile's.
func (c *Code) callAt(i, live int) int {
	if live <= 0 {
		return i
	}
	return i%live*c.inner + i/live
}

// argWords is one call's word block: cpu.Args' layout, which the kernel
// reads, held as words. A word is an address and its offset, which past a
// slice's last element is no pointer the collector accepts (a field of
// cpu.Args is), or a count; the addresses are of buffers its caller holds.
type argWords [(unsafe.Sizeof(cpu.Args{}) + 7) / 8]uint64

// argBlocks is n word blocks seen as the cpu.Args a kernel call takes: the
// collector scans an object by the type it was allocated as.
func argBlocks(n int) []cpu.Args {
	w := make([]argWords, n)
	return unsafe.Slice((*cpu.Args)(unsafe.Pointer(&w[0])), n)
}

// coord is call at's coordinate along axis ax: a Share code's first row
// for calls of rows rows, else coordOf's.
func (c *Code) coord(at, ax, rows int) int {
	if rows > 0 {
		return at * rows / max(c.share.ShareUnit, 1)
	}
	return coordOf(c.calls, at, ax)
}

// coordOf is call at's coordinate along axis ax of the calls' space calls,
// the last axis fastest.
func coordOf(calls []int, at, ax int) int {
	for a := len(calls) - 1; a > ax; a-- {
		at /= calls[a]
	}
	return at % calls[ax]
}

// region runs the calls [lo, hi) of the kernel the device's launch names, on
// worker w's words.
func (r *phase) region(w, lo, hi int) {
	a := &r.d.args[w]
	for i := lo; i < hi; i++ {
		r.d.words(a, r.b, r.c, r.c.callAt(i, r.live))
		if r.d.Prof {
			r.d.profCall(r.c, a)
			continue
		}
		r.c.code.Call(a)
	}
}

func (d *Device) profCall(c *Code, a *cpu.Args) {
	t := time.Now()
	c.code.Call(a)
	off := int64(t.Sub(d.t0))
	atomic.AddInt64(&c.pcalls, 1)
	atomic.AddInt64(&c.pcallNs, int64(time.Since(t)))
	atomic.AddInt64(&c.pstartNs, off)
	for {
		m := atomic.LoadInt64(&d.maxStart)
		if off <= m || atomic.CompareAndSwapInt64(&d.maxStart, m, off) {
			return
		}
	}
}

// groupRegion runs the calls [lo, hi) of the phase's kernels on worker w's
// words.
func (r *phase) groupRegion(w, lo, hi int) {
	r.d.groupRun(&r.d.args[w], r.b, r.g, lo, hi)
}

// Calls runs kernels cs in order, each with those its phase joins
// (Form.joinPhases).
func (d *Device) Calls(b *Call, cs []Code) {
	for i := 0; i < len(cs); i++ {
		c := &cs[i]
		if c.chain > 0 {
			g := cs[i : i+c.chain+1]
			i += c.chain
			if !d.Prof {
				d.chain(b, g, c.units)
				continue
			}
			t0 := time.Now()
			d.chain(b, g, c.units)
			c.nanos += int64(time.Since(t0))
			c.runs++
			continue
		}
		if c.join == 0 {
			d.Call(b, c)
			continue
		}
		g := cs[i : i+c.join+1]
		i += c.join
		if !d.Prof {
			d.group(b, g)
			continue
		}
		t0 := time.Now()
		d.t0, d.maxStart = t0, 0
		d.group(b, g)
		dt := int64(time.Since(t0))
		g[0].pmaxNs += d.maxStart
		ends := d.ends
		for j := range g {
			n := ends[j]
			if j > 0 {
				n -= ends[j-1]
			}
			g[j].nanos += dt * int64(n) / int64(ends[len(ends)-1])
			g[j].runs++
		}
	}
}

// group runs the kernels g as one phase over the pool: their calls one
// space, in order of the kernels.
func (d *Device) group(b *Call, g []Code) {
	d.part = d.workers.Workers()
	d.ends = d.ends[:0]
	n := 0
	for j := range g {
		if g[j].share.Share > 0 {
			_, k := g[j].share.ShareRows(d.part)
			n += k
		} else {
			n += max(g[j].n, 1)
		}
		d.ends = append(d.ends, n)
	}
	if d.workers.Workers() <= 1 {
		d.groupRun(&d.args[0], b, g, 0, n)
		return
	}
	r := &d.run
	r.d, r.b, r.g = d, b, g
	d.workers.DoLabeled(g[0].label, n, 1, r.gfn)
}

// chain runs the kernels g as one phase over the pool, units units:
// a unit its calls of each kernel in turn (chainPhases).
func (d *Device) chain(b *Call, g []Code, units int) {
	d.part = d.workers.Workers()
	if d.part <= 1 {
		d.chainRun(&d.args[0], b, g, units, 0, units)
		return
	}
	r := &d.run
	r.d, r.b, r.g, r.units = d, b, g, units
	d.workers.DoLabeled(g[0].label, units, 1, r.cfn)
}

// chainRun runs the units [lo, hi) of the kernels g's phase on a.
func (d *Device) chainRun(a *cpu.Args, b *Call, g []Code, units, lo, hi int) {
	for u := lo; u < hi; u++ {
		for k := range g {
			q := g[k].n / units
			for at := u * q; at < (u+1)*q; at++ {
				d.words(a, b, &g[k], at)
				g[k].code.Call(a)
			}
		}
	}
}

// chainRegion runs the units [lo, hi) of the phase's kernels on worker w's
// words.
func (r *phase) chainRegion(w, lo, hi int) {
	r.d.chainRun(&r.d.args[w], r.b, r.g, r.units, lo, hi)
}

// groupRun runs the calls [lo, hi) of the kernels g's one space on a.
func (d *Device) groupRun(a *cpu.Args, b *Call, g []Code, lo, hi int) {
	ends := d.ends
	k := 0
	for i := lo; i < hi; i++ {
		for i >= ends[k] {
			k++
		}
		at := i
		if k > 0 {
			at -= ends[k-1]
		}
		d.words(a, b, &g[k], at)
		if d.Prof {
			d.profCall(&g[k], a)
			continue
		}
		g[k].code.Call(a)
	}
}

// words writes call at's words of kernel k into a.
func (d *Device) words(a *cpu.Args, b *Call, k *Code, at int) {
	base := unsafe.Pointer(a)
	rows, _ := k.share.ShareRows(d.part)
	for i := range k.args {
		ar := &k.args[i]
		var p unsafe.Pointer
		switch ar.src {
		case SrcValue:
			*(*uint64)(unsafe.Add(base, ar.field)) = ar.val
			continue
		case SrcShare:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(min(rows, k.share.Share-at*rows))
			continue
		case SrcCoord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(k.coord(at, int(ar.coord), rows))
			continue
		case SrcWord:
			*(*uint64)(unsafe.Add(base, ar.field)) = uint64(b.Own[ar.i][ar.off])
			continue
		case SrcX:
			p = unsafe.Pointer(&b.X[ar.stream*b.Stride])
		case SrcIO:
			p = b.IO[ar.i]
		case SrcOwn:
			p = unsafe.Pointer(&b.Own[ar.i][0])
		case SrcWeight, SrcVector:
			p = b.Weights[ar.i]
		case SrcRope:
			p = b.Rope[ar.inst]
		case SrcPlanes:
			p = b.Planes[ar.inst]
		default:
			p = b.At[ar.src]
		}
		off := ar.off
		if ar.moves {
			for ax := range k.calls {
				if ax < len(ar.per) && ar.per[ax] != 0 {
					off += k.coord(at, ax, rows) * ar.per[ax]
				}
			}
		}
		*(*uint64)(unsafe.Add(base, ar.field)) = uint64(uintptr(p)) + uint64(off*ar.elem)
	}
}

// Profile is a kernel's launches as Device.Prof counts them: its name and
// what it writes, its calls' space and count, whether they run one at a
// time, its code's bytes, how many kernels after it its phase joins or
// chains, and its time and launches, then its calls' count, time, start
// after the launch's and each launch's last start.
type Profile struct {
	Name                              string
	Calls                             []int
	N                                 int
	Serial                            bool
	Bytes                             int
	Join, Chain                       int
	Nanos, Runs                       int64
	CallCount, CallNs, StartNs, MaxNs int64
}

// Profile is kernel c's launches as Device.Prof counted them.
func (c *Code) Profile() Profile {
	return Profile{Name: c.name + c.desc, Calls: c.calls, N: c.n, Serial: c.serial, Bytes: c.code.Size, Join: c.join, Chain: c.chain,
		Nanos: c.nanos, Runs: c.runs, CallCount: c.pcalls, CallNs: c.pcallNs, StartNs: c.pstartNs, MaxNs: c.pmaxNs}
}

// ResetProfile zeroes the counters of every kernel of f.
func (f *Form) ResetProfile() {
	for i := range f.Codes {
		c := &f.Codes[i]
		c.nanos, c.runs, c.pcalls, c.pcallNs, c.pstartNs, c.pmaxNs = 0, 0, 0, 0, 0, 0
	}
}

// Layout is kernel c's weight and coordinate words: each one's source,
// plane, steps along its calls, element bytes and offset.
func (c *Code) Layout() string {
	var b strings.Builder
	for _, a := range c.args {
		if a.src == SrcWeight || a.src == SrcCoord {
			fmt.Fprintf(&b, " [src %d region %d per %v elem %d off %d]", a.src, a.region, a.per, a.elem, a.off)
		}
	}
	return b.String()
}
