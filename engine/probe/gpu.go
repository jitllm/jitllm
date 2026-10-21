package probe

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/metal"
)

// GPU measures device d: what a launch costs the host and a kernel the
// device beyond its work, the read rate of all its threads, its nearest
// cache's size on one multiprocessor and its last cache's size, each one's
// rate and latency, its memory's latency, how many loads a thread keeps in
// flight in a loop using each as it goes, and its float rate and delay.
// No per-thread read rate: a thread's loads are bound by their latency,
// which the estimate takes from the kernel's loads themselves. The caches
// are found from one thread's chase over sets growing by halves: the
// nearest, a multiprocessor's own, is the largest set under twice the
// smallest set's latency, the last the largest past it under the midpoint
// of its latency and the memory's; its latency is the median over the sets
// past twice the nearest. Read rates cannot find them: the
// multiprocessors' nearest caches together can hold more than the last.
// The last's rate is every workgroup reading one set of half its size, so
// each multiprocessor reads more than its own cache holds. The last cache is found
// from the read rate over sets doubling in size: the nearest caches serve
// near the peak, the last more than two and a half times the memory's
// rate; its rate is read at half its size, its latency the shortest of a
// chase over sets past the nearest caches. Reads of the whole memory warm
// the device first, and on a device with its own clock a read of the whole
// memory goes before each timed launch of an instruction class, whose
// values live in registers, so a measurement of one thread runs at the
// clock a loaded device runs at. Every measuring kernel is
// generated IR, compiled for the device in front of it, and timed on the
// device's clock where the backend has one.
func GPU(d backend.Device) (jit.Device, error) {
	const lanes, width = 32, 256
	out := jit.Device{Kind: jit.DeviceGPU, Slots: d.Slots(), Lanes: lanes}
	if out.Slots == 0 {
		out.Slots = defaultSlots
	}
	if u, ok := d.(backend.MultiprocessorDevice); ok {
		out.Units = u.Multiprocessors()
	}
	threads := out.Slots
	p := &gpuProbe{d: d}
	defer p.close()

	empty := ir.New("probe_empty", [3]int{width, 1, 1})
	empty.Param("p", ir.F32)
	scratch, err := p.alloc(4 * width)
	if err != nil {
		return out, err
	}
	if out.Floor, err = p.time(empty.Done(), 1, width, 64, scratch); err != nil {
		return out, err
	}
	if out.Launch, err = p.wall(empty.Done(), 1, width, 256, scratch); err != nil {
		return out, err
	}

	free, _, err := d.Mem()
	if err != nil {
		return out, err
	}
	big := int64(1) << 20
	for big*2 <= min(int64(free)/4, 1<<30) {
		big *= 2
	}
	in, err := p.alloc(int(big))
	if err != nil {
		return out, err
	}
	sums, err := p.alloc(4 * threads)
	if err != nil {
		return out, err
	}
	reads := func(set int64, threads int, whole bool) (float64, error) {
		traffic := max(set, 256<<20)
		if threads <= width {
			traffic = 16 << 20
		}
		per := max(1, int(traffic/4)/threads)
		k := gpuReadKernel(fmt.Sprintf("probe_read_%d_%d_%v", threads, set, whole), threads, per, int(set/4-1), width, whole)
		secs, err := p.time(k, max(1, threads/width), min(width, threads), 8, in, sums)
		return float64(per*threads*4) / secs, err
	}
	read := func(set int64, threads int) (float64, error) { return reads(set, threads, false) }
	for range 8 {
		if _, err := read(big, threads); err != nil {
			return out, err
		}
	}
	if out.Bandwidth, err = read(big, threads); err != nil {
		return out, err
	}
	warm := gpuReadKernel("probe_warm", threads, max(1, int(big/4)/threads), int(big/4-1), width, false)
	wk, err := p.compile(warm)
	if err != nil {
		return out, err
	}
	loaded := &warmLaunch{k: wk, groups: max(1, threads/width), width: min(width, threads), bufs: []backend.Buf{in, sums}}

	if out.Latency, err = p.chase(big/2, in); err != nil {
		return out, err
	}
	peak := out.Bandwidth
	for set := int64(64 << 10); set <= 1<<20; set *= 2 {
		r, err := read(set, threads)
		if err != nil {
			return out, err
		}
		peak = max(peak, r)
	}
	type point struct {
		set int64
		lat float64
	}
	var sweep []point
	for set := int64(8 << 10); set <= min(big/4, 64<<20); set *= 2 {
		for _, x := range []int64{set, set + set/2} {
			lat, err := p.chase(x, in)
			if err != nil {
				return out, err
			}
			sweep = append(sweep, point{x, lat})
		}
	}
	var l1, l2 int64
	l2lat := math.Inf(1)
	for _, pt := range sweep {
		if pt.lat < 2*sweep[0].lat {
			l1 = pt.set
		} else {
			l2lat = min(l2lat, pt.lat)
		}
	}
	for _, pt := range sweep {
		if pt.set > l1 && pt.lat < (l2lat+out.Latency)/2 {
			l2 = pt.set
		}
	}
	if l1 > 0 {
		lat, err := p.chase(l1/2, in)
		if err != nil {
			return out, err
		}
		alone, err := read(8<<10, 1)
		if err != nil {
			return out, err
		}
		out.Overlap = max(1, lat*alone/4)
		out.Caches[0] = jit.Cache{Bytes: l1, Rate: jit.Rate{Total: peak}, Latency: lat, Kernel: true}
	}
	var plateau []float64
	for _, pt := range sweep {
		if pt.set > 2*l1 && pt.set <= l2 {
			plateau = append(plateau, pt.lat)
		}
	}
	if len(plateau) > 0 {
		slices.Sort(plateau)
		l2lat = plateau[len(plateau)/2]
	}
	if l2 > 0 && !math.IsInf(l2lat, 1) {
		rate, err := reads(max(2*l1, l2/2), threads, true)
		if err != nil {
			return out, err
		}
		out.Caches[1] = jit.Cache{Bytes: l2, Rate: jit.Rate{Total: rate}, Latency: l2lat}
	}

	ops := map[jit.Class]gpuOp{
		jit.ClassFloat: {t: ir.F32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value {
			return b.Fma(x, b.ConstF32(0.999), b.ConstF32(0.001))
		}},
		jit.ClassSpecial: {t: ir.F32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value { return b.Exp(x) }},
		jit.ClassInt: {t: ir.U32, per: 2, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value {
			return b.Xor(ir.U32, x, b.Shr(ir.U32, x, b.Const(ir.U32, 1)))
		}},
		jit.ClassIndex: {t: ir.U32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value {
			return b.Add(ir.U32, b.Mul(ir.U32, x, x), b.Const(ir.U32, 1))
		}},
		jit.ClassLanes: {t: ir.F32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value { return b.ShuffleXor(ir.F32, x, 1) }},
		jit.ClassGroup: {t: ir.F32, op: groupSum, shared: true},
	}
	p.warm = loaded
	defer func() { p.warm = nil }()
	for cl, op := range ops {
		rate := func(chains, deep, threads int) (float64, error) {
			trips := 1 << 12
			if threads <= width {
				trips = 1 << 15
			}
			if cl == jit.ClassGroup {
				trips /= 16
			}
			k := gpuOpKernel(fmt.Sprintf("probe_op%d_%d_%d_%d", cl, chains, deep, threads), chains, deep, trips, min(width, threads), op)
			secs, err := p.time(k, max(1, threads/width), min(width, threads), 4, sums)
			return max(1, op.per) * float64(chains*deep*trips*threads) / secs, err
		}
		chains := 8
		if cl == jit.ClassGroup {
			chains = 1
		}
		total, err := rate(chains, 1, threads)
		if err != nil {
			return out, err
		}
		alone := width
		if cl != jit.ClassGroup && cl != jit.ClassLanes {
			alone = 1
		}
		one, err := rate(chains, 1, alone)
		if err != nil {
			return out, err
		}
		chain, err := rate(1, 8, alone)
		if err != nil {
			return out, err
		}
		out.Rates[cl] = jit.Rate{Thread: one / float64(alone), Total: total}
		out.Delay[cl] = float64(alone) / chain
	}
	out.Delay[jit.ClassMem] = out.Caches[0].Latency
	return out, nil
}

// defaultSlots is the resident threads taken for a device whose runtime
// will not say (Vulkan, Metal): the figure the device tier takes for it.
const defaultSlots = 16384

// gpuProbe holds the device the probe measures and what it allocated, and
// the launch that keeps it loaded between timed ones.
type gpuProbe struct {
	d    backend.Device
	bufs []backend.Buf
	ks   []backend.Kernel
	warm *warmLaunch
}

// warmLaunch is a kernel reading the whole memory, launched before each
// timed launch so a measurement of one thread starts at the clock a loaded
// device runs at rather than the one an idle device falls to.
type warmLaunch struct {
	k             backend.Kernel
	groups, width int
	bufs          []backend.Buf
}

func (p *gpuProbe) close() {
	for _, b := range p.bufs {
		b.Free()
	}
	for _, k := range p.ks {
		k.Close()
	}
}

func (p *gpuProbe) alloc(n int) (backend.Buf, error) {
	b, err := p.d.Alloc(n)
	if err != nil {
		return nil, err
	}
	p.bufs = append(p.bufs, b)
	return b, b.Write(make([]byte, n))
}

func (p *gpuProbe) compile(k *ir.Kernel) (backend.Kernel, error) {
	c, err := p.d.Compile(k)
	if err != nil {
		return nil, err
	}
	p.ks = append(p.ks, c)
	return c, nil
}

// time is the device seconds a launch of k takes, after one that warms it:
// the mean of n on the device's own clock where its backend has one, else
// the median of the wall's around each waited launch.
func (p *gpuProbe) time(k *ir.Kernel, groups, width, n int, bufs ...backend.Buf) (float64, error) {
	c, err := p.compile(k)
	if err != nil {
		return 0, err
	}
	if err := c.Launch(groups, width, bufs...); err != nil {
		return 0, err
	}
	if t, ok := Timer(p.d); ok {
		return p.timed(t, k.Name, c, groups, width, n, bufs...)
	}
	var times []float64
	for range n {
		t0 := time.Now()
		if err := c.Launch(groups, width, bufs...); err != nil {
			return 0, err
		}
		times = append(times, time.Since(t0).Seconds())
	}
	slices.Sort(times)
	return times[len(times)/2], nil
}

// KernelTimer is a backend's per-kernel device clock: arming it, and each
// kernel's summed microseconds and launches since the last read, by name.
type KernelTimer struct {
	Arm   func(bool)
	Times func() (us map[string]float64, n map[string]int)
}

// Timer is device d's per-kernel clock, where its backend has one: CUDA's
// event pairs, Metal's command buffers' own stamps.
func Timer(d backend.Device) (KernelTimer, bool) {
	switch d.API() {
	case "ptx":
		return KernelTimer{backend.SetCUDAKernelTiming, backend.CUDAKernelTimes}, true
	case "msl":
		return KernelTimer{metal.SetKernelTiming, metal.KernelTimes}, true
	}
	return KernelTimer{}, false
}

// timed is the mean device seconds of n launches of c in one session, on
// clock t.
func (p *gpuProbe) timed(t KernelTimer, name string, c backend.Kernel, groups, width, n int, bufs ...backend.Buf) (float64, error) {
	t.Arm(true)
	defer t.Arm(false)
	t.Times()
	var errs error
	p.d.Session(func(s backend.Session) {
		for range n {
			if w := p.warm; w != nil {
				if err := s.Launch(w.k, w.groups, w.width, w.bufs...); err != nil {
					errs = err
					return
				}
			}
			if err := s.Launch(c, groups, width, bufs...); err != nil {
				errs = err
				return
			}
		}
		errs = s.Sync()
	})
	if errs != nil {
		return 0, errs
	}
	us, cnt := t.Times()
	if cnt[name] == 0 {
		return 0, fmt.Errorf("probe: %s was not timed", name)
	}
	return us[name] / float64(cnt[name]) / 1e6, nil
}

// wall is the seconds a launch of k costs n back to back in one session:
// the rate the host issues launches at, where the kernel takes less.
func (p *gpuProbe) wall(k *ir.Kernel, groups, width, n int, bufs ...backend.Buf) (float64, error) {
	c, err := p.compile(k)
	if err != nil {
		return 0, err
	}
	var errs error
	var secs float64
	p.d.Session(func(s backend.Session) {
		if errs = s.Sync(); errs != nil {
			return
		}
		t0 := time.Now()
		for range n {
			if errs = s.Launch(c, groups, width, bufs...); errs != nil {
				return
			}
		}
		errs = s.Sync()
		secs = time.Since(t0).Seconds()
	})
	return secs / float64(n), errs
}

// chase is the seconds one read waits, a single thread walking a random
// cycle through the 128-byte lines of set bytes of buf.
func (p *gpuProbe) chase(set int64, buf backend.Buf) (float64, error) {
	const line = 128 / 4
	lines := int(max(2, set/128))
	words := make([]uint32, lines*line)
	order := rand.Perm(lines)
	for i, at := range order {
		words[at*line] = uint32(order[(i+1)%lines] * line)
	}
	if err := buf.Write(unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), len(words)*4)); err != nil {
		return 0, err
	}
	const steps = 1 << 16
	b := ir.New(fmt.Sprintf("probe_chase_%d", lines), [3]int{1, 1, 1})
	in, out := b.Param("in", ir.U32), b.Param("out", ir.U32)
	first := b.Const(ir.U32, int64(order[0]*line))
	b.Loop(steps)
	at := b.Phi(ir.U32, first)
	b.SetPhi(at, b.Load(ir.U32, in, at, 0))
	b.EndLoop()
	b.Store(out, b.Const(ir.U32, 0), at, 0)
	sink, err := p.alloc(4)
	if err != nil {
		return 0, err
	}
	secs, err := p.time(b.Done(), 1, 1, 3, buf, sink)
	return secs / steps, err
}

// gpuReadKernel sums, in each of threads threads, per words of the first
// buffer strided by threads and wrapped by mask, a power of two less one,
// so a set smaller than the reads is read over and over; and stores each
// sum. Where whole is set each workgroup reads the whole set, strided by
// its width.
func gpuReadKernel(name string, threads, per, mask, width int, whole bool) *ir.Kernel {
	b := ir.New(name, [3]int{min(width, threads), 1, 1})
	in, out := b.Param("in", ir.F32), b.Param("out", ir.F32)
	t := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	first, stride := t, threads
	if whole {
		first, stride = b.TID(), min(width, threads)
	}
	zero := b.ConstF32(0)
	b.Loop(int64(per))
	j := b.Phi(ir.U32, first)
	acc := b.Phi(ir.F32, zero)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Load(ir.F32, in, j, 0)))
	b.SetPhi(j, b.And(ir.U32, b.Add(ir.U32, j, b.Const(ir.U32, int64(stride))), b.Const(ir.U32, int64(mask))))
	b.EndLoop()
	b.Store(out, t, acc, 0)
	return b.Done()
}

// gpuOp is instructions of a class a probe chains, ones the device's
// compiler cannot fold across a chain (a multiply-add of the value by
// itself, a shift and an exclusive or, not an add of a constant): its
// value's type, the instructions applied to a value, given the workgroup
// memory the kernel declares where shared is set and the thread's index,
// and how many instructions that is where more than one. The integer class
// is a shift and an exclusive or, the index class a multiply-add: an
// integer multiply is slower than the rest on some devices.
type gpuOp struct {
	t      ir.Type
	op     func(b *ir.Builder, sh, tid, x ir.Value) ir.Value
	shared bool
	per    float64
}

// groupSum is a workgroup's sum the way the composer's GPU lowering forms
// one: each subgroup's butterfly of shuffles, the subgroups' sums through
// workgroup memory and a barrier, a butterfly over them.
func groupSum(b *ir.Builder, sh, tid, x ir.Value) ir.Value {
	butterfly := func(v ir.Value) ir.Value {
		for m := ir.SubgroupLanes / 2; m >= 1; m /= 2 {
			v = b.Add(ir.F32, v, b.ShuffleXor(ir.F32, v, int64(m)))
		}
		return v
	}
	b.Store(sh, b.Shr(ir.U32, tid, b.Const(ir.U32, 5)), butterfly(x), 0)
	b.Barrier()
	return butterfly(b.Load(ir.F32, sh, b.And(ir.U32, tid, b.Const(ir.U32, ir.SubgroupLanes-1)), 0))
}

// gpuOpKernel runs, in each thread of workgroups of width, chains
// independent chains of trips of op, deep of them a chain a trip, and
// stores their sum.
func gpuOpKernel(name string, chains, deep, trips, width int, op gpuOp) *ir.Kernel {
	b := ir.New(name, [3]int{width, 1, 1})
	out := b.Param("out", ir.F32)
	var sh ir.Value
	if op.shared {
		sh = b.Shared("probe_group", ir.F32, width)
	}
	tid := b.TID()
	t := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), tid)
	inits := make([]ir.Value, chains)
	for i := range inits {
		x := b.Add(ir.U32, t, b.Const(ir.U32, int64(i)))
		if op.t == ir.F32 {
			x = b.Mul(ir.F32, b.CvtF32(x), b.ConstF32(1e-6))
		}
		inits[i] = x
	}
	b.Loop(int64(trips))
	accs := make([]ir.Value, chains)
	for i := range accs {
		accs[i] = b.Phi(op.t, inits[i])
	}
	for _, acc := range accs {
		next := acc
		for range deep {
			next = op.op(b, sh, tid, next)
		}
		b.SetPhi(acc, next)
	}
	b.EndLoop()
	sum := accs[0]
	for _, acc := range accs[1:] {
		sum = b.Add(op.t, sum, acc)
	}
	if op.t != ir.F32 {
		sum = b.CvtF32(sum)
	}
	b.Store(out, t, sum, 0)
	return b.Done()
}
