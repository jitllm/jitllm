package probe

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// GPU measures device d: what a launch costs the host and a kernel the
// device beyond its work, both after reads of the whole memory have woken
// it, the read rate of all its threads, its nearest
// cache's size on one multiprocessor and its last cache's size, each one's
// rate and latency, its memory's latency, how many loads a thread keeps in
// flight in a loop using each as it goes, and its float rate and delay.
// No per-thread read rate: a thread's loads are bound by their latency,
// which the estimate takes from the kernel's loads themselves. The nearest
// cache, a multiprocessor's own, is found from one thread's chase over sets
// growing by halves: the largest set under twice the smallest set's
// latency. The last cache is the size the runtime states where it states
// one (CUDA), else the largest set past the nearest under the midpoint of
// its latency and the memory's; its latency is the median over the sets
// past twice the nearest and within the TLB's reach. Read rates cannot find them: the
// multiprocessors' nearest caches together can hold more than the last.
// The last's rate is every warp streaming one set of half its size from a
// phase of its own, so each multiprocessor reads more than its own cache
// holds. Reads of the
// whole memory warm the device first, and on a device with its own clock a
// read of the whole memory goes before each timed launch of an instruction
// class, so a measurement of one thread runs at the clock a loaded device
// runs at. A launch is the composer's smallest form and a read of the
// whole memory its column reduction, each formed for the device in front
// of it as a model's part is. The device's rate of a class an expression
// issues (gpuStep) is the composer's form of rows through chains of the
// class's terms, the instructions of the class the estimator counts in it
// over the seconds, taken between two depths so a launch's own cost
// cancels. A thread's rate, a chain's delay and the chase are one row's
// iteration, a loop the code keeps, timed between two trip counts: a
// thread alone running straight code long enough to time stalls on
// fetching its instructions. A read of a set smaller than its traffic is an
// iteration too, each column's index wrapped at the set and its sum a
// float's. The last cache's rate is every warp streaming the set from a
// phase of its own, so a multiprocessor's warps together read more than its
// nearest cache holds. The lanes' and the workgroup's exchanges are still
// kernels built from IR. Each is timed on the device's clock where the
// backend has one.
func GPU(d backend.Device) (cost.Device, error) {
	const lanes, width = 32, 256
	out := cost.Device{Device: composer.Device{Kind: composer.DeviceGPU, Slots: d.Slots(), Lanes: lanes, ExactDiv: composer.ExactDivOn(d.API())}}
	if out.Slots == 0 {
		out.Slots = defaultSlots
	}
	if u, ok := d.(backend.MultiprocessorDevice); ok {
		out.Units = u.Multiprocessors()
	}
	threads := out.Slots
	p := newGPUProbe(d)
	defer p.close()

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
	reads := func(set int64, threads int, whole bool) (float64, error) {
		traffic := max(set, 256<<20)
		if threads <= width {
			traffic = 16 << 20
		}
		f, words, err := p.read(fmt.Sprintf("probe_read_%d_%d_%v", threads, set, whole), in, set, traffic, threads, whole)
		if err != nil {
			return 0, err
		}
		secs, err := p.time(f, 8)
		return float64(4*words) / secs, err
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
	loaded, _, err := p.read("probe_warm", in, big, big, threads, false)
	if err != nil {
		return out, err
	}
	empty, err := p.empty()
	if err != nil {
		return out, err
	}
	if out.Floor, err = p.time(empty, 64); err != nil {
		return out, err
	}
	if out.Launch, err = p.wall(empty, 256); err != nil {
		return out, err
	}

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
	if st, ok := d.(backend.L2Device); ok && st.L2Bytes() > 0 {
		l2 = int64(st.L2Bytes())
	} else {
		for _, pt := range sweep {
			if pt.set > l1 && pt.lat < (l2lat+out.Latency)/2 {
				l2 = pt.set
			}
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
		out.Caches[0] = cost.Cache{Bytes: l1, Rate: cost.Rate{Total: peak}, Latency: lat, Kernel: true}
	}
	var plateau []float64
	for _, pt := range sweep {
		if pt.set > 2*l1 && pt.set <= min(l2, tlbReach) {
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
		out.Caches[1] = cost.Cache{Bytes: l2, Rate: cost.Rate{Total: rate}, Latency: l2lat}
	}

	sums, err := p.alloc(4 * threads)
	if err != nil {
		return out, err
	}
	p.warm = loaded
	defer func() { p.warm = nil }()
	for cl, op := range gpuOps() {
		rate := func(chains, deep, threads int) (float64, error) {
			trips := 1 << 12
			if threads <= width {
				trips = 1 << 15
			}
			if cl == cost.ClassGroup {
				trips /= 16
			}
			k, err := gpuOpKernel(fmt.Sprintf("probe_op%d_%d_%d_%d", cl, chains, deep, threads), chains, deep, trips, min(width, threads), op)
			if err != nil {
				return 0, err
			}
			f, err := p.kernel(k, max(1, threads/width), min(width, threads), sums)
			if err != nil {
				return 0, err
			}
			secs, err := p.time(f, 4)
			return max(1, op.per) * float64(chains*deep*trips*threads) / secs, err
		}
		chains := 8
		if cl == cost.ClassGroup {
			chains = 1
		}
		var total float64
		if _, _, ok := gpuStep(cl); ok {
			total, err = p.slope(cl, 16*threads, chains, 16, 64)
		} else {
			total, err = rate(chains, 1, threads)
		}
		if err != nil {
			return out, err
		}
		alone := width
		if cl != cost.ClassGroup {
			alone = lanes
		}
		var one, chain float64
		if _, _, ok := gpuStep(cl); ok {
			// A thread's rate and a chain's delay: a subgroup of rows
			// through an iteration, timed between two trip counts.
			iter := func(chains, deep int) (float64, error) {
				return p.slopeOf(cl, func(trips int) (*graph.Graph, bool) { return iterateGraph(cl, alone, chains, deep, trips) }, 1<<10, 1<<13)
			}
			if one, err = iter(chains, 1); err != nil {
				return out, err
			}
			if chain, err = iter(1, 8); err != nil {
				return out, err
			}
		} else {
			if one, err = rate(chains, 1, alone); err != nil {
				return out, err
			}
			if chain, err = rate(1, 8, alone); err != nil {
				return out, err
			}
		}
		out.Rates[cl] = cost.Rate{Thread: one / float64(alone), Total: total}
		out.Delay[cl] = float64(alone) / chain
	}
	out.Delay[cost.ClassMem] = out.Caches[0].Latency
	return out, nil
}

// defaultSlots is the resident threads taken for a device whose runtime
// will not say (Vulkan, Metal): the figure the device tier takes for it.
const defaultSlots = 16384

// tlbReach is the largest set a chase measures a cache's latency over: past
// a few megabytes a random walk waits for page walks as much as for the
// cache (an A100's or an H100's L2 read 96 MiB and 256 KiB that way).
const tlbReach = 4 << 20

// gpuProbe holds the device the probe measures, what it allocated, the
// composer it forms its kernels with and the device it forms them for, and
// the launch that keeps the device loaded between timed ones.
type gpuProbe struct {
	d    backend.Device
	reg  *composer.Registry
	dev  composer.Device
	bufs []backend.Buf
	ks   []backend.Kernel
	warm *gpuForm
	// walks are the chase's two forms, over walked words of walkOn, its
	// first word read from start.
	walks  [2]*gpuForm
	walked int
	walkOn backend.Buf
	start  backend.Buf
}

// gpuForm is kernels launched in order, each with its workgroups, width and
// buffers: a graph the composer formed, compiled.
type gpuForm struct {
	issued [cost.NumClasses]int64
	ks     []backend.Kernel
	names  []string
	groups []int
	widths []int
	args   [][]backend.Buf
}

func newGPUProbe(d backend.Device) *gpuProbe {
	lanes := 1
	if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); ok {
		lanes = ir.SubgroupLanes
	}
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: d.Slots(), Lanes: lanes, ExactDiv: composer.ExactDivOn(d.API()),
		ExactArith: composer.ExactArithOn(d.API()), VectorLoads: composer.VectorLoadsOn(d.API()), Buffers: backend.Bindings(d)}
	if u, ok := d.(backend.MultiprocessorDevice); ok {
		dev.Units = u.Multiprocessors()
	}
	if s, ok := d.(backend.SharedMemoryDevice); ok {
		dev.Shared = s.SharedPerMultiprocessor()
	}
	return &gpuProbe{d: d, reg: composer.Default(), dev: dev}
}

func (p *gpuProbe) close() {
	for _, b := range p.bufs {
		b.Free()
	}
	for _, k := range p.ks {
		k.Close()
	}
}

// words is a buffer of its own holding ws.
func (p *gpuProbe) words(ws []uint32) (backend.Buf, error) {
	b, err := p.alloc(max(4*len(ws), 4))
	if err != nil {
		return nil, err
	}
	return b, b.Write(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(ws))), 4*len(ws)))
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

// kernel is k compiled, launched at groups workgroups of width over bufs.
func (p *gpuProbe) kernel(k *ir.Kernel, groups, width int, bufs ...backend.Buf) (*gpuForm, error) {
	c, err := p.compile(k)
	if err != nil {
		return nil, err
	}
	return &gpuForm{ks: []backend.Kernel{c}, names: []string{k.Name}, groups: []int{groups}, widths: []int{width}, args: [][]backend.Buf{bufs}}, nil
}

// form is g's kernels as the composer forms them for the device at one row,
// compiled, each value bound to its buffer in vals or one of its own, and
// each constant block, scratch and the step's row count to one of its own.
func (p *gpuProbe) form(g *graph.Graph, vals map[graph.ValueID]backend.Buf) (*gpuForm, error) {
	req := p.request(g)
	ks, codes, err := p.reg.FormKernels(req)
	if err != nil {
		return nil, err
	}
	own := map[composer.Operand]backend.Buf{}
	f := &gpuForm{}
	for i, c := range codes {
		n := cost.Issued(ks[i], req.Step, cost.Device{Device: p.dev})
		for cl := range n {
			f.issued[cl] += n[cl]
		}
		if c.GPU == nil {
			return nil, fmt.Errorf("probe: %s has no GPU code", c.Name)
		}
		k, err := p.compile(c.GPU)
		if err != nil {
			return nil, err
		}
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			b, ok := own[o]
			switch {
			case ok:
			case o.Kind == composer.OperandValue && vals[o.Value] != nil:
				b = vals[o.Value]
			case o.Kind == composer.OperandValue:
				b, err = p.alloc(4 * max(g.Values[o.Value].Width, 1))
			case bd.Words != nil:
				b, err = p.words(bd.Words)
			case o.Kind == composer.OperandTable && o.Name == "rows.n":
				b, err = p.words([]uint32{1})
			case o.Kind == composer.OperandScratch:
				b, err = p.alloc(int(max(bd.Bytes, 4)))
			default:
				return nil, fmt.Errorf("probe: %s binds %v, which a probe does not hold", c.Name, o)
			}
			if err != nil {
				return nil, err
			}
			own[o], args[i] = b, b
		}
		f.ks, f.names = append(f.ks, k), append(f.names, c.GPU.Name)
		f.groups, f.widths, f.args = append(f.groups, c.Groups), append(f.widths, c.GPU.Group[0]), append(f.args, args)
	}
	return f, nil
}

// slope is the instructions of class cl a second a row of n issues through
// chains chains, from the chains at two depths, short and long: what a
// launch costs beside its chains cancels in the difference. A chain is
// straight code, its launches back to back with no read of the memory
// between them: a read would take the code out of the last cache, and a
// thread alone stalls on fetching it. The device's clock times them where
// its backend has one, else the wall of a session's launches.
func (p *gpuProbe) slope(cl cost.Class, n, chains, short, long int) (float64, error) {
	return p.slopeOf(cl, func(deep int) (*graph.Graph, bool) { return rateGraph(cl, n, chains, deep) }, short, long)
}

// slopeOf is the instructions of class cl a second between the forms of
// graphAt at two sizes, short and long, as slope takes them.
func (p *gpuProbe) slopeOf(cl cost.Class, graphAt func(int) (*graph.Graph, bool), short, long int) (float64, error) {
	warm := p.warm
	p.warm = nil
	defer func() { p.warm = warm }()
	_, clocked := Timer(p.d)
	var count, secs [2]float64
	for i, size := range [2]int{short, long} {
		g, ok := graphAt(size)
		if !ok {
			return 0, fmt.Errorf("probe: no chain of the %v class", cl)
		}
		f, err := p.form(g, nil)
		if err != nil {
			return 0, err
		}
		count[i] = f.count(countedAs(cl))
		if clocked {
			secs[i], err = p.time(f, 16)
		} else {
			secs[i], err = p.wall(f, 64)
		}
		if err != nil {
			return 0, err
		}
	}
	if !(count[1] > count[0]) || !(secs[1] > secs[0]) {
		return 0, fmt.Errorf("probe: %v chains of %d and %d count %v and took %v seconds", cl, short, long, count, secs)
	}
	return (count[1] - count[0]) / (secs[1] - secs[0]), nil
}

// request asks for g at one row on the device the probe measures, its
// inputs and outputs buffers of their own.
func (p *gpuProbe) request(g *graph.Graph) composer.Request {
	return composer.Request{Graph: g, Formats: map[jlm.Role]jlm.Type{}, Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: p.dev, Apart: true}
}

// empty is the smallest form: one element, doubled.
func (p *gpuProbe) empty() (*gpuForm, error) {
	b := graph.New()
	x := b.In("probe_empty_in", 1)
	b.Out(b.Map("probe_empty", graph.Attrs{Fn: graph.MapMul, Const: 2}, nil, 1, x))
	return p.form(b.G, nil)
}

// read is a column reduction of set bytes of buf, as threads columns of at
// least traffic bytes in all, and the words it reads: column j sums the
// set's floats j, j+threads, ... wrapped at the set, so a set smaller than
// the reads is read over and over. Where whole is set each warp's columns
// stream the whole set from a phase of their own (streamStarts). A set read
// once is the composer's reduction; a set the reads wrap is an iteration per
// column carrying its index and its sum, since a reduction over a tile is a
// kernel of its own, which reads the set once and writes every period of
// it.
func (p *gpuProbe) read(name string, buf backend.Buf, set, traffic int64, threads int, whole bool) (*gpuForm, int, error) {
	words := int(set / 4)
	per := max(1, int(traffic/4)/threads)
	n := per * threads
	if whole || n > words {
		stride, wrap, at := threads, words, make([]uint32, threads)
		for c := range at {
			at[c] = uint32(c)
		}
		if whole {
			// The stream wraps at the set's largest power of two.
			for wrap = 1; 2*wrap <= words; wrap *= 2 {
			}
			stride, at = 32, streamStarts(threads, wrap)
		}
		starts, err := p.words(at)
		if err != nil {
			return nil, 0, err
		}
		g, tab, first := wrapGraph(words, threads, per, stride, wrap-1)
		f, err := p.form(g, map[graph.ValueID]backend.Buf{tab: buf, first: starts})
		return f, n, err
	}
	b := graph.New()
	x := b.In(name+"_set", n)
	b.Out(b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: name, Width: threads}}, nil,
		graph.Attrs{Fn: graph.MapAdd, HeadDim: threads, Span: per, Order: graph.OrderSeq}, nil, nil)[0])
	f, err := p.form(b.G, map[graph.ValueID]backend.Buf{x: buf})
	return f, n, err
}

// count is the instructions of class cl one launch of f issues.
func (f *gpuForm) count(cl cost.Class) float64 { return float64(f.issued[cl]) }

func (f *gpuForm) run(s backend.Session) error {
	for i, k := range f.ks {
		if err := s.Launch(k, f.groups[i], f.widths[i], f.args[i]...); err != nil {
			return err
		}
	}
	return nil
}

func (f *gpuForm) launch() error {
	for i, k := range f.ks {
		if err := k.Launch(f.groups[i], f.widths[i], f.args[i]...); err != nil {
			return err
		}
	}
	return nil
}

// time is the device seconds a launch of f takes, after one that warms it:
// the mean of n on the device's own clock where its backend has one, else
// the median of the wall's around each waited launch.
func (p *gpuProbe) time(f *gpuForm, n int) (float64, error) {
	if err := f.launch(); err != nil {
		return 0, err
	}
	if t, ok := Timer(p.d); ok {
		return p.timed(t, f, n)
	}
	var times []float64
	for range n {
		t0 := time.Now()
		if err := f.launch(); err != nil {
			return 0, err
		}
		times = append(times, time.Since(t0).Seconds())
	}
	slices.Sort(times)
	return times[len(times)/2], nil
}

// ReadRate is device d's read rate over set bytes, in bytes a second: every
// thread reading its share of the set once, as GPU measures the device's
// Bandwidth, after reads of the whole set have woken it. It is the wall a
// decode row is a fraction of (RULE 1), measured in the process that runs it.
func ReadRate(d backend.Device, set int64, rounds int) (float64, error) {
	threads := d.Slots()
	if threads == 0 {
		threads = defaultSlots
	}
	p := newGPUProbe(d)
	defer p.close()
	in, err := p.alloc(int(set))
	if err != nil {
		return 0, err
	}
	f, words, err := p.read("probe_wall", in, set, set, threads, false)
	if err != nil {
		return 0, err
	}
	for range 4 {
		if _, err := p.time(f, 1); err != nil {
			return 0, err
		}
	}
	secs, err := p.time(f, max(rounds, 1))
	return float64(4*words) / secs, err
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
	if c, ok := d.(backend.KernelClock); ok {
		return KernelTimer{c.SetKernelTiming, c.KernelTimes}, true
	}
	return KernelTimer{}, false
}

// timed is the mean device seconds of n launches of f in one session, on
// clock t: each of its kernels' mean, summed.
func (p *gpuProbe) timed(t KernelTimer, f *gpuForm, n int) (float64, error) {
	if w := p.warm; w != nil {
		for _, name := range f.names {
			if slices.Contains(w.names, name) {
				return 0, fmt.Errorf("probe: %s is a kernel of the warm launch too", name)
			}
		}
	}
	t.Arm(true)
	defer t.Arm(false)
	t.Times()
	var errs error
	p.d.Session(func(s backend.Session) {
		for range n {
			if w := p.warm; w != nil {
				if errs = w.run(s); errs != nil {
					return
				}
			}
			if errs = f.run(s); errs != nil {
				return
			}
		}
		errs = s.Sync()
	})
	if errs != nil {
		return 0, errs
	}
	us, cnt := t.Times()
	secs := 0.0
	for _, name := range slices.Compact(slices.Sorted(slices.Values(f.names))) {
		if cnt[name] == 0 {
			return 0, fmt.Errorf("probe: %s was not timed", name)
		}
		secs += us[name] / float64(n) / 1e6
	}
	return secs, nil
}

// wall is the seconds a launch of f costs n back to back in one session:
// the rate the host issues launches at, where the kernel takes less; the
// median of rounds sessions.
func (p *gpuProbe) wall(f *gpuForm, n int) (float64, error) {
	var times []float64
	for range rounds {
		var errs error
		p.d.Session(func(s backend.Session) {
			if errs = s.Sync(); errs != nil {
				return
			}
			t0 := time.Now()
			for range n {
				if errs = f.run(s); errs != nil {
					return
				}
			}
			errs = s.Sync()
			times = append(times, time.Since(t0).Seconds()/float64(n))
		})
		if errs != nil {
			return 0, errs
		}
	}
	slices.Sort(times)
	return times[len(times)/2], nil
}

// chaseShort and chaseLong are the walk's two lengths: their difference is
// the steps a chase times.
const chaseShort, chaseLong = 1 << 12, 1 << 16

// chase is the seconds one read waits, one row walking a random cycle
// through the 128-byte lines of set bytes of buf, between two walks'
// lengths. Its forms are made once, over the largest set it walks.
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
	if p.start == nil {
		start, err := p.alloc(4)
		if err != nil {
			return 0, err
		}
		p.start = start
	}
	first := []uint32{uint32(order[0] * line)}
	if err := p.start.Write(unsafe.Slice((*byte)(unsafe.Pointer(&first[0])), 4)); err != nil {
		return 0, err
	}
	if buf != p.walkOn || len(words) > p.walked {
		for i, trips := range [2]int{chaseShort, chaseLong} {
			g, tab, at := chaseGraph(len(words), trips)
			f, err := p.form(g, map[graph.ValueID]backend.Buf{tab: buf, at: p.start})
			if err != nil {
				return 0, err
			}
			p.walks[i] = f
		}
		p.walkOn, p.walked = buf, len(words)
	}
	var secs [2]float64
	for i, f := range p.walks {
		var err error
		if secs[i], err = p.time(f, 3); err != nil {
			return 0, err
		}
	}
	if !(secs[1] > secs[0]) {
		return 0, fmt.Errorf("probe: walks of %d and %d steps took %v seconds", chaseShort, chaseLong, secs)
	}
	return (secs[1] - secs[0]) / (chaseLong - chaseShort), nil
}

// gpuOp is instructions of a class a probe chains, ones the device's
// compiler cannot fold across a chain (a multiply-add of the value by
// itself, a shift and an exclusive or, not an add of a constant): its
// value's type, the instructions applied to a value, given the workgroup
// memory the kernel declares where shared is set and the thread's index,
// and how many instructions the estimator counts in it where more than
// one. The integer class is a shift and an exclusive or, the index class a
// multiply-add: an integer multiply is slower than the rest on some
// devices. The estimator counts a multiply-add of integers as a multiply
// and an add, as it counts the composed index chain whose total rate it
// takes, though a device issues it as one instruction.
type gpuOp struct {
	t      ir.Type
	op     func(b *ir.Builder, sh, tid, x ir.Value) ir.Value
	shared bool
	per    float64
}

// gpuOps is the instructions each class's rate is measured over.
func gpuOps() map[cost.Class]gpuOp {
	return map[cost.Class]gpuOp{
		cost.ClassFloat: {t: ir.F32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value {
			return b.Fma(x, b.ConstF32(0.999), b.ConstF32(0.001))
		}},
		cost.ClassSpecial: {t: ir.F32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value { return b.Exp(x) }},
		cost.ClassInt: {t: ir.U32, per: 2, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value {
			return b.Xor(ir.U32, x, b.Shr(ir.U32, x, b.Const(ir.U32, 1)))
		}},
		cost.ClassIndex: {t: ir.U32, per: 2, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value {
			return b.Add(ir.U32, b.Mul(ir.U32, x, x), b.Const(ir.U32, 1))
		}},
		cost.ClassLanes: {t: ir.F32, op: func(b *ir.Builder, _, _, x ir.Value) ir.Value { return b.ShuffleXor(ir.F32, x, 1) }},
		cost.ClassGroup: {t: ir.F32, op: groupSum, shared: true},
	}
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
func gpuOpKernel(name string, chains, deep, trips, width int, op gpuOp) (*ir.Kernel, error) {
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
