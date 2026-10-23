// Command mvbench measures the GPU decode matvec against a device read-bandwidth
// ceiling measured in the same process.
//
// Decode is bandwidth-bound (every weight byte read once per token), so a
// matvec kernel is good to the extent it saturates memory; the fraction of the
// wall is the figure of merit. See streamWall for why the wall itself is not
// quotable.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// gemma-2b's real decode shapes, from `jitllm info`.
var shapes = []struct {
	name    string
	rows, k int
	t       kernels.Quant
}{
	{"k/v (GQA)", 256, 2048, kernels.Q4_0},
	{"q/o", 2048, 2048, kernels.Q4_0},
	// The fused shapes: q, k and v share one activation vector, as do gate
	// and up, so each group can be one matvec over a taller matrix.
	{"qkv fused", 2560, 2048, kernels.Q4_0},
	{"gate+up fused", 32768, 2048, kernels.Q4_0},
	{"gate/up", 16384, 2048, kernels.Q4_0},
	{"ffn_down", 2048, 16384, kernels.Q4_0},
	{"lm_head (tied)", 256128, 2048, kernels.Q8_0},

	// tinyllama's k-quant shapes, which do more unpack arithmetic per byte.
	{"tll q/o Q4_K", 2048, 2048, kernels.Q4_K},
	{"tll gate/up Q3_K", 5632, 2048, kernels.Q3_K},
	{"tll ffn_down Q4_K", 2048, 5632, kernels.Q4_K},
	{"tll gate/up Q5_K", 5632, 2048, kernels.Q5_K},
	{"tll lm_head Q6_K", 32000, 2048, kernels.Q6_K},

	// Llama-3.2-1B-Instruct-Q4_K_M's decode shapes: NEmbd 2048, 8 KV heads of
	// 64, FFN 8192, tied Q6_K head over a 128256 vocabulary.
	{"l1b q/o Q4_K", 2048, 2048, kernels.Q4_K},
	{"l1b k/v Q4_K", 512, 2048, kernels.Q4_K},
	{"l1b gate/up Q4_K", 8192, 2048, kernels.Q4_K},
	{"l1b ffn_down Q4_K", 2048, 8192, kernels.Q4_K},
	{"l1b lm_head Q6_K", 128256, 2048, kernels.Q6_K},

	// Ministral-3-3B-Instruct-Q4_K_M's: NEmbd 3072, 32 heads and 8 KV heads of
	// 128, FFN 9216, an untied Q6_K head over 131072. Q4_K_M puts attn_v and
	// ffn_down in Q6_K on half the blocks.
	{"m3b q Q4_K", 4096, 3072, kernels.Q4_K},
	{"m3b k/v Q4_K", 1024, 3072, kernels.Q4_K},
	{"m3b v Q6_K", 1024, 3072, kernels.Q6_K},
	{"m3b o Q4_K", 3072, 4096, kernels.Q4_K},
	{"m3b gate/up Q4_K", 9216, 3072, kernels.Q4_K},
	{"m3b ffn_down Q4_K", 3072, 9216, kernels.Q4_K},
	{"m3b ffn_down Q6_K", 3072, 9216, kernels.Q6_K},
	{"m3b lm_head Q6_K", 131072, 3072, kernels.Q6_K},

	// One shape, every format. On its own this is misleading: a fixed shape
	// is not a fixed byte count, so the rows sit at different cache levels.
	// Use -eqbytes (see equalBytes) to compare formats.
	{"same/Q4_0", 32768, 2048, kernels.Q4_0},
	{"same/Q8_0", 32768, 2048, kernels.Q8_0},
	{"same/Q4_K", 32768, 2048, kernels.Q4_K},
	{"same/Q6_K", 32768, 2048, kernels.Q6_K},
	{"same/Q3_K", 32768, 2048, kernels.Q3_K},
	{"same/Q5_K", 32768, 2048, kernels.Q5_K},
}

// equalBytes rewrites the same/* rows so every format reads the same total
// bytes rather than covering the same shape. Matching bytes matches cache
// pressure, so the remaining difference is the kernel; at one shape the 4-bit
// rows read above the wall (a harness artefact) and looked falsely faster.
func equalBytes(miB int) error {
	target := int64(miB) << 20
	for i := range shapes {
		if !strings.HasPrefix(shapes[i].name, "same/") {
			continue
		}
		// Price one row from a 64-row sample: PackedWords' d plane rounds the
		// row count up to even on the narrow formats, so a single row would
		// over-state it.
		const probe = 64
		qs, d, sc, err := kernels.PackedWords(shapes[i].t, probe, shapes[i].k)
		if err != nil {
			return fmt.Errorf("mvbench: %s: %w", shapes[i].name, err)
		}
		perRow := float64(qs+d+sc) * 4 / probe
		rows := int(float64(target) / perRow)
		// The kernel tiles rows; keep it a multiple of 64 so no shape is a
		// ragged special case, and never zero.
		rows = (rows / 64) * 64
		if rows < 64 {
			rows = 64
		}
		shapes[i].rows = rows
		shapes[i].name = fmt.Sprintf("eq/%s", strings.TrimPrefix(shapes[i].name, "same/"))
	}
	return nil
}

func main() {
	rounds := flag.Int("rounds", 21, "timed rounds per shape")
	ntok := flag.Int("ntok", 0, "price the BATCHED prefill matvec at this many token columns")
	rowt := flag.Int("rowt", 1, "rows per thread in the batched matvec")
	mma := flag.Bool("mma", false, "price the batched matvec on the warp matrix instruction instead")
	wallMiB := flag.Int("wallmib", 256, "working set for the read-wall probe, MiB. It does NOT "+
		"agree with itself across sizes on the M4 (256 MiB -> 90 GB/s, 1 GiB -> 65, against MLX's "+
		"105.7 over 1 GiB) while the rows hold at 96-99, so this is a knob for understanding the "+
		"probe rather than a setting -- see streamWall")
	eqMiB := flag.Int("eqbytes", 0, "EQUAL-BYTES control: replace the same/* rows with a row count "+
		"per format that reads this many MiB, so cache pressure is matched across formats")
	wideW := flag.Int("wallw", 0, "sweep the read-wall probe at 1,2,4,8 CONSECUTIVE words per "+
		"thread and print nothing else. This IR has no vector load, so the question is whether "+
		"the MSL compiler merges w consecutive scalar Loads -- see kernels.StreamWide")
	only := flag.String("only", "", "time only the shapes whose name has this prefix")
	flag.Parse()
	if *only != "" {
		kept := shapes[:0]
		for _, s := range shapes {
			if strings.HasPrefix(s.name, *only) {
				kept = append(kept, s)
			}
		}
		shapes = kept
	}
	if *eqMiB > 0 {
		if err := equalBytes(*eqMiB); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	devs := backend.Open()
	if *wideW != 0 {
		for _, d := range devs {
			fmt.Printf("\n== %s  [%s]  read-wall probe, %d MiB working set\n", d.Name(), d.API(), *wallMiB)
			for _, a := range []struct {
				w, sp int
			}{{1, 0}, {2, 0}, {4, 0}, {8, 0}, {4, probeThreads}, {8, probeThreads}} {
				w := a.w
				wallW, wallSpread = a.w, a.sp
				gb, err := streamWall(d, *rounds, *wallMiB)
				if err != nil {
					fmt.Printf("  %d word(s)/thread/step: %v\n", w, err)
					continue
				}
				kind := "adjacent"
				if a.sp > 0 {
					kind = "STRIDED by one grid stride"
				}
				fmt.Printf("  %d word(s)/thread/step = %2d B per lane, %-20s: %6.1f GB/s\n",
					w, w*4, kind, gb)
			}
			d.Close()
		}
		return
	}
	if len(devs) == 0 {
		fmt.Fprintln(os.Stderr, "no GPU backend on this host")
		os.Exit(1)
	}
	for _, d := range devs {
		defer d.Close()
		fmt.Printf("\n== %s  [%s]\n", d.Name(), d.API())
		wall, err := streamWall(d, *rounds, *wallMiB)
		if err != nil {
			fmt.Printf("  bandwidth probe failed: %v\n", err)
			continue
		}
		fmt.Printf("  read wall: %.1f GB/s  (grid-strided, same access shape as the matvec)\n", wall)
		if fl, err := launchFloor(d, *rounds); err == nil {
			fmt.Printf("  launch floor: %.2f us through one Session; a shape whose whole time is\n"+
				"    near this cannot be priced in isolation at all\n",
				float64(fl.Nanoseconds())/1e3)
		}
		if *ntok > 0 {
			batchTable(d, wall, *ntok, *rowt, *rounds, *mma)
			continue
		}
		fmt.Println("  shape                  rows       k   type   split=1      2      4      8     16     32   best")
		for _, s := range shapes {
			nb := s.k / 32
			line := fmt.Sprintf("  %-20s %6d %7d  %5s", s.name, s.rows, s.k, s.t)
			best, bestSp := 0.0, 1
			for _, sp := range []int{1, 2, 4, 8, 16, 32} {
				if nb%sp != 0 {
					line += "      -"
					continue
				}
				gb, _, err := timeMatVec(d, s.t, s.rows, s.k, sp, *rounds)
				if err != nil {
					line += "    err"
					continue
				}
				line += fmt.Sprintf(" %6.0f", gb)
				if gb > best {
					best, bestSp = gb, sp
				}
			}
			// The same sweep with the split reduced in-group, printed beside
			// the shipped one: the pair shows whether a small matvec can spend
			// threads on Split without paying for the global reduce.
			gBest, gSp := 0.0, 0
			for _, sp := range []int{2, 4, 8, 16, 32} {
				if nb%sp != 0 || 128%sp != 0 || s.rows%(128/sp) != 0 {
					continue
				}
				gb, _, err := timeMatVecMode(d, s.t, s.rows, s.k, sp, *rounds, true)
				if err != nil {
					continue
				}
				if gb > gBest {
					gBest, gSp = gb, sp
				}
			}
			grp := ""
			if gBest > 0 {
				grp = fmt.Sprintf("   group %.0f @%d (%+.0f%%)", gBest, gSp, 100*(gBest-best)/best)
			}
			fmt.Printf("%s   %d (%.0f GB/s, %.0f%% of wall)%s\n", line, bestSp, best, 100*best/wall, grp)
		}
	}
}

// wallW is words per thread per step in the read-wall probe; see
// kernels.StreamWide. 1 is the shipped probe.
var wallW = 1

// wallSpread > 0 strides a lane's own words instead of making them adjacent.
var wallSpread = 0

// probeThreads is the read-wall probe's grid: enough to fill any of these devices.
const probeThreads = 256 * 1024

// streamWall measures the device's read bandwidth, the denominator of every
// "% of wall" in this tool.
//
// It does not agree with itself across working sets (on one arm64 host it peaks
// at 256 MiB and falls on both sides, always below a contiguous-sum figure),
// while the matvec rows stay stable. So compare rows against each other and
// do not quote the percentage; -wallmib exposes the working set.
func streamWall(d backend.Device, rounds, wallMiB int) (float64, error) {
	// probeThreads is shared with the -wallw sweep, which needs it as the stride
	// of its strided arm: a lane's w words must be exactly w grid strides apart
	// or the tiles overlap and the row reads above the roofline.
	const threads = probeThreads
	per := (wallMiB << 20) / 4 / threads
	if per < 1 {
		per = 1
	}
	n := threads * per
	k, err := kernels.StreamWideSpread(per, wallW, wallSpread)
	if err != nil {
		return 0, err
	}
	kern, err := d.Compile(k)
	if err != nil {
		return 0, err
	}
	defer kern.Close()

	in, err := d.Alloc(n * 4)
	if err != nil {
		return 0, err
	}
	defer in.Free()
	if err := in.Write(make([]byte, n*4)); err != nil {
		return 0, err
	}
	np, err := d.Alloc(4)
	if err != nil {
		return 0, err
	}
	defer np.Free()
	np.Write(u32b([]uint32{threads}))
	out, err := d.Alloc(threads * 4)
	if err != nil {
		return 0, err
	}
	defer out.Free()

	run := func() error { return kern.Launch(threads/256, 256, in, np, out) }
	for i := 0; i < 5; i++ {
		if err := run(); err != nil {
			return 0, err
		}
	}
	bytes := float64(n) * 4
	var rs []float64
	for r := 0; r < rounds; r++ {
		t0 := time.Now()
		if err := run(); err != nil {
			return 0, err
		}
		rs = append(rs, bytes/time.Since(t0).Seconds()/1e9)
	}
	sort.Float64s(rs)
	return rs[len(rs)/2], nil
}

// chooseSplit mirrors tier.chooseSplit so the benchmark measures what ships.
func chooseSplit(rows, nb int) int {
	const slots = 30720
	split := 1
	for split < 32 && rows*split < slots && nb%(split*2) == 0 {
		split *= 2
	}
	return split
}

func timeMatVec(d backend.Device, q kernels.Quant, nrows, k, split, rounds int) (float64, float64, error) {
	return timeMatVecMode(d, q, nrows, k, split, rounds, false)
}

// timeMatVecMode prices the matvec with the split reduced either through a
// global partial buffer and a second Reduce launch (the shipped path) or inside
// the threadgroup (GroupSplit). Small matvecs want a big Split, and the global
// round trip is what makes a big Split cost more than it buys.
func timeMatVecMode(d backend.Device, q kernels.Quant, nrows, k, split, rounds int, groupSplit bool) (float64, float64, error) {
	if groupSplit && (split < 2 || 128%split != 0 || nrows%(128/split) != 0) {
		return 0, 0, fmt.Errorf("shape refuses GroupSplit")
	}
	nb := k / 32
	raw := make([]byte, nrows*nb*q.BlockBytes())
	rng := rand.New(rand.NewSource(1))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	qs, dw, scw, err := kernels.PackWeights(q, raw, nrows, k)
	if err != nil {
		return 0, 0, err
	}
	av, as, asum := randActivations(rng, k)
	kk, err := kernels.MatVec(kernels.MatVecShape{T: q, K: k, Rows: nrows, Split: split, GroupSplit: groupSplit})
	if err != nil {
		return 0, 0, err
	}
	kern, err := d.Compile(kk)
	if err != nil {
		return 0, 0, err
	}
	defer kern.Close()

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(p []byte) (backend.Buf, error) {
		b, err := d.Alloc(len(p))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		return b, b.Write(p)
	}
	bQS, err := up(u32b(qs))
	if err != nil {
		return 0, 0, err
	}
	bD, _ := up(u32b(dw))
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	bSC, _ := up(u32b(scw))
	bA, _ := up(u32b(av))
	ax := append(append([]float32{}, as...), asum...)
	bAX, _ := up(f32b(ax))
	outN := nrows * split
	if groupSplit {
		outN = nrows // the kernel writes the final row; there is no partial plane
	}
	bOut, err := up(make([]byte, outN*4))
	if err != nil {
		return 0, 0, err
	}
	bFinal := bOut
	var red backend.Kernel
	if split > 1 && !groupSplit {
		if bFinal, err = up(make([]byte, nrows*4)); err != nil {
			return 0, 0, err
		}
		rk, err := kernels.Reduce(nrows, split)
		if err != nil {
			return 0, 0, err
		}
		if red, err = d.Compile(rk); err != nil {
			return 0, 0, err
		}
		defer red.Close()
	}

	const width = 128
	groups := (nrows + width - 1) / width

	// Timed through Session and amortised over iters, as tier.timeSplit does:
	// Kernel.Launch synchronises (~19 us on CUDA), so timing one blocking
	// launch per sample measures the drain, not the kernel.
	run := func(s backend.Session, n int) error {
		for i := 0; i < n; i++ {
			if err := s.Launch(kern, (nrows*split+width-1)/width, width,
				bQS, bD, bSC, bA, bAX, bOut); err != nil {
				return err
			}
			if split > 1 && !groupSplit {
				if err := s.Launch(red, groups, width, bOut, bFinal); err != nil {
					return err
				}
			}
		}
		return s.Sync()
	}
	// Enough repeats that the sample is far longer than one launch.
	iters := 1
	for {
		var took time.Duration
		var err error
		d.Session(func(s backend.Session) {
			t0 := time.Now()
			err = run(s, iters)
			took = time.Since(t0)
		})
		if err != nil {
			return 0, 0, err
		}
		if took > 2*time.Millisecond || iters >= 512 {
			break
		}
		iters *= 2
	}

	// Weight traffic only: the activation vector is tiny and stays in cache.
	bytes := float64(len(qs)+len(dw)+len(scw)) * 4
	var rs []float64
	for r := 0; r < rounds; r++ {
		var took time.Duration
		var err error
		d.Session(func(s backend.Session) {
			t0 := time.Now()
			err = run(s, iters)
			took = time.Since(t0)
		})
		if err != nil {
			return 0, 0, err
		}
		rs = append(rs, bytes*float64(iters)/took.Seconds()/1e9)
	}
	sort.Float64s(rs)
	med := rs[len(rs)/2]
	return med, (rs[len(rs)*3/4] - rs[len(rs)/4]) / med, nil
}

// launchFloor is the fixed cost of one launch through the same harness, so the
// per-shape table can be read against it: a shape whose whole time is near this
// cannot be priced in isolation at all.
func launchFloor(d backend.Device, rounds int) (time.Duration, error) {
	k, err := kernels.Add(128)
	if err != nil {
		return 0, err
	}
	c, err := d.Compile(k)
	if err != nil {
		return 0, err
	}
	defer c.Close()
	var bufs [3]backend.Buf
	for i := range bufs {
		if bufs[i], err = d.Alloc(512); err != nil {
			return 0, err
		}
		defer bufs[i].Free()
	}
	const iters = 512
	best := time.Hour
	for r := 0; r < rounds; r++ {
		var took time.Duration
		d.Session(func(s backend.Session) {
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				s.Launch(c, 1, 128, bufs[0], bufs[1], bufs[2])
			}
			s.Sync()
			took = time.Since(t0)
		})
		if took < best {
			best = took
		}
	}
	return best / iters, nil
}

func u32b(v []uint32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}
func f32b(v []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// batchTable prices the batched matvec -- the prefill kernel, with NTok token
// columns and Tok of them per thread. Prefill reads the weights once per token
// group, so it reports achieved Gmac/s and weight GB/s; at large Tok it is meant
// to be arithmetic-bound and can sit far under the wall.
func batchTable(d backend.Device, wall float64, ntok, rowt, rounds int, mma bool) {
	fmt.Printf("  batched matvec, NTok=%d Rowt=%d: Gmac/s (device weight GB/s)\n", ntok, rowt)
	fmt.Printf("  %-22s %6s %7s %6s", "shape", "rows", "k", "type")
	toks := []int{1, 2, 4, 8, 16, 32}
	if mma {
		// For the matrix path the sweep is over the warp tile: t is NT, the
		// number of 8-token columns, and rowt is MT, the number of 16-row
		// blocks. Tok has no meaning there -- a lane holds four accumulators
		// per tile whatever the tile is.
		toks = []int{1, 2, 4, 8}
	}
	for _, t := range toks {
		fmt.Printf(" %13s", fmt.Sprintf("tok=%d", t))
	}
	fmt.Println()
	for _, sh := range shapes {
		fmt.Printf("  %-22s %6d %7d %6s", sh.name, sh.rows, sh.k, sh.t)
		for _, t := range toks {
			if ntok%t != 0 {
				fmt.Printf(" %13s", "-")
				continue
			}
			if sh.rows%rowt != 0 {
				fmt.Printf(" %13s", "-")
				continue
			}
			us, packed, err := timeBatched(d, sh.t, sh.rows, sh.k, ntok, t, rowt, rounds, mma)
			if err != nil || us <= 0 {
				fmt.Printf(" %13s", "err")
				continue
			}
			gmac := float64(sh.rows) * float64(sh.k) * float64(ntok) / us / 1e3
			reads := ntok / t
			if mma {
				reads = ntok / (8 * t)
			}
			wb := packed * float64(reads)
			fmt.Printf(" %7.0f(%5.0f)", gmac, wb/us/1e3)
		}
		fmt.Println()
	}
	_ = wall
}

// timeBatched returns microseconds per launch and the device bytes of one full
// pass over the weights. The byte count comes from the packed arrays, not from
// BlockBytes, which is the source block (256 elements for a k-quant) and would
// overcount k-quants eightfold.
func timeBatched(d backend.Device, q kernels.Quant, nrows, k, ntok, tok, rowt, rounds int, mma bool) (float64, float64, error) {
	nb := k / 32
	raw := make([]byte, nrows*nb*q.BlockBytes())
	rng := rand.New(rand.NewSource(1))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	qs, dw, scw, err := kernels.PackWeights(q, raw, nrows, k)
	if err != nil {
		return 0, 0, err
	}
	// One flat pAX over all the tokens, the layout the kernel reads:
	// kernels.Quantize writes all the scales, then all the per-16 sums, so
	// token j's scales sit at j*nb and its sums at NTok*nb + j*(K/16).
	var av []uint32
	var as, asum []float32
	for t := 0; t < ntok; t++ {
		v, s1, s2 := randActivations(rng, k)
		av = append(av, v...)
		as = append(as, s1...)
		asum = append(asum, s2...)
	}
	shape := kernels.MatVecShape{T: q, K: k, Rows: nrows, Split: 1, NTok: ntok, Tok: tok, Rowt: rowt}
	build := kernels.MatVec
	if mma {
		shape.MT, shape.NT = rowt, tok
		build = kernels.MatVecMMA
	}
	kk, err := build(shape)
	if err != nil {
		return 0, 0, err
	}
	kern, err := d.Compile(kk)
	if err != nil {
		return 0, 0, err
	}
	defer kern.Close()

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(p []byte) (backend.Buf, error) {
		b, err := d.Alloc(len(p))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		return b, b.Write(p)
	}
	bQS, err := up(u32b(qs))
	if err != nil {
		return 0, 0, err
	}
	bD, _ := up(u32b(dw))
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	bSC, _ := up(u32b(scw))
	bA, _ := up(u32b(av))
	bAX, _ := up(f32b(append(append([]float32{}, as...), asum...)))
	bOut, err := up(make([]byte, nrows*ntok*4))
	if err != nil {
		return 0, 0, err
	}
	threads := (nrows / rowt) * (ntok / tok)
	if mma {
		// One WARP per (m,n) tile.
		threads = (nrows / (16 * rowt)) * (ntok / (8 * tok)) * 32
	}
	// One Sync at the end: CUDA launches are asynchronous, so without it this
	// times the enqueue. Repeat until the sample is far longer than one
	// launch, then divide.
	run := func(n int) (time.Duration, error) {
		var took time.Duration
		var e error
		d.Session(func(s backend.Session) {
			t0 := time.Now()
			for i := 0; i < n; i++ {
				if err := s.Launch(kern, (threads+127)/128, 128, bQS, bD, bSC, bA, bAX, bOut); err != nil {
					e = err
					return
				}
			}
			e = s.Sync()
			took = time.Since(t0)
		})
		return took, e
	}
	iters := 1
	for {
		took, e := run(iters)
		if e != nil {
			return 0, 0, e
		}
		if took > 2*time.Millisecond || iters >= 512 {
			break
		}
		iters *= 2
	}
	best := 0.0
	for r := 0; r < rounds; r++ {
		took, e := run(iters)
		if e != nil {
			return 0, 0, e
		}
		us := float64(took.Nanoseconds()) / 1e3 / float64(iters)
		if best == 0 || us < best {
			best = us
		}
	}
	return best, float64(len(qs)+len(dw)+len(scw)) * 4, nil
}

// randActivations is a row's int8 form for timing: random codes, a fixed
// scale per 32 and zero sums. A matvec's time does not depend on the values
// it reads, so nothing is quantized.
func randActivations(rng *rand.Rand, k int) (a []uint32, as, sum []float32) {
	a = make([]uint32, k/4)
	for i := range a {
		a[i] = rng.Uint32()
	}
	as = make([]float32, k/32)
	for i := range as {
		as[i] = 1.0 / 127
	}
	return a, as, make([]float32, 2*(k/32))
}
