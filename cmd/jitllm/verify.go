package main

// The verification harness: decode the same prompt on both tiers and diff the
// token ids before printing any rate, and run the paired A/Bs. A rate from a
// tier that computes something else is worthless, so the diff comes first.

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

func verifyCmd(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	migrate := fs.Bool("migrate", false, "move the CPU/GPU seam during the teacher-forced diff")
	// Off by default: no single constant covers every model and seam (a fully
	// placed dense model can move logits more than a deep hybrid's partial
	// seam), so the caller states the band.
	band := fs.Float64("dlogit", 0, "the largest per-logit CPU/device difference to "+
		"accept, 0 to not check it. Setting it also lets an argmax flip inside that "+
		"movement count as a TIE rather than a disagreement")
	// "The device agrees with the host" is a claim about one device, so the
	// harness takes the same device grammar as run.
	dev := fs.String("devices", "auto", "which device(s) to verify; the -devices grammar")
	n := fs.Int("n", 32, "tokens to generate")
	// 0 asks the device, like run's.
	vram := bytesFlag(fs, "vram", 0, "device weight budget, bytes or 3G/512M; 0 asks the device")
	maxmem := bytesFlag(fs, "maxmem", 0, "host weight residency budget, bytes or 3G/512M; 0 works it out")
	ab := fs.String("ab", "", "instead of verifying, run a paired A/B: split, submit, head, graph, softmax, fuse, actfuse, ropetab, persist, persistdeps, persistdeps-grid, persist-aa, gateup, gateup-aa, attnchunk, attnpair or regs")
	rounds := fs.Int("rounds", 21, "A/B rounds")
	verify := fs.Bool("verify", false, "recompute every served matvec on the host and report disagreements")
	prefill := fs.Int("prefill", 0, "gate the BATCHED device prefill: prefill this many tokens on both tiers and diff")
	depth := fs.Int("depth", 0, "context depth to run an -ab arm at; attention is 1% of a token at 0 and 30% at 1536")
	registers := fs.Bool("registers", false, "instead of verifying, run decode and the batched prefill on the "+
		"device and report every compiled kernel's registers, spill, shared memory and occupancy")
	asJSON := fs.Bool("json", false, "with -registers, print the report as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: jitllm verify [flags] <model.jlm>")
	}
	// The same option set every other command gets: the model and nn packages
	// read no environment, so JITLLM_* reaches them only through here.
	m, err := model.Open(fs.Arg(0), loadOpts(nil, *maxmem)...)
	if err != nil {
		return err
	}
	defer m.Close()
	// The split tuner is pinned: a k-split is the f32 reduction order, so two
	// States that tuned separately differ in the last bits, and on a deep
	// hybrid that difference is amplified past any useful diff. It goes before
	// gpuOptions so JITLLM_GPU_TUNE still overrides it; the rate below is
	// therefore a fitted-split rate.
	g, err := tier.OpenWith(append([]tier.Option{
		tier.WithDevices(*dev), tier.WithBudget(*vram),
		tier.WithDeviceTune(tier.TuneOff)}, gpuOptions()...)...)
	if err != nil {
		return err
	}
	defer g.Close()
	g.Verify = *verify
	// KVF16 is an environment variable, not an -ab arm: it changes the V
	// cache's layout and size at PrepLayer, so flipping it mid-run would
	// reinterpret the cache. The value is read, not just its presence.
	if v := os.Getenv("JITLLM_KV_F16"); v != "" {
		g.KVF16 = v == "1"
	}
	fmt.Printf("device: %s\n", g.Name())
	fmt.Printf("model:  %s  L=%d d=%d\n", fs.Arg(0), m.Cfg.NLayer, m.Cfg.NEmbd)

	if *registers {
		// The kernels a model runs are the ones it compiles: decode, then the
		// batched prefill (diffed against per-token decode, so a report is never
		// read off a path that computes something else).
		benchRun(m, nil, *n)
		benchRun(m, g, *n)
		if err := prefillDiff(m, g, max(*prefill, 40), 4); err != nil {
			// The diff is printed above it; the report still describes the
			// kernels that ran, which is what was asked for here.
			fmt.Printf("  (prefill diff: %v)\n", err)
		}
		return registerReport(os.Stdout, cpu.KernelRegisters(), g.KernelResources(), *asJSON)
	}
	if *ab != "" {
		abRun(m, g, *ab, *n, *rounds, *depth)
		return nil
	}
	if *prefill > 0 {
		return prefillDiff(m, g, *prefill, *n)
	}
	cpuRate := benchRun(m, nil, *n)
	fmt.Printf("\ncpu        %7.2f tok/s\n", cpuRate)

	gpuRate := benchRun(m, g, *n)
	// Stats() sums the per-device counters; DevStats() below keeps them apart.
	st := g.Stats()
	fmt.Printf("gpu        %7.2f tok/s   %.2fx\n", gpuRate, gpuRate/cpuRate)
	fmt.Printf("  device weights: %.2f GiB   (prep: %.0f ms pack + %.0f ms upload)\n",
		float64(g.Bytes())/(1<<30),
		float64(st.TPack.Microseconds())/1000, float64(st.TUpload.Microseconds())/1000)
	if g.Devices() > 1 {
		// Which card took what: the totals cannot tell "both took half" from
		// "the second took nothing".
		per := g.DevStats()
		for i, n := range g.Placed() {
			fmt.Printf("  device %-28s %d block(s), %d matvec(s), %.2f GiB of KV\n",
				per[i].Device, n, per[i].Served, float64(per[i].KVBytes)/(1<<30))
		}
		fmt.Printf("  %d device change(s) per token (the residual stream crosses the host at each)\n",
			g.Crossings())
	}
	if st.NoRoom > 0 {
		fmt.Printf("  ! %d tensors the DEVICE refused (out of memory); those matvecs ran on the CPU,\n"+
			"    so this number is not comparable with a run that had the card to itself\n", st.NoRoom)
	}
	fmt.Printf("  %d matvecs on device, %d refused (no kernel for the quant format), %d over budget\n",
		st.Served, st.NoKernel, st.Declined)
	if st.TooSmall > 0 {
		fmt.Printf("  %d matvecs left on the host: this model does not carry enough weight bytes"+
			" per host/device crossing to pay for one (tier.mvPays)\n", st.TooSmall)
	}
	if st.Failed > 0 {
		fmt.Printf("  ! %d matvec kernel(s) did not compile on the device; those matvecs ran on the CPU\n", st.Failed)
	}
	if st.LastErr != "" {
		fmt.Printf("  kernel compilation failed: %s\n", st.LastErr)
	}
	if st.Blocks > 0 {
		fn := float64(*n)
		// Say where the projection is: under a partial seam placeHead gives up
		// blocks for the head, so the block count alone is misleading.
		where := "host"
		if g.HeadResident() {
			where = "device"
		}
		fmt.Printf("  %d blocks on device (%.1f per token), output projection on the %s, %.2f ms per token in Layers\n",
			st.Blocks, float64(st.Blocks)/fn, where, float64(st.TLayer.Microseconds())/1e3/fn)
		// One capture per prompt plus one per scoreGrain tokens is the design;
		// one per token means the launch sequence is moving and the recording
		// is pure overhead.
		fmt.Printf("  %d launch-sequence captures\n", st.Captures)
		// Encode time is what a recording (for example an indirect command
		// buffer) could remove, so it is the ceiling on that change.
		if st.TEmit > 0 {
			em := float64(st.TEmit.Microseconds()) / 1e3 / fn
			fmt.Printf("  %.2f ms per token encoding launches (%.0f%% of Layers) -- the ceiling on a recording\n",
				em, 100*em/(float64(st.TLayer.Microseconds())/1e3/fn))
		}
		// Which softmax ran. The warp version sums the row as a tree and the
		// scalar one left to right, so a token that disagrees below wants this
		// line read first.
		if st.SoftmaxLanes == 1 {
			fmt.Printf("  softmax: 1 thread per head (%s)\n", st.SoftmaxWhy)
		} else {
			fmt.Printf("  softmax: %d lanes per head (subgroup shuffle)\n", st.SoftmaxLanes)
		}
		// The subgroup guarantee that selected it: every kernel that needs a
		// subgroup is chosen from this one device fact.
		switch {
		case st.Lanes == 1:
			fmt.Printf("  subgroup: no %d-lane guarantee, scalar twins (%s)\n",
				ir.SubgroupLanes, st.LanesWhy)
		case st.Lanes > 0 && st.LanesWhy != "":
			fmt.Printf("  subgroup: %d lanes -- %s\n", st.Lanes, st.LanesWhy)
		case st.Lanes > 0:
			fmt.Printf("  subgroup: %d lanes guaranteed for the compute stage\n", st.Lanes)
		}
	}
	if st.Served > 0 {
		fn := float64(*n)
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 / fn }
		fmt.Printf("  per token: prep %.2f ms, device %.2f ms, widen %.2f ms\n",
			ms(st.TStage), ms(st.TLaunch), ms(st.TConv))
	}

	// A device that took no blocks makes the "gpu" arm run on the host, where
	// it agrees perfectly with the CPU arm. Fail instead of reporting that.
	if st.Blocks == 0 && st.Served == 0 {
		fmt.Printf("\n! the device took NO blocks and served NO matvecs: this is a CPU-vs-CPU\n" +
			"  comparison and both the rate and the agreement below are meaningless\n")
		if st.LastErr != "" {
			return fmt.Errorf("device declined everything: %s", st.LastErr)
		}
		return fmt.Errorf("device declined everything and reported no error")
	}
	return forcedDiff(m, g, *n, *migrate, *band)
}

// prefillDiff is the gate on the batched device prefill.
//
// It diffs the device against itself, batched against per-token, not against
// the host: the two device paths run the same kernels in the same order (the
// mask is -inf and the surplus AttnAcc terms are 0*v), so any row-mapping bug
// is a wide-margin disagreement rather than a tie.
//
// It keeps decoding after the prompt, teacher-forced from a fixed token
// stream, because a chunk writes ntok rows of K and V but returns one row of
// logits; decode reads every position the chunk wrote.
//
// The length is deliberately not a multiple of the batch width: the ragged last
// chunk writes K and V past the prompt, and decode must overwrite them.
func prefillDiff(m *model.Model, g *tier.GPU, ntok, n int) error {
	if ntok < 2 {
		return fmt.Errorf("verify -prefill: want at least 2 tokens")
	}
	// Both splits are pinned to one segment, which removes the two
	// reassociations unrelated to batching and makes the batched path
	// bit-identical to the per-token one; the gate is equality. Set before the
	// first PrepLayer, which is where initScratch reads them.
	os.Setenv("JITLLM_GPU_SPLIT", "1")
	os.Setenv("JITLLM_GPU_ACCSPLIT", "1")
	// Real text, not random ids: a random stream is off-distribution and sits
	// on argmax ties. The text must also vary within a chunk, since identical
	// rows hide a broken row mapping.
	toks := naturalPrompt(m, ntok)
	if len(toks) < ntok {
		return fmt.Errorf("verify -prefill: could only tokenize %d of %d tokens", len(toks), ntok)
	}
	drive := make([]int32, n)
	for i := range drive {
		drive[i] = toks[(i*29)%len(toks)]
	}

	arm := func(noBatch bool) ([][]float32, error) {
		g.NoBatch = noBatch
		s := m.NewState(ntok + n + 16)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			return nil, err
		}
		// Hand the blocks back before the next arm: a tier holds one KV cache
		// per block, and a second State's PrepLayer would decline on budget.
		defer s.SetGPULayers(0)
		if s.GPULayers() == 0 {
			return nil, fmt.Errorf("verify -prefill: the device took no blocks: %s", g.Err())
		}
		out := make([][]float32, 0, n+1)
		keep := func(l []float32) { out = append(out, append([]float32(nil), l...)) }
		l, err := s.Prefill(toks)
		if err != nil {
			return nil, fmt.Errorf("%w (device: %s)", err, g.Err())
		}
		keep(l)
		for i := 0; i < n; i++ {
			if l, err = s.Forward(drive[i]); err != nil {
				return nil, err
			}
			keep(l)
		}
		return out, nil
	}
	batched, err := arm(false)
	if err != nil {
		return err
	}
	perTok, err := arm(true)
	if err != nil {
		return err
	}
	// The control is the per-token arm run twice; it must be bit-identical,
	// or nothing here is measurable.
	control, err := arm(true)
	if err != nil {
		return err
	}
	g.NoBatch = false

	top2 := func(l []float32) (int32, float64) {
		bi, best, second := int32(0), l[0], float32(math.Inf(-1))
		for j, v := range l[1:] {
			if v > best {
				best, second, bi = v, best, int32(j+1)
			} else if v > second {
				second = v
			}
		}
		return bi, float64(best - second)
	}
	// decided counts flips where the model was NOT undecided: the winning
	// margin was wider than the whole perturbation at that position, so no
	// amount of rounding could have moved it.
	decided := 0
	cmp := func(label string, x, y [][]float32, show bool) (int, float64) {
		worst, bad := 0.0, 0
		for i := range x {
			a, b := x[i], y[i]
			d := 0.0
			for j := range a {
				if e := math.Abs(float64(a[j] - b[j])); e > d {
					d = e
				}
			}
			if d > worst {
				worst = d
			}
			ai, am := top2(a)
			bi, bm := top2(b)
			if ai != bi {
				if am > d && bm > d {
					decided++
				}
				if show && bad < 5 {
					where := "prompt"
					if i > 0 {
						where = fmt.Sprintf("decode %d", i-1)
					}
					fmt.Printf("  %s %s: %d (margin %.3g) vs %d (margin %.3g), max|dlogit| %.3g\n",
						label, where, ai, am, bi, bm, d)
				}
				bad++
			}
		}
		return bad, worst
	}
	bad, worst := cmp("batched", batched, perTok, true)
	cbad, cworst := cmp("control", perTok, control, false)
	fmt.Printf("\nprefill %d tok + %d decode, splits pinned to 1\n", ntok, n)
	fmt.Printf("  batched vs per-token   %d/%d disagree, max|dlogit| %.3g\n", bad, len(batched), worst)
	fmt.Printf("  per-token vs itself    %d/%d disagree, max|dlogit| %.3g   CONTROL\n", cbad, len(control), cworst)
	if cworst != 0 {
		return fmt.Errorf("the control is not bit-identical (max|dlogit| %g), so nothing here is measurable", cworst)
	}
	// The bar depends on the arithmetic the batched path ran. Without binary16
	// operands it is bit equality. Matrix-instruction attention scores and
	// matvecs that read the activation as binary16 (GemmTile, sm_70's twins)
	// cannot match the int8 per-token path exactly, so there the bar is token
	// ids and a bounded max|dlogit|.
	if st := g.Stats(); !st.AttnMMA && st.TileMV == 0 && st.VoltaMV == 0 {
		if worst != 0 {
			return fmt.Errorf("batched prefill is not bit-identical to per-token: max|dlogit| %g, %d/%d argmax flips",
				worst, bad, len(batched))
		}
		return nil
	}
	fmt.Printf("  (the batched path runs binary16 operands -- the scores on the matrix\n" +
		"   instruction or the matvecs on binary16 activations -- so the bar is token\n" +
		"   ids and a bounded max|dlogit|, not equality)\n")
	// A flip counts only when the winning margin exceeds the whole
	// perturbation at that position; narrower flips are ties.
	if decided > 0 {
		return fmt.Errorf("binary16 prefill changed %d token ids on a WIDE margin, "+
			"which rounding cannot explain", decided)
	}
	if bad > 0 {
		fmt.Printf("  %d/%d ids differ, every one of them on a margin narrower than the\n"+
			"   perturbation at that position -- ties, not defects\n", bad, len(batched))
	}
	// Far past the per-token arm's own float32 rounding band: a wrong
	// fragment, not a rounding.
	if worst > 8 {
		return fmt.Errorf("binary16 prefill moved a logit by %g, which is past rounding", worst)
	}
	return nil
}

// naturalPrompt returns at least n token ids of ordinary prose, tokenized by
// the model's own vocabulary so the ids are ones it has actually seen together.
// Falls back to a strided walk of the vocabulary where the file carries no
// usable tokenizer.
func naturalPrompt(m *model.Model, n int) []int32 {
	const para = "The city grew along the river, and every spring the water rose " +
		"until the bridges were islands. Merchants kept their ledgers on the upper " +
		"floors. A clerk named Aldis counted barrels of salt, wool, and dried fish, " +
		"and wrote the totals in a small hand that nobody else could read. In winter " +
		"the ice held long enough for carts to cross, and the tolls were collected " +
		"on the far bank by a man who never gave change. "
	if m.Vocab != nil {
		text := ""
		for len(text) < n*8 {
			text += para
		}
		if ids := m.Vocab.Encode(text, true); len(ids) >= n {
			return ids[:n]
		}
	}
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(1 + (i*7919)%(m.Cfg.NVocab-1))
	}
	return out
}

// isFinite is the guard every logit comparison in this file needs before it
// subtracts: an Inf or a NaN makes every ordering test below silently false.
func isFinite(v float32) bool {
	f := float64(v)
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// forcedDiff drives both tiers with the same token ids (the CPU's argmax) and
// compares the logits at every position. Two free-running chains cannot
// localise a disagreement: one tie broken differently changes every later id.
// Per position, the margin between the top two logits says whether a flip is a
// computation error or a tie.
func forcedDiff(m *model.Model, g *tier.GPU, n int, doMigrate bool, band float64) error {
	cs := m.NewState(n + 16)
	defer cs.Close()
	gs := m.NewState(n + 16)
	defer gs.Close()
	if err := gs.SetDevice(g); err != nil {
		return err
	}

	top2 := func(l []float32) (int32, float64) {
		bi, best, second := int32(0), l[0], float32(math.Inf(-1))
		for j, v := range l[1:] {
			if v > best {
				best, second, bi = v, best, int32(j+1)
			} else if v > second {
				second = v
			}
		}
		return bi, float64(best - second)
	}

	// -migrate moves the seam while the chains are compared, since a stale KV
	// cache or a replayed graph over freed buffers yields plausible logits. The
	// pattern walks the seam down to zero and back up to where it started.
	//
	// This State's own placement is printed: a full card can give this second
	// State fewer blocks than the benchmark session.
	full := gs.GPULayers()
	fmt.Printf("  teacher-forced arm: %d block(s) on the device, runs %v\n",
		full, gs.DeviceBlocks())
	migrate := func(i int) {
		if !doMigrate || full == 0 || i == 0 || i%4 != 0 {
			return
		}
		steps := []int{full / 2, 0, full / 2, full}
		want := steps[(i/4-1)%len(steps)]
		got := gs.SetGPULayers(want)
		fmt.Printf("  pos %d: seam -> %d blocks (asked %d)\n", i, got, want)
	}

	// Start at the model's own BOS, not id 1 (which is <eos> on gemma): an
	// off-distribution first token is ill-conditioned and amplifies legitimate
	// rounding at every later position.
	tok, bad, ties := int32(1), 0, 0
	if m.Vocab != nil && m.Vocab.BOS >= 0 && int(m.Vocab.BOS) < m.Cfg.NVocab {
		tok = m.Vocab.BOS
	}
	worst := 0.0
	// Every position's error is kept, not just the flips: its shape tells a
	// cache written wrong once (diverges and stays) from a scratch read before
	// it is written (alternates).
	trace := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		migrate(i)
		cl, err := cs.Forward(tok)
		if err != nil {
			return err
		}
		gl, err := gs.Forward(tok)
		if err != nil {
			return err
		}
		// Non-finite first: math.Abs(NaN) > d is false, so a NaN row would
		// score max|dlogit| 0. A non-finite logit is a hard failure.
		for j := range cl {
			if !isFinite(cl[j]) || !isFinite(gl[j]) {
				return fmt.Errorf("pos %d: logit %d is non-finite (cpu %v, gpu %v) -- "+
					"a NaN row scores max|dlogit| 0 here, so this must fail before the diff",
					i, j, cl[j], gl[j])
			}
		}
		d := 0.0
		for j := range cl {
			if e := math.Abs(float64(cl[j] - gl[j])); e > d {
				d = e
			}
		}
		if d > worst {
			worst = d
		}
		trace = append(trace, d)
		ci, cm := top2(cl)
		gi, gm := top2(gl)
		if ci != gi {
			// A flip is a tie when the CPU's gap between the two candidates is
			// within the error on exactly those two logits: device reduction
			// order legitimately moves logits, more so across a deep seam.
			// band anchors it: without a ceiling on movement, a device wrong
			// everywhere would explain every flip.
			gap := math.Abs(float64(cl[ci] - cl[gi]))
			moved := math.Abs(float64(gl[ci]-cl[ci])) + math.Abs(float64(gl[gi]-cl[gi]))
			tie := band > 0 && gap <= moved && moved <= band
			if tie {
				ties++
			} else {
				bad++
			}
			if ties+bad <= 5 {
				what := "DISAGREES"
				if tie {
					what = "tie"
				}
				fmt.Printf("  pos %d: %s -- cpu %d (margin %.3g), gpu %d (margin %.3g), "+
					"gap %.3g against %.3g of movement on those two, max|dlogit| %.3g\n",
					i, what, ci, cm, gi, gm, gap, moved, d)
			}
		}
		tok = ci // the CPU drives BOTH chains, so position i+1 sees one input
	}
	if doMigrate && full > 0 {
		// Regrow the whole prefix twice from empty, cold and prewarmed; the
		// difference is the stall a service feels when it takes memory back.
		regrow := func(prewarm bool) time.Duration {
			gs.SetGPULayers(0)
			g.DropStage()
			if prewarm {
				<-gs.PrewarmGPU(full)
			}
			t0 := time.Now()
			gs.SetGPULayers(full)
			return time.Since(t0)
		}
		cold := regrow(false)
		warm := regrow(true)
		fmt.Printf("\nregrow to %d blocks:  cold %.0f ms   prewarmed %.0f ms   (%.2fx)\n",
			full, float64(cold.Microseconds())/1000, float64(warm.Microseconds())/1000,
			float64(cold)/float64(warm))
	}
	if worst > 0 {
		fmt.Printf("\nmax|dlogit| by position:")
		for i, d := range trace {
			if i%8 == 0 {
				fmt.Printf("\n  %3d:", i)
			}
			fmt.Printf(" %8.3g", d)
		}
		fmt.Println()
	}
	how := "no -dlogit band, so every flip counts"
	if band > 0 {
		how = fmt.Sprintf("against a %.3g band", band)
	}
	fmt.Printf("\nteacher-forced: %d/%d positions disagree, %d tie(s) inside the "+
		"arithmetic, max|dlogit| %.3g %s\n", bad, n, ties, worst, how)
	if bad > 0 {
		return fmt.Errorf("%d/%d positions disagree", bad, n)
	}
	// The size is checked even when no argmax moved: a device wrong everywhere
	// still picks the same token when margins are wide.
	if band > 0 && worst > band {
		return fmt.Errorf("max|dlogit| %.3g is outside the %.3g band, so the two "+
			"tiers are not running the same arithmetic however the argmaxes fell "+
			"(-dlogit raises it)", worst, band)
	}
	return nil
}

// benchRun generates n tokens greedily and returns the decode rate. The first
// few tokens are discarded: codegen, tuning, weight upload and first-touch page
// faults are not per-token costs. Correctness is forcedDiff's job, not this
// one's -- a free-running chain is the right thing to TIME and the wrong thing
// to DIFF.
func benchRun(m *model.Model, dev *tier.GPU, n int) float64 {
	s := m.NewState(n + 16)
	defer s.Close()
	if dev != nil {
		if err := s.SetDevice(dev); err != nil {
			panic(err)
		}
	}
	tok := int32(1)
	const warm = 8
	firstBad := -1
	var start time.Time
	for i := 0; i < warm+n; i++ {
		if i == warm {
			start = time.Now()
		}
		logits, err := s.Forward(tok)
		if err != nil {
			panic(err)
		}
		// v > best is false for NaN, so a NaN row would leave the argmax at 0
		// and report an ordinary rate. Say so.
		if !isFinite(logits[0]) && firstBad < 0 {
			firstBad = i
			fmt.Printf("  ! the benchmark arm is NON-FINITE from position %d; "+
				"the rate below is the speed of computing NaN\n", i)
		}
		best, bi := logits[0], int32(0)
		for j, v := range logits {
			if v > best {
				best, bi = v, int32(j)
			}
		}
		tok = bi
	}
	return float64(n) / time.Since(start).Seconds()
}

// abRun runs a paired A/B of `which`, interleaved in one process on one State,
// starting at a context depth of `depth`. Depth matters for attention arms,
// which are a small share of a token at short context.
//
// The state is sized for the whole run rather than reset back to depth,
// because re-prefilling to depth would be timed as part of an arm.
func abRun(m *model.Model, g *tier.GPU, which string, n, rounds, depth int) {
	s := m.NewState(depth + rounds*n*2 + 16)
	defer s.Close()
	// attnchunk and attnpair compare host attention layouts, which do not run
	// at all with the blocks on a device; every other arm needs the device.
	hostArm := which == "attnchunk" || which == "attnpair"
	if !hostArm {
		if err := s.SetDevice(g); err != nil {
			panic(err)
		}
		if s.GPULayers() == 0 {
			panic(fmt.Errorf("the device took no blocks; there is nothing to compare"))
		}
	}
	if depth > 0 {
		ids := make([]int32, depth)
		for i := range ids {
			ids[i] = int32(1 + i%(m.Cfg.NVocab-1))
		}
		if _, err := s.Prefill(ids); err != nil {
			panic(err)
		}
	}
	lim := s.MaxSeq() - 2
	// The context grows over the A/B; -depth is where it starts. For a
	// measurement at a depth, keep rounds*n*2 small against depth.
	fmt.Printf("  context runs %d..%d over the whole A/B (-depth is the START)\n",
		depth, min(depth+rounds*n*2, lim))
	runOn := func(s *model.State, iters int) {
		tok := int32(1)
		for i := 0; i < iters; i++ {
			logits, err := s.Forward(tok)
			if err != nil {
				panic(err)
			}
			best, bi := logits[0], int32(0)
			for j, v := range logits {
				if v > best {
					best, bi = v, int32(j)
				}
			}
			tok = bi
			// Only when the cache is actually full (the state is sized so it
			// is not); the depth is then restored.
			if s.Pos() >= lim {
				s.Reset()
				if depth > 0 {
					ids := make([]int32, depth)
					for j := range ids {
						ids[j] = int32(1 + j%(m.Cfg.NVocab-1))
					}
					if _, err := s.Prefill(ids); err != nil {
						panic(err)
					}
				}
			}
		}
	}
	step := func(iters int) { runOn(s, iters) }
	var a, b bench.Case
	switch which {
	case "spill":
		// The batched prefill on two tiers of the same card, each iteration
		// four -n token prompts from a fresh history. A is a second tier that
		// allows GemmVolta's spilling tiles unless JITLLM_AB_SPILL_A=0; B is the
		// verify tier, which refuses them unless JITLLM_GPU_SPILLING_TILES=1.
		// Setting both alike is the A/A self-control; swapping them prices
		// which tier is which.
		aAllow := os.Getenv("JITLLM_AB_SPILL_A") != "0"
		g2, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda:0"),
			tier.WithDeviceTune(tier.TuneOff)}, append(gpuOptions(),
			tier.WithConfig(func(c *tier.Config) { c.SpillingTiles = aAllow }))...)...)
		if err != nil {
			panic(err)
		}
		defer g2.Close()
		s2 := m.NewState(n + 16)
		defer s2.Close()
		if err := s2.SetDevice(g2); err != nil {
			panic(err)
		}
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32(1 + (i*7919)%(m.Cfg.NVocab-1))
		}
		// bench.AB hands each arm -n as its iteration count, which here is
		// the prompt length; a round is four prompts instead.
		pre := func(st *model.State, _ int) {
			for i := 0; i < 4; i++ {
				st.Reset()
				if _, err := st.Prefill(ids); err != nil {
					panic(err)
				}
			}
		}
		name := func(allow bool, which string) string {
			if allow {
				return "spilling tiles allowed (" + which + ")"
			}
			return "spilling tiles refused (" + which + ")"
		}
		a = bench.Case{Name: name(aAllow, "second tier"), Fn: func(it int) { pre(s2, it) }}
		b = bench.Case{Name: name(g.SpillingTiles, "verify tier"), Fn: func(it int) { pre(s, it) }}
		defer func() {
			fmt.Printf("tiles refused for spilling: A %d, B %d\n", g2.Stats().SpillRefused, g.Stats().SpillRefused)
		}()
	case "regs":
		// B is a second tier on the same card whose every CUDA kernel is
		// capped at JITLLM_AB_MAXREGS registers (.maxnreg), with a State of
		// its own. 0 caps nothing, which makes this the A/A self-control: two
		// tiers, two States, the same kernels.
		maxRegs, _ := strconv.Atoi(os.Getenv("JITLLM_AB_MAXREGS"))
		g2, err := tier.OpenWith(append([]tier.Option{tier.WithDevices("cuda:0"),
			tier.WithDeviceTune(tier.TuneOff)}, append(gpuOptions(), tier.WithCUDAMaxRegs(maxRegs))...)...)
		if err != nil {
			panic(err)
		}
		defer g2.Close()
		s2 := m.NewState(depth + rounds*n*2 + 16)
		defer s2.Close()
		if err := s2.SetDevice(g2); err != nil {
			panic(err)
		}
		if s2.GPULayers() != s.GPULayers() {
			panic(fmt.Errorf("the capped tier took %d blocks and the shipped one %d: not one comparison",
				s2.GPULayers(), s.GPULayers()))
		}
		if depth > 0 {
			ids := make([]int32, depth)
			for i := range ids {
				ids[i] = int32(1 + i%(m.Cfg.NVocab-1))
			}
			if _, err := s2.Prefill(ids); err != nil {
				panic(err)
			}
		}
		a = bench.Case{Name: "shipped registers", Fn: step}
		b = bench.Case{Name: fmt.Sprintf("capped at %d registers", maxRegs), Fn: func(it int) { runOn(s2, it) }}
	case "attnchunk":
		// Needs -depth to mean anything. With GQA, adjacent query heads share
		// a kv head; one head per task scatters them across workers.
		// JITLLM_AB_CHUNK sets B's width: only chunk == gqa puts a whole
		// sharing group on one worker, at a cost in parallelism.
		bc := 2
		if v := os.Getenv("JITLLM_AB_CHUNK"); v != "" {
			if k, err := strconv.Atoi(v); err == nil && k > 0 {
				bc = k
			}
		}
		a = bench.Case{Name: "1 head per task", Fn: func(it int) { s.SetAttnChunk(1); step(it) }}
		b = bench.Case{Name: fmt.Sprintf("%d heads per task", bc), Fn: func(it int) { s.SetAttnChunk(bc); step(it) }}
	case "kvlayout":
		// The KV layout is baked at NewState, so it cannot be alternated
		// inside one State. This case exists only to say so.
		panic(fmt.Errorf("use -kvlayout, not -ab kvlayout: the layout is fixed " +
			"when a State is created, so it cannot be alternated inside one"))
	case "attnpair":
		// Both arms at a chunk of 2, which pairing requires, so the only
		// difference is whether one K (and V) load serves two heads.
		//
		// JITLLM_AB_PAIR selects what B shares: 1 scores, 2 accumulate, 3 both.
		bp := 3
		if v := os.Getenv("JITLLM_AB_PAIR"); v != "" {
			if k, err := strconv.Atoi(v); err == nil && k >= 1 && k <= 3 {
				bp = k
			}
		}
		// JITLLM_AB_SHIP=1 makes A the shipping configuration (one head per
		// task, unpaired), which prices the chunk change too. Do not compose
		// the two ratios.
		// JITLLM_AB_PAIR_A sets what A shares, so the marginal value of one
		// half is measured on top of the other (A=2 against B=3 prices scores
		// pairing on top of accumulate pairing).
		ap := -1
		if v := os.Getenv("JITLLM_AB_PAIR_A"); v != "" {
			if k, err := strconv.Atoi(v); err == nil && k >= 0 && k <= 3 {
				ap = k
				if k == 0 {
					ap = -1
				}
			}
		}
		an, ac := fmt.Sprintf("paired(%d), 2 heads/task", ap), 2
		if ap < 0 {
			an = "unpaired, 2 heads/task"
		}
		if os.Getenv("JITLLM_AB_SHIP") != "" {
			an, ac = "shipping: 1 head/task, unpaired", 1
		}
		a = bench.Case{Name: an, Fn: func(it int) { s.SetAttnChunk(ac); s.SetAttnPair(ap); step(it) }}
		b = bench.Case{Name: fmt.Sprintf("paired(%d), 2 heads/task", bp), Fn: func(it int) { s.SetAttnChunk(2); s.SetAttnPair(bp); step(it) }}
	case "split":
		a = bench.Case{Name: "table split", Fn: func(it int) { g.TableSplit = true; step(it) }}
		b = bench.Case{Name: "measured split", Fn: func(it int) { g.TableSplit = false; step(it) }}
	case "head":
		if !s.HeadOnDevice() {
			panic(fmt.Errorf("the device did not take the head; there is nothing to compare"))
		}
		a = bench.Case{Name: "head on host", Fn: func(it int) { s.SetHeadOnDevice(false); step(it) }}
		b = bench.Case{Name: "head in the block submission", Fn: func(it int) { s.SetHeadOnDevice(true); step(it) }}
	case "submit":
		// Graphs off on both arms: layersOnce disables capture under
		// PerLayerSubmit, so leaving graphs on would measure two changes at
		// once. The graph's own cost is -ab graph.
		g.NoGraph = true
		a = bench.Case{Name: "one submission per block", Fn: func(it int) { g.PerLayerSubmit = true; step(it) }}
		b = bench.Case{Name: "one submission per token", Fn: func(it int) { g.PerLayerSubmit = false; step(it) }}
	case "graph":
		// Both arms share the CAPTURED graph -- flipping the flag only decides
		// whether the token replays it or reissues the launches -- so no round
		// pays for a capture that the other round caused.
		a = bench.Case{Name: "launch by launch", Fn: func(it int) { g.NoGraph = true; step(it) }}
		b = bench.Case{Name: "one replayed graph", Fn: func(it int) { g.NoGraph = false; step(it) }}
	case "softmax":
		// Needs a long -n or -depth: the softmax row is pos+1 entries, so what
		// the warp reduction removes grows with context.
		if g.Stats().SoftmaxLanes != 32 {
			panic(fmt.Errorf("this device runs the scalar softmax either way; there is nothing to compare"))
		}
		a = bench.Case{Name: "1 thread per head", Fn: func(it int) { g.ScalarSoftmax = true; step(it) }}
		b = bench.Case{Name: "32 lanes per head", Fn: func(it int) { g.ScalarSoftmax = false; step(it) }}
	case "fuse":
		// Both arms' kernels are built (the fused ones beside their parts)
		// and each arm's graph is captured once, so the rounds compare the
		// launch sequences and nothing else.
		g.NoFuse = false
		step(2)
		if g.Stats().FusedLaunches == 0 {
			panic(fmt.Errorf("this device fused nothing; there is nothing to compare"))
		}
		a = bench.Case{Name: "per-op launches", Fn: func(it int) { g.NoFuse = true; step(it) }}
		b = bench.Case{Name: "fused kernels", Fn: func(it int) { g.NoFuse = false; step(it) }}
		if os.Getenv("JITLLM_AB_AA") == "1" {
			// The A/A self-control on this harness: both arms per-op.
			b = bench.Case{Name: "per-op launches (A/A)", Fn: func(it int) { g.NoFuse = true; step(it) }}
		}
	case "actfuse":
		// The operand prologue alone: every other fusion on in both arms.
		g.NoFuse, g.NoActFuse = false, false
		// Long enough for the split tuner to settle: the prologue is built
		// only for the splits it pays at (tier.actInMaxGroups).
		step(max(n, 64))
		switch st := g.Stats(); {
		case st.ActInLaunches == 0 && st.ActInTimed == 0:
			panic(fmt.Errorf("this device built no operand prologue; there is nothing to compare (%s)", g.Err()))
		case st.ActInLaunches == 0:
			// The tuner kept no site here: the two arms are one launch
			// sequence, and the row is what shipping costs on this card.
			fmt.Printf("the tuner kept none of %d sites on this device\n", st.ActInTimed)
		}
		a = bench.Case{Name: "operand by its own launch", Fn: func(it int) { g.NoActFuse = true; step(it) }}
		b = bench.Case{Name: "operand in the matvec prologue", Fn: func(it int) { g.NoActFuse = false; step(it) }}
		if os.Getenv("JITLLM_AB_AA") == "1" {
			b = bench.Case{Name: "operand by its own launch (A/A)", Fn: func(it int) { g.NoActFuse = true; step(it) }}
		}
	case "ropetab":
		// Refuse rather than compare a knob that reached nothing: a device
		// with no rotary-table kernel uploads on both arms.
		if w := g.Stats().RopeTableWhy; w != "" {
			panic(fmt.Errorf("this device uploads the rotary table either way (%s); "+
				"there is nothing to compare", w))
		}
		a = bench.Case{Name: "table uploaded from the host", Fn: func(it int) { g.RopeTableHost = true; step(it) }}
		b = bench.Case{Name: "table built on the device", Fn: func(it int) { g.RopeTableHost = false; step(it) }}
	case "persist":
		// Both arms keep their recording (the graph key carries the arm), so
		// no round pays for the other's capture or composition.
		a = bench.Case{Name: "per-op launches", Fn: func(it int) { g.Persist = false; step(it) }}
		b = bench.Case{Name: "persistent block kernels", Fn: func(it int) { g.Persist = true; step(it) }}
	case "persistdeps":
		// Dependency counters against the per-op path (Config.PersistDeps).
		a = bench.Case{Name: "per-op launches", Fn: func(it int) { g.Persist, g.PersistDeps = false, false; step(it) }}
		b = bench.Case{Name: "dependency-counter block kernels", Fn: func(it int) { g.Persist, g.PersistDeps = true, true; step(it) }}
	case "persistdeps-grid":
		// Dependency counters against grid barriers, both composed.
		a = bench.Case{Name: "grid-barrier block kernels", Fn: func(it int) { g.Persist, g.PersistDeps = true, false; step(it) }}
		b = bench.Case{Name: "dependency-counter block kernels", Fn: func(it int) { g.Persist, g.PersistDeps = true, true; step(it) }}
	case "gateup":
		a = bench.Case{Name: "gate, up, down", Fn: func(it int) { g.GateUpFuse = false; step(it) }}
		b = bench.Case{Name: "gate+up, down forming act(gate)*up", Fn: func(it int) { g.GateUpFuse = true; step(it) }}
	case "gateup-aa":
		a = bench.Case{Name: "gate, up, down", Fn: func(it int) { g.GateUpFuse = false; step(it) }}
		b = bench.Case{Name: "gate, up, down (again)", Fn: func(it int) { g.GateUpFuse = false; step(it) }}
	case "persist-aa":
		// The A/A self-control for "persist": the same arm twice through the
		// same harness (RULE 2).
		a = bench.Case{Name: "per-op launches", Fn: func(it int) { g.Persist = false; step(it) }}
		b = bench.Case{Name: "per-op launches (again)", Fn: func(it int) { g.Persist = false; step(it) }}
	default:
		panic(fmt.Errorf("jitllm: -ab takes split, submit, head, graph, softmax, fuse, actfuse, ropetab, persist, persistdeps, persistdeps-grid, persist-aa, gateup or gateup-aa, not %q", which))
	}
	r := bench.AB(a, b, rounds, n)
	fmt.Printf("\n%d blocks on device\n%s\n", s.GPULayers(), r)
	if which == "ropetab" {
		st := g.Stats()
		fmt.Printf("%d table(s) built on the device, %d uploaded\n", st.RopeTables, st.RopeTableUploads)
	}
	if strings.HasPrefix(which, "gateup") {
		fmt.Printf("gate/up fused in %d FFNs %s\n", g.Stats().GateUpFused, g.Stats().GateUpWhy)
	}
	if strings.HasPrefix(which, "persist") {
		st := g.Stats()
		fmt.Printf("persistent: %d launches (%d with dependency counters, %d tile edges) ran %d per-op launches in %d phases; %d built, %d fell back%s\n",
			st.PersistLaunches, st.PersistDepLaunches, st.PersistDepTiles, st.PersistParts, st.PersistPhases, st.PersistCompiles, st.PersistFallbacks,
			map[bool]string{true: "; declined: " + st.PersistWhy, false: ""}[st.PersistWhy != ""])
	}
	if which == "fuse" || which == "actfuse" {
		st := g.Stats()
		fmt.Printf("%d captures; %d operand-prologue launches encoded, replacing %d; %d of %d sites kept by the tuner\n",
			st.Captures, st.ActInLaunches, st.ActInSaved, st.ActInKept, st.ActInTimed)
	}
	if which == "graph" {
		fmt.Printf("%d captures over the whole A/B\n", g.Stats().Captures)
		if e := g.Err(); e != "" {
			fmt.Printf("last device error: %s\n", e)
		}
	}
	if !r.Stable() {
		fmt.Println("rejected: the spread is too wide to decide")
		return
	}
	switch {
	case r.Median > 1.02:
		fmt.Printf("%s wins by %.1f%%\n", r.B, 100*(r.Median-1))
	case r.Median < 0.98:
		fmt.Printf("%s LOSES by %.1f%%\n", r.B, 100*(1-r.Median))
	default:
		fmt.Println("no difference worth the code")
	}
}
