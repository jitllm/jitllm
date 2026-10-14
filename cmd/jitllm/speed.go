package main

import (
	"cmp"
	"flag"
	"fmt"
	"github.com/jitllm/jitllm/engine/sched"
	"maps"
	"os"
	"runtime"
	"runtime/pprof"
	"slices"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// speedCmd is llama-bench's pp/tg measurement: a prompt of -p tokens and a
// greedy decode of -n, each timed on its own, -r times on fresh States after
// one untimed warm-up of both.
//
// run's "prompt" figure is not a prefill rate: the first prompt of a process
// compiles and loads the batched kernels, which llama-bench excludes as
// warm-up.
func speedCmd(args []string) error {
	backend.SetCUDATiming(os.Getenv("JITLLM_CUDA_TIMING") == "1")
	backend.SetCUDAKernelTiming(os.Getenv("JITLLM_CUDA_KERNEL_TIMING") == "1")
	fs := flag.NewFlagSet("speed", flag.ExitOnError)
	dev := fs.String("devices", "auto", "where to run the blocks, as run's -devices")
	p := fs.Int("p", 512, "prompt tokens")
	n := fs.Int("n", 128, "generated tokens")
	reps := fs.Int("r", 3, "timed repetitions")
	vram := bytesFlag(fs, "vram", 0, "device weight budget, bytes or 3G/512M; 0 asks each device what it has free")
	cpuProf := fs.String("cpuprofile", "", "write a CPU profile of the timed prompts (not the warm-up or the decodes) here")
	tgProf := fs.String("tgprofile", "", "write a CPU profile of the timed decodes here")
	gcStats := fs.Bool("gcstats", false, "print what the Go collector did during each timed prompt and decode: allocations a token, cycles, pauses, GC CPU")
	memProf := fs.String("memprofile", "", "write allocation profiles at every allocation (MemProfileRate 1) around the first timed prompt and decode, as PREFIX.{0,pp,tg0,tg}; diff with pprof -base")
	ttft := fs.Bool("ttft", false, "time to first token instead of pp/tg: cold (from opening the model to the "+
		"first prompt's first token) and warm (a fresh session tokenizing a -p token prompt, prefilling it and "+
		"sampling, -r times)")
	sessions := fs.Int("sessions", 0, "decode N sessions -n tokens, one after another and together "+
		"(model.Step), interleaved over -r rounds")
	spec := addSpecFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	specOpts, err := spec.options()
	if err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: jitllm speed [-devices SPEC] [-p N] [-n N] [-r N] [-vram B] <model.jlm>")
	}
	if *memProf != "" {
		runtime.MemProfileRate = 1
	}
	t00 := time.Now() // cold time to first token counts from here
	if *ttft && *cpuProf != "" {
		// -ttft profiles the cold path, open to the first token; ttftRun stops it.
		f, err := os.Create(*cpuProf)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
	}
	m, err := model.Open(fs.Arg(0), loadOpts(nil, 0)...)
	if err != nil {
		return err
	}
	defer m.Close()
	if err := preloadWait(m); err != nil {
		return err
	}
	tOpen := time.Since(t00)
	d, closeDev, name, err := openDevices(*dev, *vram, 0, m.StreamGroups())
	if err != nil {
		return err
	}
	tDev := time.Since(t00) - tOpen
	if closeDev != nil {
		defer closeDev()
	}
	seed := m.Vocab.Encode("The capital of France is", true)
	prompt := make([]int32, *p)
	for i := range prompt {
		prompt[i] = seed[i%len(seed)]
	}
	// attach puts a fresh State of n positions on the device.
	attach := func(n int) (*model.State, error) {
		st := m.NewState(n)
		if d != nil {
			if err := st.SetDevice(d); err != nil {
				st.Close()
				return nil, err
			}
		}
		return st, nil
	}
	one := func(timed bool) (pp, tg time.Duration, err error) {
		// The prompt's State is closed before the decode's is opened: two
		// live sessions each hold a KV cache on every card, and the tier does
		// not budget a second one. llama-bench's pp and tg are separate runs.
		st, err := attach(*p + 1)
		if err != nil {
			return 0, 0, err
		}
		if timed && *cpuProf != "" {
			f, perr := os.Create(*cpuProf)
			if perr != nil {
				return 0, 0, perr
			}
			defer f.Close()
			// 2 kHz: a prompt is a few hundred ms, and the host's share of it
			// sits in gaps of a few ms that 100 Hz samples cannot see.
			runtime.SetCPUProfileRate(2000)
			sched.SetRegionLabels(true) // attribution for generated code; see sched.regionLabels
			if perr := pprof.StartCPUProfile(f); perr != nil {
				return 0, 0, perr
			}
		}
		if timed && *memProf != "" {
			if perr := writeAllocProfile(*memProf + ".0"); perr != nil {
				return 0, 0, perr
			}
		}
		backend.CUDAKernelTimes() // clear: only the timed prompt is reported
		g0 := readGC()
		t0 := time.Now()
		// The timer stops when Prefill returns: it hands back the last row's
		// logits read from the device, so the device has finished; that is
		// llama-bench's pp boundary (decode of the batch with the last row's
		// logits, then a synchronize), with no sampling and no tokenizing.
		lg, err := st.Prefill(prompt)
		pp = time.Since(t0)
		if err == nil {
			err = warmHonest(st, len(prompt), lg, m.Cfg.NVocab)
		}
		if *gcStats {
			gcDelta(fmt.Sprintf("pp%d", *p), g0, readGC(), *p)
		}
		if timed && *memProf != "" {
			if perr := writeAllocProfile(*memProf + ".pp"); perr != nil {
				return 0, 0, perr
			}
		}
		if timed && *cpuProf != "" {
			pprof.StopCPUProfile()
		}
		if timed && err == nil {
			printKernelTimes("a prompt", 1)
		}
		ppOn, spills, fails, tails := st.GPULayers(), st.Spills(), st.DeviceDemotions(), st.TailRowChunks()
		moves := st.Relocations() + st.Reclaims()
		st.Close()
		if err != nil {
			return 0, 0, err
		}
		// The decode starts on a fresh State at position 1, as llama-bench's
		// tg does, so it is not a decode at the prompt's depth. Speculation
		// starts from the seed sentence instead: a prediction block's
		// acceptance is a property of the text, and a one-token prompt has
		// none to speak of.
		st2, err := attach(*n + len(seed) + 2)
		if err != nil {
			return 0, 0, err
		}
		defer st2.Close()
		var sp *model.Speculator
		var next int32
		if specOpts != nil {
			if sp, err = st2.Speculate(specOpts...); err != nil {
				return 0, 0, err
			}
			defer sp.Close()
			if next, err = sp.Start(seed, nil); err != nil {
				return 0, 0, err
			}
		} else if next, err = st2.ForwardGreedy(seed[0]); err != nil {
			return 0, 0, err
		}
		if timed && *tgProf != "" {
			f, perr := os.Create(*tgProf)
			if perr != nil {
				return 0, 0, perr
			}
			defer f.Close()
			sched.SetRegionLabels(true) // attribution for generated code; see sched.regionLabels
			if perr := pprof.StartCPUProfile(f); perr != nil {
				return 0, 0, perr
			}
			defer pprof.StopCPUProfile()
		}
		backend.CUDAGraphTimes() // clear: only the timed decode is reported
		backend.CUDAKernelTimes()
		if timed && *memProf != "" {
			if perr := writeAllocProfile(*memProf + ".tg0"); perr != nil {
				return 0, 0, perr
			}
		}
		g0 = readGC()
		t0 = time.Now()
		if sp != nil {
			for got := 0; got < *n; {
				toks, err := sp.Next(nil)
				if err != nil {
					return 0, 0, err
				}
				got += len(toks)
			}
		} else {
			for i := 0; i < *n; i++ {
				if next, err = st2.ForwardGreedy(next); err != nil {
					return 0, 0, err
				}
			}
		}
		tg = time.Since(t0)
		if sp != nil && timed {
			printSpec(os.Stdout, sp)
		}
		if *gcStats {
			gcDelta(fmt.Sprintf("tg%d", *n), g0, readGC(), *n)
		}
		if timed && *memProf != "" {
			if perr := writeAllocProfile(*memProf + ".tg"); perr != nil {
				return 0, 0, perr
			}
		}
		if timed {
			printKernelTimes("a token", *n)
		} else {
			backend.CUDAKernelTimes()
		}
		if busy, gap := backend.CUDAGraphTimes(); timed && len(busy) > 0 {
			// JITLLM_CUDA_TIMING=1: the device clock's view of the decode.
			fmt.Printf("cuda graph timing: %d replays, device %.1f us, gap before %.1f us (medians); wall %.1f us/token\n",
				len(busy), medianF32(busy), medianF32(gap), float64(tg.Microseconds())/float64(*n))
		}
		if timed && d != nil {
			fmt.Printf("device %s: prompt on %d/%d blocks, decode on %d, %d spill(s), %d relocation(s)+reclaim(s), %s\n", name,
				ppOn, m.Cfg.NLayer, st2.GPULayers(), spills+st2.Spills(),
				moves+st2.Relocations()+st2.Reclaims(), deviceStats(d))
			if tails > 0 {
				// engine/model/tailrows.go
				fmt.Printf("prompt: %d few-row chunk(s) ran as a ragged step\n", tails)
			}
		}
		// A demotion moves the blocks to the host mid-run: the rate is then the
		// host's, and printing it without saying so is a wrong row.
		if n := fails + st2.DeviceDemotions(); n > 0 && d != nil {
			fmt.Printf("★ the device failed %d time(s) and its blocks moved to the host: %s\n", n, deviceErr(d))
		}
		return pp, tg, nil
	}
	if *ttft {
		return ttftRun(m, attach, *p, *reps, t00, tOpen, tDev)
	}
	if *sessions > 0 {
		return sessionsRun(attach, prompt, *sessions, *n, *reps)
	}
	if _, _, err := one(false); err != nil {
		return err
	}
	var pps, tgs []float64
	for r := 0; r < *reps; r++ {
		pp, tg, err := one(r == 0)
		if err != nil {
			return err
		}
		ppr, err := perSecond(fmt.Sprintf("pp%d", *p), *p, pp)
		if err != nil {
			return err
		}
		// -n 0 times no decode (the prefill harness's warm row): its span is
		// whatever the clock read around nothing, zero on some runs, and a
		// refusal there threw away the prompt's rate with it.
		tgr := 0.0
		if *n > 0 {
			if tgr, err = perSecond(fmt.Sprintf("tg%d", *n), *n, tg); err != nil {
				return err
			}
		}
		pps = append(pps, ppr)
		tgs = append(tgs, tgr)
	}
	// The warm-up policy goes beside the rates, so a row says what it is.
	fmt.Printf("warm prefill, pp%d, no prefix reuse: 1 untimed warm-up prompt and decode, then %d timed, "+
		"every sample reported; each a fresh State at position 0, all %d rows computed (warmHonest), the "+
		"last row's logits read back before the timer stops\n", *p, *reps, *p)
	fmt.Printf("pp%d %s tok/s   tg%d %s tok/s\n", *p, rates(pps), *n, rates(tgs))
	return nil
}

// medianF32 is the median of v's non-negative entries, or -1.
func medianF32(v []float32) float32 {
	var w []float32
	for _, x := range v {
		if x >= 0 {
			w = append(w, x)
		}
	}
	if len(w) == 0 {
		return -1
	}
	slices.Sort(w)
	return w[len(w)/2]
}

// sessionsRun is speed -sessions: k sessions on the device, each prefilled
// with prompt, decode n tokens a round, alternately one session after another
// and all together through model.Step. Each arm's rate is the k sessions'
// tokens over its wall.
func sessionsRun(attach func(int) (*model.State, error), prompt []int32, k, n, reps int) error {
	var alone, together []float64
	for r := 0; r < 2*reps; r++ {
		joint := r%2 == 1
		sts := make([]*model.State, k)
		toks := make([]int32, k)
		for i := range sts {
			// Closed at the end of the round; an error ends the process.
			st, err := attach(len(prompt) + n + 2)
			if err != nil {
				return err
			}
			lg, err := st.Prefill(prompt)
			if err != nil {
				return err
			}
			sts[i], toks[i] = st, model.Greedy(lg)
		}
		backend.CUDAKernelTimes() // clear: only this arm's steps are reported
		t0 := time.Now()
		for range n {
			if joint {
				lgs, err := model.Step(sts, toks)
				if err != nil {
					return err
				}
				for i, lg := range lgs {
					toks[i] = model.Greedy(lg)
				}
				continue
			}
			for i, st := range sts {
				lg, err := st.Forward(toks[i])
				if err != nil {
					return err
				}
				toks[i] = model.Greedy(lg)
			}
		}
		rate, err := perSecond(fmt.Sprintf("%d sessions x tg%d", k, n), k*n, time.Since(t0))
		if err != nil {
			return err
		}
		if joint {
			printKernelTimes(fmt.Sprintf("a step of %d sessions together", k), n)
		} else {
			printKernelTimes(fmt.Sprintf("a step of %d sessions one after another", k), n)
		}
		if joint {
			together = append(together, rate)
		} else {
			alone = append(alone, rate)
		}
		for _, st := range sts {
			st.Close()
		}
	}
	slices.Sort(alone)
	slices.Sort(together)
	fmt.Printf("sessions %d x tg%d: one after another %s tok/s, together %s tok/s (%.2fx)\n", k, n,
		rates(alone), rates(together), together[len(together)/2]/alone[len(alone)/2])
	return nil
}

// printKernelTimes prints and clears the per-kernel device time CUDA recorded
// (JITLLM_CUDA_KERNEL_TIMING=1), per unit over steps units; it prints nothing
// when nothing was recorded. A decode token's are event nodes inside its
// captured graph; a prompt's are an event pair around each launch, since a
// prompt chunk is not captured (scripts/kshim.c with KSHIM_DIRECT=1 takes the
// same of llama.cpp's prompt).
func printKernelTimes(unit string, steps int) {
	us, cnt := backend.CUDAKernelTimes()
	if len(us) == 0 {
		return
	}
	names := slices.SortedFunc(maps.Keys(us), func(a, b string) int { return cmp.Compare(us[b], us[a]) })
	tot := 0.0
	for _, k := range names {
		tot += us[k]
	}
	fmt.Printf("cuda kernel timing: %.1f us of kernels %s\n", tot/float64(steps), unit)
	for _, k := range names[:min(25, len(names))] {
		fmt.Printf("  %9.1f us %6.1f  %s\n", us[k]/float64(steps), float64(cnt[k])/float64(steps), k)
	}
}

// perSecond is count over d, refusing a run the clock did not see. Go's clock
// on Windows advances once per timer interrupt, 0.5 to 15.6 ms, so a short run
// can measure zero, and count/0 printed "+Inf tok/s" as if it were a rate.
func perSecond(what string, count int, d time.Duration) (float64, error) {
	if d <= 0 {
		return 0, fmt.Errorf("%s ran in less time than the clock resolves; time more tokens", what)
	}
	return float64(count) / d.Seconds(), nil
}

func rates(v []float64) string {
	s := ""
	for i, x := range v {
		if i > 0 {
			s += " / "
		}
		s += fmt.Sprintf("%.2f", x)
	}
	return s
}

// deviceErr is the tier's last refusal, or "" for a device that has none.
func deviceErr(d nn.Device) string {
	if g, ok := d.(*tier.GPU); ok {
		return g.Err()
	}
	return ""
}

// ttftText is -ttft's prompt: prose repeated until it tokenizes to at least p
// tokens, so the time to first token includes tokenizing it, as a request's
// does.
func ttftText(m *model.Model, p int) string {
	const s = "The river rose every spring until the bridges were islands, and the clerk counted barrels of salt on the upper floor. "
	t := s
	for len(m.Vocab.Encode(t, true)) < p {
		t += s
	}
	return t
}

// ttftRun is speed -ttft: a request is text in and one token out. The first is
// the process's -- the container opened, the blocks placed and uploaded, the
// kernels compiled -- and each after it is a fresh session on the loaded
// model: tokenize, take pages, prefill, sample.
func ttftRun(m *model.Model, attach func(int) (*model.State, error), p, reps int, t00 time.Time, tOpen, tDev time.Duration) error {
	text := ttftText(m, p)
	var tAttach time.Duration
	var compiled, tuned int64
	first := func() (time.Duration, int, error) {
		t0 := time.Now()
		ids := m.Vocab.Encode(text, true)
		st, err := attach(len(ids) + 1)
		if err != nil {
			return 0, 0, err
		}
		tAttach = time.Since(t0)
		defer st.Close()
		// Ready: the blocks are placed, every kernel compiled and the tuners
		// done. What follows, to the first token, must do none of either.
		c0, u0 := backend.Compiles(), tier.TuneTrials()
		lg, err := st.Prefill(ids)
		if err != nil {
			return 0, 0, err
		}
		model.Greedy(lg)
		compiled, tuned = backend.Compiles()-c0, tier.TuneTrials()-u0
		return time.Since(t0), len(ids), nil
	}
	el, _, err := first()
	if err != nil {
		return err
	}
	cold := time.Since(t00)
	pprof.StopCPUProfile()
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }
	fmt.Printf("cold: open %.0f ms, devices %.0f ms, attach %.0f ms (placement and upload), prompt %.0f ms\n",
		ms(tOpen), ms(tDev), ms(tAttach), ms(el-tAttach))
	// The two rows vs-llamacpp.sh reads: load is open to ready (the container
	// opened, the devices up, the kernels compiled, the tuners run, the blocks
	// placed and uploaded); ttft (model loaded) is ready to the first token of
	// the process's first prompt, tokenizing excluded as llama.cpp's prompt
	// eval excludes it.
	if err := readyHonest(compiled, tuned); err != nil {
		return err
	}
	tStart := t00.Sub(procStart)
	fmt.Printf("load %.1f ms (start %.0f + open %.0f + devices %.0f + attach %.0f)   ttft-loaded %.1f ms (ready to the first token; %d compiles, %d tuner trials after ready; %d background tuner trials so far)\n",
		ms(tStart+tOpen+tDev+tAttach), ms(tStart), ms(tOpen), ms(tDev), ms(tAttach), ms(el-tAttach), compiled, tuned, tier.BackgroundTuneTrials())
	var warm []float64
	ntok := 0
	for r := 0; r < reps; r++ {
		el, n, err := first()
		if err != nil {
			return err
		}
		warm, ntok = append(warm, float64(el.Microseconds())/1e3), n
	}
	slices.Sort(warm)
	fmt.Printf("ttft %d tok   cold %.1f ms (open to the first prompt's token)   warm %s ms (median %.2f, a fresh session)\n",
		ntok, float64(cold.Microseconds())/1e3, rates(warm), warm[len(warm)/2])
	return nil
}

// warmHonest refuses a timed prompt that was not the whole prompt computed
// from nothing: a fresh State at position 0 that ran every row through its
// blocks (none restored from a prefix cache) and handed back the last row's
// logits, which the device must have finished to return.
func warmHonest(st *model.State, n int, logits []float32, vocab int) error {
	switch {
	case st.KVRestored() != 0:
		return fmt.Errorf("speed: the timed prompt restored %d position(s) from a prefix cache", st.KVRestored())
	case st.PrefillRows() != n || st.Pos() != n:
		return fmt.Errorf("speed: the timed prompt ran %d of %d rows and ended at position %d", st.PrefillRows(), n, st.Pos())
	case len(logits) != vocab:
		return fmt.Errorf("speed: the timed prompt returned %d logits, not the %d of a row read back", len(logits), vocab)
	}
	return nil
}

// readyHonest refuses a ready mark that came too early: a kernel compiled or
// a tuner trial timed between ready and the first token would put load work
// inside the ttft row.
func readyHonest(compiled, tuned int64) error {
	if compiled != 0 || tuned != 0 {
		return fmt.Errorf("speed: %d kernel compile(s) and %d tuner trial(s) ran after the model was ready", compiled, tuned)
	}
	return nil
}
