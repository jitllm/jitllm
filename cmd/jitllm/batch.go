package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine"
)

// batchCmd decodes one prompt as `-batch` sessions leasing one instance on one
// device, continuously batched (engine.Batcher), and reports the aggregate
// rate: the concurrent-serving number, where run reports one conversation's.
//
// Every session gets the same prompt, admitted in chunks beside the decoding
// rows. The rate is generated tokens over the whole wall, prompt included --
// vLLM's end-to-end accounting, so the two print comparable numbers.
func batchCmd(args []string) error {
	fs := flag.NewFlagSet("batch", flag.ExitOnError)
	dev := fs.String("devices", "gpu:0", "the one device every block and the head go on")
	nseq := fs.Int("batch", 16, "sequences decoded together")
	n := fs.Int("n", 128, "tokens to generate per sequence")
	cpuProf := fs.String("cpuprofile", "", "write a CPU profile here")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cpuProf != "" {
		f, err := os.Create(*cpuProf)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() < 2 {
		usage()
	}
	mo, io := loadOpts(nil, 0)
	io = append(io, regionLabels(*cpuProf != "")...)
	m, err := model.Open(fs.Arg(0), mo...)
	if err != nil {
		return err
	}
	defer m.Close()
	ids := m.Vocab.Encode(strings.Join(fs.Args()[1:], " "), true)
	d, closeDev, name, err := openDevices(*dev, 0, 0, m.StreamGroups(), 0)
	if err != nil {
		return err
	}
	if d == nil {
		return fmt.Errorf("batch needs a device; -devices %q opened none", *dev)
	}
	defer closeDev()
	// The instance is composed once, here; the sessions lease it.
	capacity := len(ids) + *n + 1
	in, err := engine.NewInstance(m, capacity, io...)
	if err != nil {
		return err
	}
	defer in.Close()
	if err := in.SetDevice(d); err != nil {
		return err
	}
	if in.GPULayers() != m.Cfg.NLayer || !in.HeadOnDevice() {
		return fmt.Errorf("device %s took %d/%d blocks, head %v: batch needs all of them",
			name, in.GPULayers(), m.Cfg.NLayer, in.HeadOnDevice())
	}
	pool := engine.NewPool()
	if err := pool.Serve(in, *nseq); err != nil {
		return err
	}
	// decode runs the whole batch and returns the wall and the steps.
	decode := func(n int) (time.Duration, int, error) {
		b := engine.NewBatcher(m, pool, *nseq, capacity)
		defer b.Close()
		for range *nseq {
			b.Submit(&engine.Seq{Prompt: ids, MaxTokens: n})
		}
		t0, steps := time.Now(), 0
		for retired := 0; retired < *nseq; steps++ {
			done, err := b.Step()
			if err != nil {
				return 0, steps, err
			}
			retired += len(done)
		}
		return time.Since(t0), steps, nil
	}
	// A warm-up first, as vLLM's and llama.cpp's benchmarks do: the first step
	// compiles every ragged matvec and captures the graph.
	if _, _, err := decode(4); err != nil {
		return err
	}
	el, steps, err := decode(*n)
	if err != nil {
		return err
	}
	total := float64(*nseq * *n)
	fmt.Printf("%d sequences x %d generated (%d steps, the prompt in chunks): %.2f tok/s aggregate, "+
		"%.2f per sequence, %.2f ms per step\n", *nseq, *n, steps,
		total/el.Seconds(), float64(*n)/el.Seconds(), el.Seconds()*1000/float64(steps))
	return nil
}
