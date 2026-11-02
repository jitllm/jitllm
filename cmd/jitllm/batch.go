package main

import (
	"flag"
	"fmt"
	"github.com/jitllm/jitllm/engine/sched"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/jitllm/jitllm/engine/model"
)

// batchCmd decodes one prompt as `-batch` independent sequences in lockstep on
// one device (model.State.ForwardBatchGreedy, or ForwardBatch with -logits) and
// reports the aggregate rate:
// the concurrent-serving number, where run reports one conversation's.
//
// Every sequence gets the same prompt, fed a token per step like everything
// after it. The rate is generated tokens over the whole wall, prompt included --
// vLLM's end-to-end accounting, so the two print comparable numbers.
func batchCmd(args []string) error {
	fs := flag.NewFlagSet("batch", flag.ExitOnError)
	dev := fs.String("devices", "gpu:0", "the one device every block and the head go on")
	nseq := fs.Int("batch", 16, "sequences decoded together")
	n := fs.Int("n", 128, "tokens to generate per sequence")
	logits := fs.Bool("logits", false, "read every row's logits back and pick the token on the host "+
		"(ForwardBatch), the path a sampled request takes, instead of the device's argmax")
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
		sched.SetRegionLabels(true) // attribution for generated code; see sched.regionLabels
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
	m, err := model.Open(fs.Arg(0), loadOpts(nil, 0)...)
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
	// decode runs the whole batch on a fresh State and returns the wall.
	// The same capacity both times, so the timed run grows nothing.
	gen := *n
	decode := func(n int) (time.Duration, error) {
		st := m.NewBatch(*nseq, len(ids)+gen+1)
		defer st.Close()
		if err := st.SetDevice(d); err != nil {
			return 0, err
		}
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			return 0, fmt.Errorf("device %s took %d/%d blocks, head %v: batch needs all of them",
				name, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice())
		}
		feed := make([]int32, *nseq)
		var out []int32
		t0 := time.Now()
		for step := 0; step < len(ids)+n-1; step++ {
			for i := range feed {
				if step < len(ids) {
					feed[i] = ids[step]
				} else {
					feed[i] = out[i]
				}
			}
			if !*logits {
				if out, err = st.ForwardBatchGreedy(feed); err != nil {
					return 0, err
				}
				continue
			}
			if _, err = st.ForwardBatch(feed); err != nil {
				return 0, err
			}
			if out == nil {
				out = make([]int32, *nseq)
			}
			for i := range out {
				out[i] = st.Sample(nil, st.BatchLogits(i))
			}
		}
		return time.Since(t0), nil
	}
	// A warm-up first, as vLLM's and llama.cpp's benchmarks do: the first step
	// compiles every ragged matvec and captures the graph.
	if _, err := decode(4); err != nil {
		return err
	}
	steps := len(ids) + *n - 1
	el, err := decode(*n)
	if err != nil {
		return err
	}
	total := float64(*nseq * *n)
	arm := "device argmax"
	if *logits {
		arm = "logits read back"
	}
	fmt.Printf("%d sequences x %d generated (+%d prompt steps, %s): %.2f tok/s aggregate, "+
		"%.2f per sequence, %.2f ms per step\n", *nseq, *n, len(ids)-1, arm,
		total/el.Seconds(), float64(*n)/el.Seconds(), el.Seconds()*1000/float64(steps))
	return nil
}
