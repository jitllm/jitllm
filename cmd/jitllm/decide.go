package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/model"
)

// decideCmd is `jitllm decide <model.jlm> <request.json|->`: a decision
// model's answers to a TypeSafe /v1/systemone request body, printed as the
// response body the server returns for it (docs/design/decision-models.md).
// With -lines, every line of stdin is one request and gets one response line.
func decideCmd(args []string) error {
	fs := flag.NewFlagSet("decide", flag.ExitOnError)
	lines := fs.Bool("lines", false, "answer every line of stdin as one request")
	maxSeq := fs.Int("max-seq", 8192, "the longest prompt one question may render to, in tokens")
	dev := fs.String("devices", "auto", "where the blocks run, as run's -devices")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() != 2 && !(*lines && fs.NArg() == 1) {
		return fmt.Errorf("usage: jitllm decide [-lines] [-max-seq N] <model.jlm> <request.json|->")
	}
	t0 := time.Now()
	mo, eo := loadOpts(nil, 0)
	m, err := model.Open(fs.Arg(0), mo...)
	if err != nil {
		return err
	}
	defer m.Close()
	in, err := engine.NewInstance(m, *maxSeq, eo...)
	if err != nil {
		return err
	}
	defer in.Close()
	g, closeDev, devName, err := openDevices(*dev, 0, 0, m.StreamGroups(), 0)
	if err != nil {
		return err
	}
	defer closeDev()
	if g != nil {
		if err := in.SetDevice(g); err != nil {
			return err
		}
	}
	d, err := engine.NewDecider(in)
	if err != nil {
		return fmt.Errorf("%s: %w", fs.Arg(0), err)
	}
	fmt.Fprintf(os.Stderr, "%s: %s, %d blocks on %s, loaded in %v\n",
		fs.Arg(0), m.Cfg.Arch, in.GPULayers(), devName, time.Since(t0).Round(time.Millisecond))
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	one := func(body []byte) error {
		_, state, qs, err := engine.ParseDecisionRequest(body)
		if err != nil {
			return err
		}
		t := time.Now()
		ans, err := d.Decide(state, qs)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%d questions, %d tokens in %v\n", len(qs), d.InputTokens(), time.Since(t).Round(time.Microsecond))
		if _, err := out.Write(d.DecisionResponse(fs.Arg(0), qs, ans)); err != nil {
			return err
		}
		return out.WriteByte('\n')
	}
	if !*lines {
		var body []byte
		if fs.Arg(1) == "-" {
			body, err = io.ReadAll(os.Stdin)
		} else {
			body, err = os.ReadFile(fs.Arg(1))
		}
		if err != nil {
			return err
		}
		return one(body)
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		if err := one(sc.Bytes()); err != nil {
			return err
		}
		if err := out.Flush(); err != nil {
			return err
		}
	}
	return sc.Err()
}
