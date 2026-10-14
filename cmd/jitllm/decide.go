package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"io"
	"os"
	"runtime/pprof"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
)

// decideCmd is `jitllm decide <model.jlm> <request.json|->`: a decision
// model's answers to a TypeSafe /v1/systemone request body, printed as the
// response body the server returns for it (docs/design/decision-models.md).
//
// With -lines, every line of stdin is one request and gets one response line,
// which is how a stream of decisions is answered without paying the load per
// request.
func decideCmd(args []string) error {
	fs := flag.NewFlagSet("decide", flag.ExitOnError)
	lines := fs.Bool("lines", false, "answer every line of stdin as one request")
	maxSeq := fs.Int("max-seq", 8192, "the longest prompt one question may render to, in tokens")
	repeat := fs.Int("repeat", 1, "answer each request this many times, to tell a cold call from a warm one")
	cpuProf := fs.String("cpuprofile", "", "write a CPU profile of the answering (not the load) here")
	dev := fs.String("devices", "auto", "where the blocks run, as run's -devices")
	layers := fs.Int("gpu-layers", -1, "cap the blocks offered to the device; -1 fits as many as the card holds")
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
	m, err := model.Open(fs.Arg(0), loadOpts(nil, 0)...)
	if err != nil {
		return err
	}
	defer m.Close()
	if m.Decision() == jlm.DecisionNone {
		return fmt.Errorf("%s is not a decision model: its container states no decision readout", fs.Arg(0))
	}
	d, err := m.NewDecider(*maxSeq)
	if err != nil {
		return err
	}
	defer d.Close()
	g, closeDev, devName, err := openDevices(*dev, 0, 0, m.StreamGroups())
	if err != nil {
		return err
	}
	defer closeDev()
	if g != nil {
		if err := d.SetDevice(g, *layers); err != nil {
			return err
		}
		defer d.SetDevice(nil, 0)
	}
	fmt.Fprintf(os.Stderr, "%s: %s, %v readout, %d blocks on %s, loaded in %v\n",
		fs.Arg(0), m.Cfg.Arch, m.Decision(), d.DeviceBlocks(), devName, time.Since(t0).Round(time.Millisecond))
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
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	one := func(body []byte) error {
		req, err := model.ParseDecisionRequest(body)
		if err != nil {
			return err
		}
		state, qs := req.State, req.Questions
		imgs := make([]image.Image, len(req.Images))
		for i, u := range req.Images {
			if imgs[i], err = model.DecodeImageDataURL(u); err != nil {
				return fmt.Errorf("images[%d]: %w", i, err)
			}
		}
		vids := make([]model.VideoFrames, len(req.Videos))
		for i, v := range req.Videos {
			vids[i].FPS = v.FPS
			for j, u := range v.Frames {
				f, err := model.DecodeImageDataURL(u)
				if err != nil {
					return fmt.Errorf("videos[%d] frame %d: %w", i, j, err)
				}
				vids[i].Frames = append(vids[i].Frames, f)
			}
		}
		var clip []float32
		if len(req.Audio) > 0 {
			if clip, err = model.DecodeAudioDataURL(req.Audio[0]); err != nil {
				return err
			}
		}
		var ans []model.DecisionAnswer
		for r := 0; r < max(1, *repeat); r++ {
			t, reads, read := time.Now(), m.PagerReads(), m.BytesRead()
			if clip != nil {
				ans, err = d.DecideAudio(state, qs, clip)
			} else {
				ans, err = d.DecideMedia(state, qs, imgs, vids)
			}
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "%d questions, %d tokens in %v; %d page reads, %d MiB read\n", len(qs), d.InputTokens(),
				time.Since(t).Round(time.Microsecond), m.PagerReads()-reads, (m.BytesRead()-read)>>20)
		}
		if _, err := out.Write(model.DecisionResponse(fs.Arg(0), qs, ans, d.InputTokens())); err != nil {
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
