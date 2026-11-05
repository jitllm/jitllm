package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/localserver"
)

// embedCmd is `jitllm embed <model.jlm> <text...>`: the text's pooled,
// L2-normalised embedding, printed as one JSON array on stdout.
//
// It reads one vector out of the last residual and never computes a logit. The
// container says which readout (mean, first, last) and whether attention looks
// both ways, so there is no flag for either: the wrong pooling would give a
// vector the weights were never trained to produce.
//
// With -lines, every line of stdin is embedded in turn and printed one array
// per line, which is how a corpus is indexed without paying the load per text.
func embedCmd(args []string) error {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	showIDs := fs.Bool("ids", false, "print the token ids to stderr")
	lines := fs.Bool("lines", false, "embed every line of stdin instead of the text arguments")
	dev := fs.String("devices", "auto", "where the blocks run, as run's -devices")
	layers := fs.Int("gpu-layers", -1, "cap the blocks offered to the device; -1 fits as many as the card holds")
	addr := fs.String("addr", localserver.DefaultAddr, "the jitllmd a model NAME is embedded through when one answers there")
	apiKey := fs.String("api-key", "", "the API key the jitllmd wants, if it has one (default: $JITLLM_API_KEY)")
	embedded := fs.Bool("embedded", false, "embed in this process even when a jitllmd answers")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() < 1 || (fs.NArg() < 2 && !*lines) {
		return fmt.Errorf("usage: jitllm embed [-ids] [-lines] <model.jlm|NAME> <text...>")
	}
	svc, path, err := runTarget(context.Background(), fs.Arg(0), *addr, keyOf(*apiKey), *embedded, engineFlags(fs, embedRemoteFlags()), os.Stderr)
	if err != nil {
		return err
	}
	if svc != nil {
		texts := []string{strings.Join(fs.Args()[1:], " ")}
		if *lines {
			texts = texts[:0]
			sc := bufio.NewScanner(os.Stdin)
			sc.Buffer(make([]byte, 1<<20), 1<<26)
			for sc.Scan() {
				texts = append(texts, sc.Text())
			}
			if err := sc.Err(); err != nil {
				return err
			}
		}
		return embedRemote(svc.Addr, keyOf(*apiKey), fs.Arg(0), texts, os.Stdout)
	}
	t0 := time.Now()
	mo, io := loadOpts(nil, 0)
	m, err := model.Open(path, mo...)
	if err != nil {
		return err
	}
	defer m.Close()
	if !m.IsEmbedding() {
		return fmt.Errorf("%s is not an embedding model: its container sets no pooling "+
			"(generate from it with `jitllm run`)", fs.Arg(0))
	}
	if m.Vocab == nil {
		return fmt.Errorf("%s: no tokenizer: %v", fs.Arg(0), m.TokErr)
	}
	e, err := engine.NewEmbedder(m, io...)
	if err != nil {
		return err
	}
	defer e.Close()
	g, closeDev, devName, err := openDevices(*dev, 0, 0, m.StreamGroups(), 0)
	if err != nil {
		return err
	}
	defer closeDev()
	if g != nil {
		if err := e.SetDeviceLayers(g, *layers); err != nil {
			return err
		}
		defer e.SetDeviceLayers(nil, 0)
	}
	fmt.Fprintf(os.Stderr, "%s: %s, %d dims, %v pooling, %d blocks on %s, loaded in %v\n",
		fs.Arg(0), m.Cfg.Arch, m.Cfg.NEmbd, m.Pooling(), e.DeviceBlocks(), devName, time.Since(t0).Round(time.Millisecond))
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	one := func(text string) error {
		ids := m.EmbedIDs(text)
		if *showIDs {
			fmt.Fprintln(os.Stderr, ids)
		}
		t := time.Now()
		v, err := e.Embed(ids)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%d tokens in %v\n", len(ids), time.Since(t).Round(time.Microsecond))
		b := make([]byte, 0, len(v)*12+2)
		b = append(b, '[')
		for i, x := range v {
			if i > 0 {
				b = append(b, ',')
			}
			b = strconv.AppendFloat(b, float64(x), 'g', 8, 32)
		}
		b = append(b, ']', '\n')
		_, err = out.Write(b)
		return err
	}
	if !*lines {
		return one(strings.Join(fs.Args()[1:], " "))
	}
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		if err := one(sc.Text()); err != nil {
			return err
		}
	}
	return sc.Err()
}
