package main

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/probe"
)

// deviceReadWall measures the first device of api's read bandwidth in this
// process, over a working set of mib MiB: the denominator of a decode row's
// "fraction of the wall" (RULE 1), taken beside the row rather than carried
// from another session (probe.ReadRate), in GB/s.
func deviceReadWall(api string, mib, rounds int) (float64, string, error) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	for _, d := range devs {
		if d.API() == api {
			rate, err := probe.ReadRate(d, int64(mib)<<20, rounds)
			return rate / 1e9, d.Name(), err
		}
	}
	return 0, "", fmt.Errorf("no %s device", api)
}
