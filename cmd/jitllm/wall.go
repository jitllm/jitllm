package main

import (
	"encoding/binary"
	"fmt"
	"sort"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// deviceReadWall measures the first device of api's read bandwidth in this
// process, over a working set of mib MiB: the denominator of a decode row's
// "fraction of the wall" (RULE 1), taken beside the row rather than carried
// from another session. It streams one word a thread a step, grid-strided,
// and reports the median of rounds launches in GB/s.
func deviceReadWall(api string, mib, rounds int) (float64, string, error) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	var d backend.Device
	for _, x := range devs {
		if x.API() == api {
			d = x
			break
		}
	}
	if d == nil {
		return 0, "", fmt.Errorf("no %s device", api)
	}
	const threads = 256 * 1024
	per := max((mib<<20)/4/threads, 1)
	n := threads * per
	k, err := kernels.StreamWide(per, 1)
	if err != nil {
		return 0, "", err
	}
	kern, err := d.Compile(k)
	if err != nil {
		return 0, "", err
	}
	defer kern.Close()
	in, err := d.Alloc(n * 4)
	if err != nil {
		return 0, "", err
	}
	defer in.Free()
	if err := in.Write(make([]byte, n*4)); err != nil {
		return 0, "", err
	}
	np, err := d.Alloc(4)
	if err != nil {
		return 0, "", err
	}
	defer np.Free()
	if err := np.Write(binary.LittleEndian.AppendUint32(nil, threads)); err != nil {
		return 0, "", err
	}
	out, err := d.Alloc(threads * 4)
	if err != nil {
		return 0, "", err
	}
	defer out.Free()
	run := func() error { return kern.Launch(threads/256, 256, in, np, out) }
	for i := 0; i < 5; i++ {
		if err := run(); err != nil {
			return 0, "", err
		}
	}
	var rs []float64
	for r := 0; r < rounds; r++ {
		t0 := time.Now()
		if err := run(); err != nil {
			return 0, "", err
		}
		rs = append(rs, float64(n)*4/time.Since(t0).Seconds()/1e9)
	}
	sort.Float64s(rs)
	return rs[len(rs)/2], d.Name(), nil
}
