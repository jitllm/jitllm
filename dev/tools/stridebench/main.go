// stridebench prices the access pattern the packed kernel actually makes.
//
// It measures the pattern, not the kernel: the device layout is
// [sub-block][word][row], so a tile of R rows reads R*4 contiguous bytes and
// then jumps nrows*4 to the next k-position.
package main

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	var sink atomic.Uint64
	const nrows = 2048
	const total = 512 << 20 // 512 MiB, far past L3
	buf := make([]byte, total)
	for i := range buf {
		buf[i] = byte(i)
	}
	stride := nrows * 4

	run := func(name string, chunk int) {
		// The workers must fit inside one step: with chunk*workers > stride
		// they read each other's bytes and past the end.
		workers := 6
		if chunk*workers > stride {
			workers = stride / chunk
		}
		if workers < 1 {
			return
		}
		var wg sync.WaitGroup
		t := time.Now()
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				var s uint64
				// Each worker owns a disjoint slice of the chunks at each
				// stride step, exactly as the pool hands out row tiles.
				for base := 0; base+stride <= total; base += stride {
					off := base + w*chunk
					if off+chunk > total {
						break
					}
					for j := off; j < off+chunk; j += 64 {
						s += uint64(buf[j])
					}
				}
				sink.Add(s)
			}(w)
		}
		wg.Wait()
		d := time.Since(t)
		read := float64(total/stride) * float64(chunk*workers)
		fmt.Printf("  %-28s %7.1f ms   %6.2f GiB/s   (%d B per %d B step, %d workers)\n",
			name, float64(d.Microseconds())/1000, read/d.Seconds()/(1<<30), chunk, stride, workers)
	}

	fmt.Printf("stride %d B, 6 workers, %d MiB buffer\n", stride, total>>20)
	for _, c := range []int{64, 256, 1024, 4096} {
		run(fmt.Sprintf("chunk %d B", c), c)
	}
	// The reference: read the same volume with no stride at all.
	t := time.Now()
	var wg sync.WaitGroup
	per := total / 6
	for w := 0; w < 6; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			var s uint64
			for j := w * per; j < (w+1)*per; j += 64 {
				s += uint64(buf[j])
			}
			sink.Add(s)
		}(w)
	}
	wg.Wait()
	d := time.Since(t)
	fmt.Printf("  %-28s %7.1f ms   %6.2f GiB/s   (fully sequential)\n",
		"no stride", float64(d.Microseconds())/1000, float64(total)/d.Seconds()/(1<<30))
	runtime.KeepAlive(buf)
}
