// readbench prices the jlm reader: parse, then page-in by four strategies.
//
// It measures the read, not the engine: a page-in is one file read of exactly
// PageSize bytes at DataOff+i*PageSize, so its rate depends only on the reader
// and the device.
package main

import (
	"flag"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
)

func main() {
	drop := flag.Bool("drop", true, "drop the page cache between strategies (needs root, best effort)")
	npage := flag.Int("pages", 24, "pages to read per strategy")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: readbench [-pages N] <model.jlm>")
		os.Exit(2)
	}
	path := flag.Arg(0)

	t0 := time.Now()
	c, err := jlm.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	parse := time.Since(t0)
	defer c.Close()
	fr, err := jlm.NewFrames(c)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer fr.Close()
	st, _ := os.Stat(path)
	fmt.Printf("%s\n  %.2f GiB, page %.1f MiB, %d blocks, %d tensors\n  parse %v\n",
		path, float64(st.Size())/(1<<30), float64(c.H.PageSize)/(1<<20),
		c.H.NBlocks, c.H.NTensors, parse)

	const strategies = 5
	n := *npage
	if n*strategies > int(c.H.NBlocks) {
		n = int(c.H.NBlocks) / strategies
	}
	if n == 0 {
		fmt.Fprintf(os.Stderr, "readbench: %d blocks is fewer than the %d strategies need\n",
			c.H.NBlocks, strategies)
		os.Exit(1)
	}
	fmt.Printf("  %d pages per strategy, disjoint ranges\n", n)
	bytes := float64(n) * float64(c.H.PageSize)

	dropCache := func() {
		if !*drop {
			return
		}
		// Best effort: without it every strategy after the first measures the
		// page cache instead of the device, which is the single easiest way to
		// report a storage rate that does not exist.
		f, err := os.OpenFile("/proc/sys/vm/drop_caches", os.O_WRONLY, 0)
		if err != nil {
			return
		}
		f.WriteString("1\n")
		f.Close()
	}

	// Every strategy reads a different set of pages, because dropping the page
	// cache needs root; a second strategy over the same pages would measure
	// RAM. The file must hold strategies*pages blocks, checked below.
	slot := 0
	run := func(name string, fn func(i int)) {
		dropCache()
		base := slot * n
		slot++
		t := time.Now()
		for i := base; i < base+n; i++ {
			fn(i)
		}
		d := time.Since(t)
		fmt.Printf("  %-22s %7.3f s   %6.2f GiB/s   %6.1f ms/page\n",
			name, d.Seconds(), bytes/d.Seconds()/(1<<30), float64(d.Milliseconds())/float64(n))
	}

	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	buf := make([]byte, c.H.PageSize)

	// 1. one ReadAt per page: the simplest thing that can work, and the queue
	//    depth is one, which is what a page fault also gives.
	run("ReadAt serial", func(i int) {
		off := int64(c.H.DataOff + uint64(i)*c.H.PageSize)
		if _, err := f.ReadAt(buf, off); err != nil {
			fmt.Fprintln(os.Stderr, "read:", err)
			os.Exit(1)
		}
	})

	// 2..4. the same page split across G goroutines. An NVMe needs several
	//       outstanding requests to reach its rate; one reader cannot produce
	//       them however large the request is.
	for _, g := range []int{4, 8, 16} {
		g := g
		run(fmt.Sprintf("ReadAt parallel x%d", g), func(i int) {
			base := int64(c.H.DataOff + uint64(i)*c.H.PageSize)
			chunk := (int(c.H.PageSize) + g - 1) / g
			var wg sync.WaitGroup
			for w := 0; w < g; w++ {
				lo := w * chunk
				hi := min(lo+chunk, int(c.H.PageSize))
				if lo >= hi {
					continue
				}
				wg.Add(1)
				go func() {
					defer wg.Done()
					if _, err := f.ReadAt(buf[lo:hi], base+int64(lo)); err != nil {
						fmt.Fprintln(os.Stderr, "read:", err)
						os.Exit(1)
					}
				}()
			}
			wg.Wait()
		})
	}

	// 5. the mapping, faulted by touching it. sink is read after the loop so
	// the compiler cannot delete the touch loop.
	var sink byte
	run("mmap fault", func(i int) {
		p := fr.Page(i)
		var s byte
		for j := 0; j < len(p); j += 4096 {
			s ^= p[j]
		}
		sink += s
	})
	if sink == 255 {
		fmt.Print("")
	}
}
