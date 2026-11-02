package kernels

import (
	"bytes"
	"math/rand"
	"runtime"
	"testing"
)

// TestPackIsIndependentOfWorkerCount gates PackWeightsInto's claim that its
// output is byte-identical for any worker count. It matters because a
// container converted on a host with a different GOMAXPROCS would otherwise be
// a different file; with rows sharing d words, a split that cuts a word's
// rows drops scales.
func TestPackIsIndependentOfWorkerCount(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	for _, q := range []Quant{Q4_0, Q8_0, Q4_K, Q6_K} {
		qi := qtab[q]
		if qi.blockE == 0 {
			continue
		}
		// Row counts chosen so ceil(nrows/n) is odd for some n, the condition for
		// a pair to straddle a worker boundary.
		for _, nrows := range []int{258, 512, 1030, 2048} {
			k := qi.blockE * 4
			src := make([]byte, nrows*(k/qi.blockE)*qi.blockB)
			r := rand.New(rand.NewSource(20260917))
			for i := range src {
				src[i] = byte(r.Intn(256))
			}
			var want []byte
			for _, n := range []int{1, 2, 3, 5, 7, 8, 16} {
				runtime.GOMAXPROCS(n)
				qs, d, sc, err := PackedWords(q, nrows, k)
				if err != nil {
					t.Fatalf("%v rows=%d: %v", q, nrows, err)
				}
				qsb, db, scb := make([]uint32, qs), make([]uint32, d), make([]uint32, sc)
				if err := PackWeightsInto(q, src, nrows, k, qsb, db, scb); err != nil {
					t.Fatalf("%v rows=%d n=%d: %v", q, nrows, n, err)
				}
				got := append(u32b(qsb), append(u32b(db), u32b(scb)...)...)
				if want == nil {
					want = got
					continue
				}
				if !bytes.Equal(got, want) {
					t.Errorf("%v rows=%d: GOMAXPROCS=%d packs different bytes from "+
						"GOMAXPROCS=1 -- a container is then a property of the machine "+
						"that converted it", q, nrows, n)
					break
				}
			}
		}
	}
}

func u32b(w []uint32) []byte {
	b := make([]byte, len(w)*4)
	for i, v := range w {
		b[i*4], b[i*4+1], b[i*4+2], b[i*4+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	return b
}
