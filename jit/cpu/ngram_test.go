//go:build amd64 || arm64

package cpu

import (
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"
)

// ngramRef is the hash as transformers' Qwen4ExpTextNGramEmbedding computes
// it, written out plainly: the window's ids (none or past a cut read as eos),
// then per head the mixed product mod its vocabulary plus its offset.
func ngramRef(win []uint32, eos uint64, mul, off, voc []uint64, per int, reset bool) (rows []uint64, next []uint32) {
	n := len(mul)
	ctx := make([]uint64, n)
	ctx[0] = uint64(win[0])
	cut := false
	for s := 1; s < n; s++ {
		v := eos
		if win[s] != 0 {
			v = uint64(win[s]) - 1
		}
		if cut {
			v = eos
		}
		if reset && v == eos {
			cut = true
		}
		ctx[s] = v
	}
	for g := 2; g <= n; g++ {
		m := ctx[0] * mul[0]
		for j := 1; j < g; j++ {
			m ^= ctx[j] * mul[j]
		}
		for k := 0; k < per; k++ {
			h := (g-2)*per + k
			rows = append(rows, m%voc[h]+off[h])
		}
	}
	next = append([]uint32{win[0] + 1}, win[1:n-1]...)
	return rows, next
}

// TestNGramHashMatchesReference holds the hash to ngramRef on every tier this
// host runs, at the release's shape (3-grams, eight heads each) and others,
// over windows with and without an end of sequence in every slot, multipliers
// past 2^44 and vocabularies near 2^25 -- and against the violation, which a
// window with an end of sequence must tell apart.
func TestNGramHashMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 13))
	const eos = 248044
	for _, em := range hcTiers() {
		tier := em.Tier
		for _, sh := range []struct{ n, per int }{{3, 8}, {2, 1}, {4, 3}, {3, 2}} {
			for _, reset := range []bool{true, false} {
				code, err := em.NGramHash(sh.n, sh.per, reset)
				if err != nil {
					t.Fatal(err)
				}
				c, err := Map(code)
				if err != nil {
					t.Fatal(err)
				}
				heads := ngramHeads(sh.n, sh.per)
				mul := make([]uint64, sh.n)
				off, voc := make([]uint64, heads), make([]uint64, heads)
				for j := range mul {
					mul[j] = 2*rng.Uint64N(1<<44) + 1
				}
				var at uint64
				for h := range voc {
					voc[h] = 20000003 + uint64(rng.IntN(1000))
					off[h] = at
					at += voc[h]
				}
				konst := NGramConsts(eos, mul, off, voc)
				ids := make([]uint64, sh.n)
				for trial := 0; trial < 400; trial++ {
					win := make([]uint32, sh.n)
					win[0] = uint32(rng.IntN(248320))
					for s := 1; s < sh.n; s++ {
						switch rng.IntN(4) {
						case 0:
							win[s] = 0 // none
						case 1:
							win[s] = eos + 1
						default:
							win[s] = uint32(rng.IntN(248320)) + 1
						}
					}
					if trial%5 == 0 {
						win[0] = eos
					}
					rows := make([]uint64, heads+1)
					next := make([]uint32, sh.n)
					rows[heads], next[sh.n-1] = 0xDEAD, 0xBEEF
					c.Call(&Args{
						Out:     (*float32)(unsafe.Pointer(&rows[0])),
						W:       (*byte)(unsafe.Pointer(&win[0])),
						Scr:     &konst[0],
						Scratch: (*byte)(unsafe.Pointer(&ids[0])),
						Out2:    (*float32)(unsafe.Pointer(&next[0])),
					})
					wantRows, wantNext := ngramRef(win, eos, mul, off, voc, sh.per, reset)
					if !slices.Equal(rows[:heads], wantRows) || !slices.Equal(next[:sh.n-1], wantNext) {
						t.Fatalf("%v %d-grams x%d reset %v, window %v: rows %v next %v, want %v %v",
							tier, sh.n, sh.per, reset, win, rows[:heads], next[:sh.n-1], wantRows, wantNext)
					}
					if rows[heads] != 0xDEAD || next[sh.n-1] != 0xBEEF {
						t.Fatalf("%v: the kernel wrote past its outputs", tier)
					}
				}
				c.Close()
			}
		}
	}
}
