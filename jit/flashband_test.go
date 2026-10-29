//go:build jitllmfault

package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestFlashProductsBand measures a 512-row prompt chunk's attention ending at
// 512 and 2048 positions on every CUDA device, at the head geometries of four
// served models: the tier's FlashPrefill70 (V float32 as it ships, and
// binary16) and the composer's flash form with its five binary16 products
// and with one a side, each against the definition in float64 over what its
// pool holds; and the flash kernels' times on the 8B's geometry at 2048.
func TestFlashProductsBand(t *testing.T) {
	testlock.GPU(t)
	shapes := []struct {
		name           string
		heads, kvh, hd int
	}{
		{"Llama-3.2-1B", 32, 8, 64},
		{"Llama-3.1-8B", 32, 8, 128},
		{"Qwen3-1.7B", 16, 8, 128},
		{"Qwen3-30B-A3B", 32, 4, 128},
	}
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, sh := range shapes {
			for _, N := range []int{512, 2048} {
				for _, sharp := range []bool{false, true} {
					timed := sh.name == "Llama-3.1-8B" && N == 2048 && !sharp
					flashBand(t, d, sh.name, sh.heads, sh.kvh, sh.hd, N, sharp, timed)
					ran++
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no CUDA device")
	}
}

type bandErr struct{ nmse, rowWorst, maxAbs float64 }

func bandOf(got []float32, want []float64, R, w int) bandErr {
	var e bandErr
	var num, den float64
	for r := range R {
		var rn, rd float64
		for i := r * w; i < (r+1)*w; i++ {
			x := float64(got[i]) - want[i]
			rn += x * x
			rd += want[i] * want[i]
			e.maxAbs = max(e.maxAbs, math.Abs(x))
		}
		num, den = num+rn, den+rd
		e.rowWorst = max(e.rowWorst, rn/rd)
		if math.IsNaN(rn) {
			e.rowWorst = math.NaN()
		}
	}
	e.nmse = num / den
	return e
}

func flashBand(t *testing.T, d backend.Device, name string, heads, kvh, hd, N int, sharp, timed bool) {
	const page, R, rounds, reps = 64, 512, 11, 5
	p0 := N - R
	w, qw := kvh*hd, heads*hd
	rng := rand.New(rand.NewSource(int64(N + hd + heads)))
	rnd := func(n int, s float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * s)
		}
		return x
	}
	qs := 1.0
	if sharp {
		qs = 4
	}
	ks, vs := make([][]float32, N), make([][]float32, N)
	off := make([]float32, w)
	if sharp {
		for range 4 {
			off[rng.Intn(w)] = 6
		}
	}
	for p := range N {
		ks[p], vs[p] = rnd(w, 1), rnd(w, 1)
		for e := range w {
			ks[p][e] += off[e]
		}
	}
	q := rnd(R*qw, qs)
	scale := 1 / math.Sqrt(float64(hd))

	// The definition: exact V, and V as a binary16 pool holds it.
	half := func(x float32) float64 { return float64(quant.DecodeHalf(quant.EncodeHalf(x))) }
	wantX, wantH := make([]float64, R*qw), make([]float64, R*qw)
	var wg sync.WaitGroup
	jobs := make(chan int)
	for range runtime.NumCPU() {
		wg.Go(func() {
			sc := make([]float64, N)
			for r := range jobs {
				end := p0 + r + 1
				for h := range heads {
					kv := h / (heads / kvh)
					s := sc[:end]
					for p := range end {
						var a float64
						for e := range hd {
							a += float64(q[r*qw+h*hd+e]) * float64(ks[p][kv*hd+e])
						}
						s[p] = a * float32Scale(scale)
					}
					oracle.Softmax(s)
					for e := range hd {
						var ox, oh float64
						for p := range end {
							v := vs[p][kv*hd+e]
							ox += s[p] * float64(v)
							oh += s[p] * half(v)
						}
						wantX[r*qw+h*hd+e], wantH[r*qw+h*hd+e] = ox, oh
					}
				}
			}
		})
	}
	for r := range R {
		jobs <- r
	}
	close(jobs)
	wg.Wait()

	tab := make([]uint32, (N+page-1)/page)
	for i := range tab {
		tab[i] = uint32(i)
	}
	tag := fmt.Sprintf("%s %dq%dkv%dd N=%d sharp=%v", name, heads, kvh, hd, N, sharp)

	// The tier's FlashPrefill70 over a pool holding every position.
	for _, f16 := range []bool{false, true} {
		fs := kernels.FlashPrefill70Shape{Heads: heads, KVHeads: kvh, Dim: hd, Rows: R, Scale: float32(scale), Page: page, F16: f16}
		k, err := kernels.FlashPrefill70(fs)
		if err != nil {
			t.Fatal(err)
		}
		kp := make([]float32, len(tab)*page*w)
		var vp []uint32
		if f16 {
			vp = make([]uint32, len(tab)*page*w/2)
		} else {
			vp = make([]uint32, len(tab)*page*w)
		}
		for p := range N {
			for e := range w {
				kp[(p/page)*page*w+e*page+p%page] = ks[p][e]
				i := p*w + e
				if f16 {
					hv := uint32(quant.EncodeHalf(vs[p][e]))
					vp[i/2] |= hv << (16 * (i % 2))
				} else {
					vp[i] = math.Float32bits(vs[p][e])
				}
			}
		}
		desc := make([]uint32, R*kernels.PRowWords)
		for r := range R {
			copy(desc[r*kernels.PRowWords:], []uint32{0, 0, uint32(p0 + r + 1), uint32(p0 + r)})
		}
		kc, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		var bufs []backend.Buf
		up := func(x []uint32) backend.Buf {
			b, err := d.Alloc(max(4*len(x), 4))
			if err != nil {
				t.Fatal(err)
			}
			if err := b.Write(words(x)); err != nil {
				t.Fatal(err)
			}
			bufs = append(bufs, b)
			return b
		}
		out := up(poisoned(R * qw))
		args := []backend.Buf{up(f32w(q)), up(f32w(kp)), up(vp), up([]uint32{^uint32(0), ^uint32(0)}), out, up(tab), up(desc)}
		launch := func() {
			if err := kc.Launch(kernels.FlashPrefill70Groups(fs), kernels.FlashPrefill70Threads, args...); err != nil {
				t.Fatal(err)
			}
		}
		launch()
		got := readWords(t, out, R*qw)
		want := wantX
		if f16 {
			want = wantH
		}
		e := bandOf(got, want, R, qw)
		msg := ""
		if timed {
			msg = fmt.Sprintf(", %.1f us", timeUs(func() { out.Read(make([]byte, 4)) }, launch, rounds, reps))
		}
		t.Logf("BAND %s main f16V=%v: NMSE %.3e, worst row %.3e, max abs %.3e%s", tag, f16, e.nmse, e.rowWorst, e.maxAbs, msg)
		kc.Close()
		for _, b := range bufs {
			b.Free()
		}
	}

	// The composer's flash form, five products and one a side.
	for _, one := range []bool{false, true} {
		reg := jit.Default()
		if one {
			reg.FlashOneProduct()
		}
		b := graph.New()
		n := b.KVAppend(b.In("k", w), b.In("v", w), 0)
		b.Out(b.Attend(b.In("q", qw), n, graph.Attrs{NHead: heads, NKVHead: kvh, HeadDim: hd, Scale: float64(float32(scale))}))
		g := b.G
		dev := gpuDevice(d)
		dev.Matrix = [3]int{16, 16, 16}
		_, codes, err := reg.Form(jit.Request{Graph: g, Step: jit.Step{Rows: R, Page: page, Keys: len(tab) * page, Seqs: 1},
			Device: dev, History: jit.History{Keys: jit.Pool{ByDim: true}, Values: jit.Pool{Half: true}}})
		if err != nil {
			t.Fatal(err)
		}
		flash := -1
		for i, cd := range codes {
			if slices.Contains(cd.Passes, "flash") {
				flash = i
			}
		}
		if flash < 0 {
			t.Fatalf("%s: the form did not take the flash pass", tag)
		}
		kpool := poisoned(len(tab) * page * w)
		vpool := poisoned(len(tab) * page * w / 2)
		for p := range p0 {
			for e := range w {
				kpool[(p/page*w+e)*page+p%page] = math.Float32bits(ks[p][e])
				hv, i := uint32(quant.EncodeHalf(vs[p][e])), p*w+e
				if e%2 == 0 {
					vpool[i/2] = vpool[i/2]&0xFFFF0000 | hv
				} else {
					vpool[i/2] = vpool[i/2]&0xFFFF | hv<<16
				}
			}
		}
		kc, vc := make([]float32, R*w), make([]float32, R*w)
		for r := range R {
			copy(kc[r*w:], ks[p0+r])
			copy(vc[r*w:], vs[p0+r])
		}
		rows := make([]uint32, R*4)
		for r := range R {
			p := uint32(p0 + r)
			copy(rows[r*4:], []uint32{0, 0, p + 1, p})
		}
		op, err := compileLone(d, codes)
		if err != nil {
			t.Fatal(err)
		}
		held := map[string]backend.Buf{}
		get := func(key string, x []uint32) backend.Buf {
			if b, ok := held[key]; ok {
				return b
			}
			b, err := d.Alloc(max(4*len(x), 4))
			if err != nil {
				t.Fatal(err)
			}
			if err := b.Write(words(x)); err != nil {
				t.Fatal(err)
			}
			held[key] = b
			return b
		}
		vals := map[graph.ValueID][]uint32{g.In[0]: f32w(kc), g.In[1]: f32w(vc), g.In[2]: f32w(q), g.Out[0]: poisoned(R * qw)}
		tables := map[string][]uint32{"rows": rows, "rows.n": {R}, "pages": tab, "kv.k": kpool, "kv.v": vpool}
		args := make([][]backend.Buf, len(codes))
		for ci, cd := range codes {
			for i, bd := range cd.Binds {
				o := bd.Operand
				switch {
				case o.Kind == jit.OperandValue && vals[o.Value] != nil:
					args[ci] = append(args[ci], get(fmt.Sprintf("v%d", o.Value), vals[o.Value]))
				case o.Kind == jit.OperandTable && tables[o.Name] != nil:
					args[ci] = append(args[ci], get("t"+o.Name, tables[o.Name]))
				case bd.Words != nil:
					args[ci] = append(args[ci], get(fmt.Sprintf("c%d.%d", ci, i), bd.Words))
				case o.Kind == jit.OperandScratch || o.Kind == jit.OperandValue:
					args[ci] = append(args[ci], get(fmt.Sprintf("s%d.%s.%d", o.Kind, o.Name, o.Value), poisoned(int(max(bd.Bytes, 4)/4))))
				default:
					t.Fatalf("%s binds %+v, which the probe does not hold", cd.Name, o)
				}
			}
		}
		one1 := func(ci int) {
			if err := op.kerns[ci].Launch(codes[ci].LiveGroups(R), codes[ci].GPU.Group[0], args[ci]...); err != nil {
				t.Fatal(err)
			}
		}
		for ci := range codes {
			one1(ci)
		}
		out := held[fmt.Sprintf("v%d", g.Out[0])]
		got := readWords(t, out, R*qw)
		e := bandOf(got, wantH, R, qw)
		ex := bandOf(got, wantX, R, qw)
		msg := ""
		if timed {
			res := ""
			if r, ok := backend.KernelResourcesOf(op.kerns[flash]); ok {
				res = fmt.Sprintf(" (regs %d, spill %d, shared %d)", r.Regs, r.SpillBytes, r.SharedBytes)
			}
			msg = fmt.Sprintf(", flash kernel %.1f us%s over %d codes", timeUs(func() { out.Read(make([]byte, 4)) }, func() { one1(flash) }, rounds, reps), res, len(codes))
		}
		t.Logf("BAND %s composed products=%d: NMSE %.3e, worst row %.3e, max abs %.3e; against exact V NMSE %.3e%s",
			tag, map[bool]int{false: 5, true: 2}[one], e.nmse, e.rowWorst, e.maxAbs, ex.nmse, msg)
		op.close()
		for _, b := range held {
			b.Free()
		}
	}
}

func float32Scale(s float64) float64 { return float64(float32(s)) }

func words(x []uint32) []byte {
	p := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(p[4*i:], v)
	}
	return p
}

func readWords(t *testing.T, b backend.Buf, n int) []float32 {
	p := make([]byte, 4*n)
	if err := b.Read(p); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[4*i:]))
	}
	return out
}

func timeUs(sync, f func(), rounds, reps int) float64 {
	var us []float64
	for range rounds {
		sync()
		t0 := time.Now()
		for range reps {
			f()
		}
		sync()
		us = append(us, float64(time.Since(t0).Microseconds())/float64(reps))
	}
	sort.Float64s(us)
	return us[len(us)/2]
}
