//go:build jitllmbench

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a t.Skip that would put a green line in every run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// Any JITLLM_* variable the test reads still selects its parameters.

package nn

import (
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestPerByteCost prices every packed format at equal bytes on the host
// kernels (the CPU analogue of mvbench -eqbytes), separating the unpack cost
// from the traffic. Equal bytes, not equal shape: at one shape a 4-bit payload
// is half an 8-bit one and sits at a different cache level. Row counts are
// chosen per format so every arm streams the same bytes.
func TestPerByteCost(t *testing.T) {
	k := 2048
	if v := os.Getenv("JITLLM_PERBYTE_K"); v != "" {
		if _, err := fmtSscan(v, &k); err != nil || k <= 0 || k%256 != 0 {
			t.Fatalf("JITLLM_PERBYTE_K=%q: want a positive multiple of 256", v)
		}
	}
	target := 96 << 20 // bytes each format must stream, well past any L2/SLC
	if v := os.Getenv("JITLLM_PERBYTE_MIB"); v != "" {
		var m int
		if _, err := fmtSscan(v, &m); err == nil && m > 0 {
			target = m << 20
		}
	}

	type row struct {
		t     quant.Type
		rows  int
		bytes int64
		gbs   float64
	}
	var out []row
	for _, qt := range quant.PackedTypes {
		if v := os.Getenv("JITLLM_PERBYTE_TYPES"); v != "" && !strings.Contains(","+v+",", ","+qt.String()+",") {
			continue
		}
		if !cpu.PackedSupported(qt) {
			t.Logf("%-5s no packed kernel on this architecture", qt)
			continue
		}
		q, ok := kernels.QuantOf(qt)
		if !ok {
			continue
		}
		// The row count is derived from the packer (PackedWords carries the d
		// and SC planes and the row-pairing), not from bits per weight.
		const probe = 64
		nq, nd, nsc, err := kernels.PackedWords(q, probe, k)
		if err != nil {
			t.Fatalf("%s: %v", qt, err)
		}
		perRow := float64(nq+nd+nsc) * 4 / probe
		rows := int(float64(target)/perRow) / 64 * 64
		if rows < 64 {
			rows = 64
		}
		// JITLLM_PERBYTE_ROWS pins a model's shape instead of equal bytes.
		if v := os.Getenv("JITLLM_PERBYTE_ROWS"); v != "" {
			if _, err := fmtSscan(v, &rows); err != nil || rows%64 != 0 {
				t.Fatalf("JITLLM_PERBYTE_ROWS=%q: want a multiple of 64", v)
			}
		}

		src := make([]byte, rows*k/int(qt.BlockElems())*int(qt.BlockBytes()))
		rng := rand.New(rand.NewSource(int64(qt)))
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		// A uniformly random f16 is Inf or NaN about one time in 32, and a
		// non-finite scale makes the timing meaningless as well as the answer:
		// the same trap TestPackedMatVecTailAgrees records.
		if !quant.PlantScales(qt, src, 0) {
			t.Fatalf("%s: no scale planter", qt)
		}
		qs, d, sc, err := kernels.PackWeights(q, src, rows, k)
		if err != nil {
			t.Fatalf("%s: %v", qt, err)
		}
		p := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
		if os.Getenv("JITLLM_PERBYTE_HUGE") == "1" {
			p = &Packed{QS: hugeCopy(p.QS), D: hugeCopy(p.D), SC: hugeCopy(p.SC)}
		}
		bytes := int64(len(p.QS) + len(p.D) + len(p.SC))

		var opts []Option
		if v := os.Getenv("JITLLM_PERBYTE_FPF"); v != "" {
			var d int
			if _, err := fmtSscan(v, &d); err != nil {
				t.Fatalf("JITLLM_PERBYTE_FPF=%q", v)
			}
			opts = append(opts, WithFusedPrefetch(d))
		}
		if v := os.Getenv("JITLLM_PERBYTE_KSPLIT"); v == "old" {
			opts = append(opts, WithFusedKSplit(-1))
		}
		j := NewJIT(k, rows, []quant.Type{qt}, opts...)
		if j == nil {
			t.Fatalf("%s: no JIT", qt)
		}
		kernelKnobs(t, j, qt)
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, rows)
		if !j.MatVecPacked(dst, qt, p, x, rows, k) {
			j.Close()
			t.Fatalf("%s: MatVecPacked declined at %d rows", qt, rows)
		}
		for i := 0; i < 3; i++ { // warm: codegen, page faults, the tuner
			j.NewInput()
			j.MatVecPacked(dst, qt, p, x, rows, k)
		}
		var rs []float64
		for r := 0; r < 9; r++ {
			j.NewInput()
			t0 := time.Now()
			j.MatVecPacked(dst, qt, p, x, rows, k)
			rs = append(rs, float64(bytes)/time.Since(t0).Seconds()/1e9)
		}
		j.Close()
		sort.Float64s(rs)
		out = append(out, row{qt, rows, bytes, rs[len(rs)/2]})
	}
	if len(out) == 0 {
		t.Fatal("no format ran: this measurement produced nothing")
	}

	// The cheapest format is the denominator, not a memory-wall probe: a
	// bandwidth probe measures its own access pattern, where a real kernel on
	// the real layout does not.
	best := 0.0
	for _, r := range out {
		if r.gbs > best {
			best = r.gbs
		}
	}
	t.Logf("equal bytes (~%d MiB each), k=%d, host packed matvec:", target>>20, k)
	for _, r := range out {
		t.Logf("  %-5s %6d rows  %7.1f MiB  %6.2f GB/s   %5.1f%% of the fastest format",
			r.t, r.rows, float64(r.bytes)/(1<<20), r.gbs, 100*r.gbs/best)
	}
}

func fmtSscan(s string, v *int) (int, error) {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, errNotANumber
		}
		n = n*10 + int(c-'0')
	}
	*v = n
	return 1, nil
}

var errNotANumber = errString("not a number")

type errString string

func (e errString) Error() string { return string(e) }

// TestShapeCost prices real decode shapes on the host kernels, streaming each
// through enough distinct copies (~192 MiB) that no call is served from a
// cache -- a decode token reads each matrix once, from DRAM. It exists to name
// which shape of a model runs under its format's per-byte rate: TestPerByteCost
// holds k at 2048, and a model is not one k.
//
// JITLLM_SHAPES="type:rows:k,..." overrides the list (types Q4_K, Q6_K, ...).
func TestShapeCost(t *testing.T) {
	type shape struct {
		t       quant.Type
		rows, k int
	}
	shapes := []shape{
		{quant.Q4_K, 2048, 2048}, {quant.Q4_K, 8192, 2048}, {quant.Q4_K, 2048, 8192}, {quant.Q6_K, 2048, 8192},
		{quant.Q4_K, 1536, 1536}, {quant.Q4_K, 256, 1536}, {quant.Q6_K, 256, 1536}, {quant.Q4_K, 8960, 1536},
		{quant.Q4_K, 1536, 8960}, {quant.Q6_K, 1536, 8960}, {quant.Q6_K, 151936, 1536}, {quant.Q6_K, 128256, 2048},
	}
	if v := os.Getenv("JITLLM_SHAPES"); v != "" {
		shapes = nil
		for _, f := range strings.Split(v, ",") {
			var name string
			var s shape
			parts := strings.Split(f, ":")
			if len(parts) != 3 {
				t.Fatalf("JITLLM_SHAPES entry %q: want type:rows:k", f)
			}
			name = parts[0]
			for _, qt := range quant.PackedTypes {
				if qt.String() == name {
					s.t = qt
				}
			}
			fmtSscan(parts[1], &s.rows)
			fmtSscan(parts[2], &s.k)
			shapes = append(shapes, s)
		}
	}
	for _, sh := range shapes {
		q, ok := kernels.QuantOf(sh.t)
		if !ok || !cpu.PackedSupported(sh.t) {
			t.Logf("%v: no packed kernel", sh.t)
			continue
		}
		rng := rand.New(rand.NewSource(int64(sh.rows)))
		// JITLLM_PADROWS packs the tensor that many rows wide and runs only the
		// first sh.rows of them, so the kernel walks a different row stride
		// over the same work -- the arm that separates the stride from the rows.
		packed := sh.rows
		if n, err := strconv.Atoi(os.Getenv("JITLLM_PADROWS")); err == nil && n > packed {
			packed = n
		}
		src := make([]byte, packed*sh.k/int(sh.t.BlockElems())*int(sh.t.BlockBytes()))
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		if !quant.PlantScales(sh.t, src, 0) {
			t.Fatalf("%s: no scale planter", sh.t)
		}
		qs, d, sc, err := kernels.PackWeights(q, src, packed, sh.k)
		if err != nil {
			t.Fatal(err)
		}
		one := int64(4 * (len(qs) + len(d) + len(sc)))
		n := int(max(1, (192<<20)/one))
		ps := make([]*Packed, n)
		for i := range ps {
			// Distinct copies: a view of one array would be served from cache.
			ps[i] = &Packed{QS: u32b(append([]uint32(nil), qs...)), D: u32b(append([]uint32(nil), d...)),
				SC: u32b(append([]uint32(nil), sc...)), Stride: packed}
		}
		var jo []Option
		if n, err := strconv.Atoi(os.Getenv("JITLLM_A64_PF")); err == nil {
			jo = append(jo, WithA64Prefetch(n))
		}
		if n, err := strconv.Atoi(os.Getenv("JITLLM_PART")); err == nil {
			jo = append(jo, WithParticipants(n))
		}
		if n, err := strconv.Atoi(os.Getenv("JITLLM_CORES")); err == nil {
			jo = append(jo, WithSched(sched.WithCores(n)))
		}
		if n, err := strconv.Atoi(os.Getenv("JITLLM_PERBYTE_KSLICES")); err == nil {
			jo = append(jo, WithFusedKSlices(n))
		}
		j := NewJIT(sh.k, sh.rows, []quant.Type{sh.t}, jo...)
		kernelKnobs(t, j, sh.t)
		x := make([]float32, sh.k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, sh.rows)
		for i := 0; i < 2*n; i++ {
			j.NewInput()
			j.MatVecPacked(dst, sh.t, ps[i%n], x, sh.rows, sh.k)
		}
		var rs []float64
		for r := 0; r < 5; r++ {
			t0 := time.Now()
			for i := 0; i < n; i++ {
				j.NewInput()
				j.MatVecPacked(dst, sh.t, ps[i], x, sh.rows, sh.k)
			}
			rs = append(rs, float64(one)*float64(n)/time.Since(t0).Seconds()/1e9)
		}
		j.Close()
		sort.Float64s(rs)
		t.Logf("%-5v %6d x %-5d  %7.2f MiB x %3d copies  %6.2f GB/s  %7.1f us/call",
			sh.t, sh.rows, sh.k, float64(one)/(1<<20), n, rs[2], float64(one)/rs[2]/1e3)
	}
}

// kernelKnobs applies the kernel-selection knobs both benches share:
// JITLLM_PERBYTE_TILED=1 takes the fused kernel away, so MatVecPacked runs the
// tiled one (fusedAt still owns and closes it), and JITLLM_PERBYTE_AHEAD
// installs the fused kernel's prefetch form directly -- WithFusedPrefetch only
// pins the per-token duel, and these loops run no tokens.
func kernelKnobs(t *testing.T, j *JIT, qt quant.Type) {
	t.Helper()
	if os.Getenv("JITLLM_PERBYTE_TILED") == "1" {
		delete(j.packedFused, qt)
		return
	}
	v := os.Getenv("JITLLM_PERBYTE_AHEAD")
	if v == "" {
		return
	}
	var d int
	if _, err := fmtSscan(v, &d); err != nil {
		t.Fatalf("JITLLM_PERBYTE_AHEAD=%q", v)
	}
	b, err := j.em.PackedFusedAhead(qt, d)
	if err != nil {
		t.Fatalf("%s: prefetch form %d: %v", qt, d, err)
	}
	c, err := cpu.MapNamed(b, qt.String()+"_ahead")
	if err != nil {
		t.Fatal(err)
	}
	// j.Close closes it with the rest of packedFused.
	j.packedFused[qt] = c
}
