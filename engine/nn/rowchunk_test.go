//go:build amd64 || arm64

package nn

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
)

// TestRowChunkDuelSettlesAndReachesTheMatmul drives the prefill tuners on
// synthetic chunk rates that depend only on the row chunk cap the batched
// matmul would read (f.chunkBytes), fastest at 64 KiB. The
// participant and token-chunk duels see ties and must settle first; the row
// chunk duel must then find 64 KiB and leave it in place for every later
// prefill chunk. Because the rate is computed from the field the matmul reads,
// a duel that picked a cap without applying it could not tell the rungs apart
// and would settle on the first -- which is the violation this fails on.
func TestRowChunkDuelSettlesAndReachesTheMatmul(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	j := NewJIT(256, 256, []quant.Type{quant.Q4_K}, WithTune(TuneForce), WithQuietTuner(true))
	if j == nil {
		t.Fatal("no JIT")
	}
	defer j.Close()
	rate := func() float64 {
		switch j.chunkBytes >> 10 {
		case 64:
			return 110
		case 256:
			return 105
		}
		return 100
	}
	for i := 0; i < 2000 && (j.ptune == nil || j.ptune.on || j.ctune.on || j.rtune.on); i++ {
		w := j.PrefillChunk()
		if (j.ptune.on || j.ctune.on || j.wtune.on) && j.chunkBytes>>10 != j.rtune.best {
			t.Fatalf("the row chunk moved (%d KiB) while an earlier duel was running", j.chunkBytes>>10)
		}
		j.ObserveChunk(w, w, time.Duration(float64(w)/rate()*float64(time.Second)))
	}
	if j.rtune.on {
		t.Fatal("the row chunk duel never settled")
	}
	if j.rtune.best != 64 {
		t.Fatalf("the row chunk duel settled at %d KiB, want 64 (the rung that wins)", j.rtune.best)
	}
	j.PrefillChunk()
	if j.chunkBytes != 64<<10 {
		t.Fatalf("after settling the matmul reads a cap of %d bytes, want %d", j.chunkBytes, 64<<10)
	}
}

// TestGEMMRowsDuelSettlesAndReachesTheGEMM is the row chunk gate's twin for
// the weight-stationary GEMM: synthetic rates that depend only on the rows the
// GEMM would run (f.gemmRows), fastest with the GEMM off.
// The duel must find it, and a MatMulPacked after it must then take the tiled
// path -- asserted on the GEMM's own call counter, so a duel that settled
// without the field reaching the matmul fails here even if it picked right.
func TestGEMMRowsDuelSettlesAndReachesTheGEMM(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	const nrows, k, ntok = 256, 256, 16
	j := NewJIT(k, nrows, []quant.Type{quant.Q4_K}, WithTune(TuneForce), WithQuietTuner(true))
	if j == nil {
		t.Fatal("no JIT")
	}
	defer j.Close()
	pk := packOne(t, quant.Q4_K, nrows, k)
	x := make([]float32, ntok*k)
	for i := range x {
		x[i] = float32(i%7) - 3
	}
	out := make([]float32, ntok*nrows)
	if !j.MatMulPacked(out, quant.Q4_K, pk, x, nrows, k, ntok) {
		t.Fatal("MatMulPacked declined")
	}
	if j.PackedGEMMCalls() != 1 {
		t.Skip("no weight-stationary GEMM on this host, so there is nothing for the duel to switch")
	}
	rate := func() float64 {
		switch j.gemmRows {
		case gemmOff:
			return 110
		case 64:
			return 105
		}
		return 100
	}
	for i := 0; i < 2000 && (j.ptune == nil || j.ptune.on || j.ctune.on || j.wtune.on); i++ {
		w := j.PrefillChunk()
		if (j.ptune.on || j.ctune.on) && j.gemmRows != j.wtune.best {
			t.Fatalf("the GEMM rows moved (%d) while an earlier duel was running", j.gemmRows)
		}
		j.ObserveChunk(w, w, time.Duration(float64(w)/rate()*float64(time.Second)))
	}
	if j.wtune.on {
		t.Fatal("the GEMM rows duel never settled")
	}
	if j.wtune.best != gemmOff {
		t.Fatalf("the GEMM rows duel settled at %d, want %d (the GEMM off)", j.wtune.best, gemmOff)
	}
	j.PrefillChunk()
	if !j.MatMulPacked(out, quant.Q4_K, pk, x, nrows, k, ntok) {
		t.Fatal("MatMulPacked declined")
	}
	if n := j.PackedGEMMCalls(); n != 1 {
		t.Fatalf("after settling on the tiled path the GEMM ran again (%d calls, want 1)", n)
	}
}

// TestPrefillDuelsContinueAcrossJITs: every model.Instance builds its own JIT, so
// a second JIT over the same shapes must pick up the same running duel where
// the first left it. Violation (a fresh duel per JIT): the second starts at
// turn 0.
func TestPrefillDuelsContinueAcrossJITs(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	mk := func() *JIT {
		j := NewJIT(256, 256, []quant.Type{quant.Q4_K}, WithTune(TuneForce), WithQuietTuner(true))
		if j == nil {
			t.Fatal("no JIT")
		}
		return j
	}
	// The participant duel, unless the pool is too small to have one (two
	// workers leave a single candidate); then the chunk duel, which runs in
	// its place.
	duel := func(j *JIT) *duelTuner {
		if len(j.ptune.cands) >= 2 {
			return j.ptune
		}
		return j.ctune
	}
	a := mk()
	defer a.Close()
	for i := 0; i < 3; i++ {
		w := a.PrefillChunk()
		a.ObserveChunk(w, w, time.Duration(w)*time.Millisecond)
	}
	if d := duel(a); !d.on || d.turn == 0 {
		t.Fatalf("the first JIT's %s duel is not mid-run (on %v, turn %d, pool %d)", d.label, d.on, d.turn, a.pool.Max())
	}
	b := mk()
	defer b.Close()
	b.PrefillChunk()
	if duel(b) != duel(a) || duel(b).turn != duel(a).turn {
		t.Fatalf("the second JIT started its own duel (turn %d) where the first had reached turn %d",
			duel(b).turn, duel(a).turn)
	}
}
