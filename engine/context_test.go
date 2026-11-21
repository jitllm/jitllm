package engine

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// fillMax bounds the models whose cache the test fills. Filling is setup for
// the errFull assertion, and running it on every model costs the sum of their
// contexts, which pushed the package past its timeout. errFull reads no
// property of the architecture, so a few small models cover it; the clamp
// assertions still run on every model. The count of filled models is checked,
// so the bound cannot silently disable that half.
const fillMax = 2048

// TestOverlongContextIsReportedNotFatal asks for far more context than the model
// has and checks that it does not panic, does not allocate past the model, and
// lets the caller see it was clamped (otherwise the failure arrives hundreds of
// tokens later as "kv cache is full"). It stays a clamp, not a refusal: a
// caller's prompt+n+1 is an upper bound, and the run usually still fits.
func TestOverlongContextIsReportedNotFatal(t *testing.T) {
	paths := modelFiles()
	var ran, filled int
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil || fi.Size() > 512<<20 {
			continue
		}
		dst, ok := sweepJLM(t, p)
		if !ok {
			continue
		}
		m, err := model.Open(dst)
		if err != nil {
			continue
		}
		ran++
		name := filepath.Base(p)
		want := m.Cfg.NCtx

		s := testInstance(t, m, want*10)
		asked, got, clamped := s.ContextClamped()
		if !clamped || asked != want*10 || got != want {
			t.Errorf("%s: asked %d got %d clamped %v, want asked %d got %d clamped true",
				name, asked, got, clamped, want*10, want)
		}
		if s.MaxSeq() != want {
			t.Errorf("%s: MaxSeq %d, want the model's context %d", name, s.MaxSeq(), want)
		}
		// It runs, and it ends with an error naming both numbers, not a panic
		// and not silently wrong tokens.
		if want <= fillMax {
			filled++
			var full errFull
			for i := 0; i < want+4; i++ {
				if _, err := s.Forward(1); err != nil {
					if !errors.As(err, &full) {
						t.Fatalf("%s: token %d: %v", name, i, err)
					}
					if full.max != want || full.asked != want*10 {
						t.Errorf("%s: errFull{max %d asked %d}, want {%d %d}",
							name, full.max, full.asked, want, want*10)
					}
					break
				}
			}
			if full.max == 0 {
				t.Errorf("%s: filled the cache without reporting it", name)
			}
		}
		s.Close()

		// An in-range request must not be reported as clamped, or the notice
		// becomes noise.
		s2 := testInstance(t, m, min(16, want))
		if _, _, c := s2.ContextClamped(); c {
			t.Errorf("%s: an in-range request reported as clamped", name)
		}
		s2.Close()
		m.Close()
	}
	if ran == 0 {
		t.Skip("no model under 512 MiB present")
	}
	if filled == 0 {
		t.Fatalf("%d models, and not one has a context of %d or under, so the cache was "+
			"never filled and the errFull half of this gate did not run", ran, fillMax)
	}
	t.Logf("%d models: over-long context clamped and reported; %d of them filled and "+
		"survived it", ran, filled)
}
