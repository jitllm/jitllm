package run_test

import (
	"bytes"
	"flag"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"github.com/jitllm/jitllm/jit"
)

var repeatModel = flag.String("repeat-model", "stories15M-q8_0", "the model TestEveryLaunchRepeatsItself steps")

// TestEveryLaunchRepeatsItself steps one token at one position again and
// again on every device, every launch's outputs held to the same step's
// before it: a kernel that races, or reads memory nothing wrote, differs
// from one step to the next where a comparison against the host may land
// inside its tolerance. Each step restarts the sequence, its recurrent
// state zero; the first step is not compared, since a buffer a launch
// writes in part still holds what the step before it left.
func TestEveryLaunchRepeatsItself(t *testing.T) {
	fresh.Must(t, testmodels.Path(*repeatModel+".jlm"))
	m, err := model.Open(testmodels.Path(*repeatModel + ".jlm"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	src := m.Composing()
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	for _, dev := range devs {
		t.Run(dev.API(), func(t *testing.T) {
			r, err := run.New(src, dev, gpu, 32)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			var codes []*jit.Code
			for pi := range r.Plan().Parts {
				for ci := range r.Plan().Parts[pi].Codes {
					codes = append(codes, &r.Plan().Parts[pi].Codes[ci])
				}
			}
			s, err := r.NewSession()
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			// Each step restarts the sequence: a recurrent state takes every
			// step, where a history's position is only rewritten. The other
			// buffers carry over, so a kernel reading what nothing wrote this
			// step reads the step before's.
			step := func() [][][]byte {
				if err := s.Restart(); err != nil {
					t.Fatal(err)
				}
				out := make([][][]byte, len(codes))
				if err := s.StepEach(1, 0, func(i int) {
					for _, b := range codes[i].Binds {
						if b.Writes {
							out[i] = append(out[i], append([]byte(nil), s.Storage(b.Storage)...))
						}
					}
				}); err != nil {
					t.Fatal(err)
				}
				return out
			}
			step()
			want := step()
			for n := range 16 {
				got := step()
				for i, cd := range codes {
					for j, b := range got[i] {
						if !bytes.Equal(b, want[i][j]) {
							t.Fatalf("step %d: launch %d (%s) wrote output %d differently", n, i, cd.Name, j)
						}
					}
				}
			}
		})
	}
}
