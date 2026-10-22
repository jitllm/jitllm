package cost

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// header is a model as a node another node routes to would send it: the
// container's configuration and tensor index, offsets dropped, no file
// behind it.
type header struct {
	cfg *jlm.Config
	by  map[[3]int32]*jlm.Entry
	all []jlm.Entry
}

func (h header) Config() *jlm.Config { return h.cfg }

func (h header) Entries() []jlm.Entry { return h.all }

func (h header) Find(role jlm.Role, block, index int32) (*jlm.Entry, bool) {
	e, ok := h.by[[3]int32{int32(role), block, index}]
	return e, ok
}

// TestHeaderPricesAsTheContainer holds a model's header, copied out of its
// container with every offset dropped and the file closed, to pricing the
// model as the container does.
func TestHeaderPricesAsTheContainer(t *testing.T) {
	f, err := jlm.Open(testmodels.Path("synth-ernie45.jlm"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := *f.Config()
	h := header{cfg: &cfg, by: map[[3]int32]*jlm.Entry{}}
	for _, e := range f.Entries() {
		e.QSOff, e.DOff, e.SCOff = 0, 0, 0
		h.by[[3]int32{int32(e.Role), e.Block, e.Index}] = &e
		h.all = append(h.all, e)
	}
	d := Device{Device: jit.Device{Kind: jit.DeviceGPU, Slots: 163840, Lanes: 32}, Units: 80, Bandwidth: 869e9, Launch: 2.9e-6, Floor: 7.5e-6}
	d.Caches[1] = Cache{Bytes: 6 << 20, Rate: Rate{Total: 3.1e12}, Latency: 1.46e-7}
	d.Rates[ClassFloat] = Rate{Thread: 0.7e9, Total: 7.8e12}
	s := jit.Step{Rows: 1, Page: 64, Keys: 9}
	on := func(jit.Part) Device { return d }
	want, err := Estimate(jit.Default(), f, s, on)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	got, err := Estimate(jit.Default(), h, s, on)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || got.Seconds <= 0 {
		t.Fatalf("the header prices %+v, the container %+v", got, want)
	}
}
