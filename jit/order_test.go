package jit

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
)

// TestScratchIsWrittenBeforeItIsRead holds a formed attention's kernels to
// their order: a kernel reading a scratch operand comes after one writing
// it. A pass that joins a later kernel into an earlier one moves it past
// the kernels between them, and one of those may write what it reads.
func TestScratchIsWrittenBeforeItIsRead(t *testing.T) {
	r := Default().WithoutInt8(DeviceGPU)
	for _, slots := range []int{0, 64} {
		for _, keys := range []int{1, 5, 128} {
			b := graph.NewSlice()
			n := b.KVAppend(b.In("k", 1024), b.In("v", 1024), 0)
			b.Out(b.Attend(b.In("q", 2048), n, graph.Attrs{NHead: 16, NKVHead: 8, HeadDim: 128, Scale: 0.088}))
			req := Request{Graph: b.G, Step: Step{Rows: 1, Page: 64, Keys: keys}, Device: Device{Kind: DeviceGPU, Slots: slots, Lanes: 32},
				History: History{Keys: Pool{ByDim: true}, Values: Pool{Half: true}}}
			c := r.context(req.Graph, req.Part, req.Formats, req.Rotaries, req.Step, req.Device)
			c.History = req.History
			prims, err := r.decomposeOps(c, 0)
			if err != nil {
				t.Fatal(err)
			}
			var ks []*Kernel
			for _, p := range prims {
				k, err := generate(p)
				if err != nil {
					t.Fatal(err)
				}
				ks = append(ks, k)
			}
			out, err := r.Profiles[DeviceGPU].run(ks, req.Device)
			if err != nil {
				t.Fatal(err)
			}
			written := map[Operand]bool{}
			for i, k := range out {
				for _, bf := range k.Bufs {
					if o := bf.Operand; o.Temp && k.reads(o) && !k.writes(o) && !written[o] {
						t.Errorf("slots %d, keys %d: kernel %d (%s) reads %s before any kernel writes it", slots, keys, i, k.Name, o.Name)
					}
				}
				for _, s := range k.Stores {
					written[k.Bufs[s.Buf].Operand] = true
				}
			}
			t.Logf("slots %d, keys %d: %d kernels", slots, keys, len(out))
		}
	}
}
