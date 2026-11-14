package engine

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRouterTypesOnTheDevice runs a mixture whose ROUTER is stored in each
// type a container may carry it in -- F32, F16, BF16 and Q8_0 -- with its
// first blocks on the device, against the host with the same router.
//
// The device read a router as raw F32 or F16 and refused every other type, and
// before that read F32 whatever the file said (Mixtral's F16 router, read at
// twice its size, ran past the buffer). The converter carries a router as it
// comes, and no model small to run here has a router past F32, so the gate
// makes one: Qwen3-MOE-4x0.6B's F32 routers are re-encoded in place and both
// arms read the same bytes, keeping the comparison about the device's read of
// the type. The F32 arm runs first as the control: its NMSE is the device band.
func TestRouterTypesOnTheDevice(t *testing.T) {
	path := testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS) -- this gate proved nothing", path)
	}
	ids := []int32{1, 415, 5565, 302, 4843, 349}
	arm := func(typ quant.Type) (float64, int) {
		t.Helper()
		m, err := model.Open(jlmOf(t, path))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if !m.Cfg.MoE() {
			t.Fatal("not a mixture -- this gate proved nothing")
		}
		routers := 0
		fr := testFrames(t, m)
		for li := range m.Layers {
			l := &m.Layers[li]
			if l.Router.Rows == 0 {
				continue
			}
			if err := m.PageIn(fr, li, &m.Layers[li]); err != nil {
				t.Fatal(err)
			}
			if l.Router.Typ != quant.F32 {
				t.Fatalf("block %d's router is %v; the control arm needs F32", li, l.Router.Typ)
			}
			routers++
			if typ == quant.F32 {
				continue
			}
			// No entry: bindOne re-resolves a tensor from its entry on every
			// page-in, which would put the file's F32 bytes back.
			l.Router = reencodeRouter(t, l.Router, typ)
		}
		if routers == 0 {
			t.Fatal("no block carries a router -- this gate proved nothing")
		}
		// The entries say the type too: an instance generates one host kernel
		// per type the container lists, so without this the host arm has no
		// matvec for it.
		if typ != quant.F32 {
			ty, ok := jlm.TypeOf(typ)
			if !ok {
				t.Fatalf("the container has no %v type", typ)
			}
			for i := range m.Container.Entries() {
				if e := &m.Container.Entries()[i]; e.Role == jlm.RoleRouter {
					e.Type = ty
					// A packed type has three planes: the two the F32 entry
					// never stated are put inside its span, so the block's
					// page read stays in the page. The tensor holds its own
					// planes and reads none of these bytes.
					if typ == quant.Q8_0 {
						e.DOff, e.SCOff = e.QSOff, e.QSOff
					}
				}
			}
		}
		host := func() [][]float32 {
			st := testInstance(t, m, 32)
			defer st.Close()
			var out [][]float32
			for _, id := range ids {
				l, err := st.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), l...))
			}
			return out
		}
		want := host()
		g, err := openGPU(t, gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		st := testInstance(t, m, 32)
		defer st.Close()
		// Four blocks, not all: a router read in the wrong format misroutes
		// every token of every placed block, and four fit a shared card.
		want4 := min(4, m.Cfg.NLayer)
		st.SetDeviceLayers(g, want4)
		placed := st.GPULayers()
		if placed != want4 {
			t.Fatalf("%v: the device took %d of the %d blocks offered (%s) -- the "+
				"router is what this gate is about, so they have to be on the card",
				typ, placed, want4, g.Err())
		}
		var num, den float64
		for p, id := range ids {
			got, err := st.Forward(id)
			if err != nil {
				t.Fatalf("%v: %v (%s)", typ, err, g.Err())
			}
			for i, w := range want[p] {
				d := float64(got[i]) - float64(w)
				if math.IsNaN(d) || math.IsInf(d, 0) {
					t.Fatalf("%v pos %d logit %d is %v against %v: not finite",
						typ, p, i, got[i], w)
				}
				num += d * d
				den += float64(w) * float64(w)
			}
		}
		return num / den, placed
	}
	ctrl, n := arm(quant.F32)
	t.Logf("%d blocks on the device: F32 router NMSE %.3e (the control)", n, ctrl)
	for _, typ := range []quant.Type{quant.F16, quant.BF16, quant.Q8_0} {
		got, _ := arm(typ)
		t.Logf("%v router NMSE %.3e", typ, got)
		// The bound, from both sides: the control is the device's f32 band; a
		// router read in the wrong format routes to the wrong experts
		// everywhere.
		if lim := math.Max(10*ctrl, 1e-3); got > lim {
			t.Fatalf("%v router NMSE %.3e against a limit of %.3e (10x the F32 control "+
				"%.3e): the device did not read the router the host read", typ, got, lim, ctrl)
		}
	}
}

// reencodeRouter is F32 router r stored as typ: F16 and BF16 as raw rows, Q8_0
// as GGUF blocks packed into the device layout, as the container holds a
// packed type (QS in Data, the planes in Packed).
func reencodeRouter(t *testing.T, r model.Tensor, typ quant.Type) model.Tensor {
	t.Helper()
	n := r.Rows * r.K
	f := make([]float32, n)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.Data[4*i:]))
	}
	out := model.Tensor{Typ: typ, Rows: r.Rows, K: r.K}
	switch typ {
	case quant.F16, quant.BF16:
		h := make([]byte, 2*n)
		for i, v := range f {
			x := uint16(math.Float32bits(v) >> 16)
			if typ == quant.F16 {
				x = halfOf(v)
			}
			binary.LittleEndian.PutUint16(h[2*i:], x)
		}
		out.Data = h
	case quant.Q8_0:
		if r.K%32 != 0 {
			t.Fatalf("a router %d wide is not whole Q8_0 blocks", r.K)
		}
		b := make([]byte, n/32*34)
		for j := 0; j < n/32; j++ {
			blk := f[32*j : 32*j+32]
			var amax float32
			for _, v := range blk {
				amax = max(amax, float32(math.Abs(float64(v))))
			}
			d := amax / 127
			binary.LittleEndian.PutUint16(b[34*j:], halfOf(d))
			for i, v := range blk {
				q := float32(0)
				if d != 0 {
					q = float32(math.Round(float64(v / d)))
				}
				b[34*j+2+i] = byte(int8(max(-127, min(127, q))))
			}
		}
		qs, dw, sc, err := layout.PackWeights(layout.Q8_0, b, r.Rows, r.K)
		if err != nil {
			t.Fatal(err)
		}
		words := func(v []uint32) []byte {
			o := make([]byte, 4*len(v))
			for i, w := range v {
				binary.LittleEndian.PutUint32(o[4*i:], w)
			}
			return o
		}
		p := &nn.Packed{QS: words(qs), D: words(dw), SC: words(sc)}
		out.Data, out.Packed = p.QS, p
	default:
		t.Fatalf("no encoding for a %v router", typ)
	}
	return out
}

// halfOf truncates f to binary16. Rounding does not matter here: both arms
// read the same bytes, and the gate is about how they are read.
func halfOf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int((b>>23)&0xff) - 127 + 15
	switch {
	case exp <= 0:
		return sign
	case exp >= 31:
		return sign | 0x7c00
	}
	return sign | uint16(exp)<<10 | uint16((b>>13)&0x3ff)
}
