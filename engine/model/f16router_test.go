package model

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestF16RouterOnTheDevice runs a mixture whose ROUTER is binary16 with its
// first blocks on the device, against the host with the same router.
//
// The device's router kernel read F32 whatever the file said, so
// Mixtral's F16 router was read as twice its size, past the buffer. No model
// small to run here has an F16 router, so the gate makes one:
// Qwen3-MOE-4x0.6B's F32 routers are rounded to binary16 in place and both arms
// read the same bytes, keeping the comparison about the device's read of the
// type. The F32 arm runs first as the control: its NMSE is the device band.
func TestF16RouterOnTheDevice(t *testing.T) {
	path := testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS) -- this gate proved nothing", path)
	}
	ids := []int32{1, 415, 5565, 302, 4843, 349}
	arm := func(f16 bool) (float64, int) {
		t.Helper()
		m, err := Open(jlmOf(t, path))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if !m.Cfg.MoE() {
			t.Fatal("not a mixture -- this gate proved nothing")
		}
		routers := 0
		for li := range m.layers {
			l := &m.layers[li]
			if l.router.rows == 0 {
				continue
			}
			if err := m.pageIn(li); err != nil {
				t.Fatal(err)
			}
			if l.router.typ != quant.F32 {
				t.Fatalf("block %d's router is %v; the control arm needs F32", li, l.router.typ)
			}
			routers++
			if !f16 {
				continue
			}
			n := l.router.rows * l.router.k
			h := make([]byte, 2*n)
			for i := 0; i < n; i++ {
				f := math.Float32frombits(binary.LittleEndian.Uint32(l.router.data[4*i:]))
				binary.LittleEndian.PutUint16(h[2*i:], halfOf(f))
			}
			// No entry: bindOne re-resolves a tensor from its entry on every
			// page-in, which would put the file's F32 bytes back.
			l.router = tensor{typ: quant.F16, data: h, rows: l.router.rows, k: l.router.k}
		}
		if routers == 0 {
			t.Fatal("no block carries a router -- this gate proved nothing")
		}
		// The entries say F16 too: a State generates one host kernel per type
		// the container lists, so without this the host arm has no F16 matvec.
		if f16 {
			ty, ok := jlm.TypeOf(quant.F16)
			if !ok {
				t.Fatal("the container has no F16 type")
			}
			for i := range m.container.Entries() {
				if e := &m.container.Entries()[i]; e.Role == jlm.RoleRouter {
					e.Type = ty
				}
			}
		}
		host := func() [][]float32 {
			st := m.NewState(32)
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
		g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		st := m.NewState(32)
		defer st.Close()
		// Four blocks, not all: a router read at the wrong width misroutes
		// every token of every placed block, and four fit a shared card.
		want4 := min(4, m.Cfg.NLayer)
		st.SetDeviceLayers(g, want4)
		placed := st.GPULayers()
		if placed != want4 {
			t.Fatalf("f16=%v: the device took %d of the %d blocks offered (%s) -- the "+
				"router is what this gate is about, so they have to be on the card",
				f16, placed, want4, g.Err())
		}
		var num, den float64
		for p, id := range ids {
			got, err := st.Forward(id)
			if err != nil {
				t.Fatalf("f16=%v: %v (%s)", f16, err, g.Err())
			}
			for i, w := range want[p] {
				d := float64(got[i]) - float64(w)
				if math.IsNaN(d) || math.IsInf(d, 0) {
					t.Fatalf("f16=%v pos %d logit %d is %v against %v: not finite",
						f16, p, i, got[i], w)
				}
				num += d * d
				den += float64(w) * float64(w)
			}
		}
		return num / den, placed
	}
	ctrl, n := arm(false)
	got, _ := arm(true)
	t.Logf("%d blocks on the device: F32 router NMSE %.3e (the control), "+
		"F16 router NMSE %.3e", n, ctrl, got)
	// The bound, from both sides: the control is the device's f32 band; a
	// router read at the wrong width routes to the wrong experts everywhere.
	if lim := math.Max(10*ctrl, 1e-3); got > lim {
		t.Fatalf("F16 router NMSE %.3e against a limit of %.3e (10x the F32 control "+
			"%.3e): the device did not read the router the host read", got, lim, ctrl)
	}
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
