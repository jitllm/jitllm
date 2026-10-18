package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// Gemma 3n's MobileNet-V5 on the device (jit/gpu/tier/conv.go) against the
// host, block by block and end to end.

// g3nvBlocks is the host's tower over the golden's picture: the stem's output
// and every block's.
func g3nvBlocks(t *testing.T, tw *Tower, g *g3nvGolden) (stem []float32, outs [][]float32) {
	t.Helper()
	s := tw.testState()
	defer s.Close()
	defer s.m.enterPager()()
	px, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	side := tw.Cfg.ImageSz / tw.Cfg.PatchSz
	if err := s.encodeBegin(px, side, side); err != nil {
		t.Fatal(err)
	}
	r := s.vis.mn
	stem = append([]float32(nil), r.act[r.cur][:r.h*r.w*r.c]...)
	s.vis.afterBlock = func(i int, x []float32) { outs = append(outs, append([]float32(nil), x...)) }
	if _, err := s.encodeBlocks(0); err != nil {
		t.Fatal(err)
	}
	return stem, outs
}

// g3nvDevice is one device arm of the tower: the worst block, each handed the
// host's input, against the host's output; the whole encode's rows; and the
// device's Stats after both.
type g3nvDevice struct {
	worst float64
	at    int
	out   []float32
	st    tier.Stats
}

// g3nvOnDevice runs the tower on spec opened with opts, every block placed:
// block by block, each handed the host's input (stem, then host[i-1]), and
// then the whole encode. ok is false when spec is not present.
func g3nvOnDevice(t *testing.T, tw *Tower, g *g3nvGolden, spec string, stem []float32, host [][]float32,
	opts ...tier.Option) (r g3nvDevice, ok bool) {
	t.Helper()
	gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, opts...)...)
	if err != nil || gpu == nil {
		return r, false
	}
	defer gpu.Close()
	s := tw.testState()
	defer s.Close()
	s.SetDevice(gpu)
	if n := s.GPUBlocks(); n != tw.Cfg.NLayer {
		t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, tw.Cfg.NLayer, gpu.Err())
	}
	px, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	func() {
		defer s.m.enterPager()()
		side := tw.Cfg.ImageSz / tw.Cfg.PatchSz
		if err := s.encodeBegin(px, side, side); err != nil {
			t.Fatal(err)
		}
		mr := s.vis.mn
		r.at = -1
		for i := range host {
			in := stem
			if i > 0 {
				in = host[i-1]
			}
			b := s.m.layers[s.lo+i].mn
			copy(mr.act[mr.cur], in)
			mr.h, mr.w, mr.c = b.hin, b.win, b.cin
			s.vis.prog.next = s.lo + i
			if _, err := s.encodeBlocks(1); err != nil {
				t.Fatal(err)
			}
			nm, _ := nmse32(host[i], mr.act[mr.cur][:len(host[i])])
			if math.IsNaN(nm) || nm > r.worst {
				r.worst, r.at = nm, i
			}
		}
		s.vis.prog = encodeProg{}
		if s.vis.devBlocks != int64(len(host)) || s.vis.hostBlocks != 0 {
			t.Fatalf("%s ran %d blocks and the host %d, of %d", spec, s.vis.devBlocks, s.vis.hostBlocks, len(host))
		}
	}()
	before := gpu.Stats().ConvBlocks
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	r.st = gpu.Stats()
	if n := r.st.ConvBlocks - before; n != int64(tw.Cfg.NLayer) {
		t.Fatalf("%s ran %d of %d tower blocks", spec, n, tw.Cfg.NLayer)
	}
	r.out = append([]float32(nil), out...)
	return r, true
}

// TestGemma3nTowerOnDeviceMatchesHost places every tower block on each device
// and holds each block, handed the host's input, to the host's output, then
// the whole encode to the host's rows. Against a host with symmetric padding
// or no residual adds the device must part, so the comparison sees both.
//
// Two arms, because the device has two arithmetics for a matrix's input.
// The int8 arm pins tier.Config.NoVolta: every matrix reads the activation
// quantized to int8 as the host quantizes it, so a block reads the host's
// output to 1e-6. The f16 arm pins NoMMA, which puts a CUDA card on sm_70's
// binary16 twin (kernels.MatVecMMA70, GemmVolta) wherever PTX lowers it;
// that twin is what a V100 ships and Metal's GemmTile is the same twin.
// It reads the activation unquantized, so it parts from the host by the
// host's own int8 rounding -- block 3 at 1.6e-4 on a V100, where the int8
// arm reads 3.8e-9 -- and the falsifiable form of it is the reference's rows
// at Q8_0 weights and f32 activations (EmbdQ8): the f16 arm must read closer
// to them than the int8 arm does.
func TestGemma3nTowerOnDeviceMatchesHost(t *testing.T) {
	m, g := openGemma3nV(t)
	defer m.Close()
	tw := m.Tower()
	stem, host := g3nvBlocks(t, tw, g)
	want := g3nvEncode(t, tw, g, nil)
	int8Arm := tier.WithConfig(func(c *tier.Config) { c.NoVolta = true })
	f16Arm := tier.WithConfig(func(c *tier.Config) { c.NoMMA = true })
	ran := 0
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			a, ok := g3nvOnDevice(t, tw, g, spec, stem, host, int8Arm)
			if !ok {
				t.Skipf("%s: not present", spec)
			}
			ran++
			if n := a.st.VoltaMV + a.st.TileMV + a.st.GemmF16; n != 0 {
				t.Fatalf("%s: the int8 arm built %d matvecs on binary16 activations", spec, n)
			}
			t.Logf("%s, int8: block by block worst NMSE %.3e at block %d against the host", spec, a.worst, a.at)
			if math.IsNaN(a.worst) || a.worst > 1e-6 {
				t.Fatalf("%s: block %d reads %.3e against the host", spec, a.at, a.worst)
			}
			nm, worst := nmse32(want, a.out)
			t.Logf("%s, int8: all %d blocks NMSE %.3e, max|d| %.4f against the host", spec, tw.Cfg.NLayer, nm, worst)
			if math.IsNaN(nm) || nm > 1e-3 {
				t.Fatalf("%s: the tower on the device reads NMSE %.3e against the host", spec, nm)
			}
			for _, v := range []struct {
				name string
				f    mnFaultKind
			}{{"symmetric padding", mnFaultSymmetric}, {"no residual adds", mnFaultNoResidual}} {
				mnFault = v.f
				bad := g3nvEncode(t, tw, g, nil)
				mnFault = mnFaultNone
				nb, _ := nmse32(bad, a.out)
				t.Logf("%s against a host with %s: NMSE %.3e", spec, v.name, nb)
				if !(nb > 10*nm && nb > 1e-3) {
					t.Errorf("%s: a host with %s reads %.3e against a clean %.3e", spec, v.name, nb, nm)
				}
			}

			f, _ := g3nvOnDevice(t, tw, g, spec, stem, host, f16Arm)
			if f.st.VoltaMV+f.st.TileMV+f.st.GemmF16 == 0 {
				t.Logf("%s: no binary16 twin on this device; its int8 arm is what it ships", spec)
				return
			}
			fh, _ := nmse32(want, f.out)
			fq, _ := nmse32(g.Image.EmbdQ8, f.out)
			aq, _ := nmse32(g.Image.EmbdQ8, a.out)
			t.Logf("%s, f16 (%d matvecs): block by block worst NMSE %.3e at block %d, all blocks %.3e against the host; "+
				"against f32 activations %.3e, the int8 arm %.3e", spec, f.st.VoltaMV+f.st.TileMV+f.st.GemmF16, f.worst, f.at, fh, fq, aq)
			if math.IsNaN(f.worst) || f.worst > 1e-3 || math.IsNaN(fh) || fh > 1e-3 {
				t.Fatalf("%s: the binary16 twin reads %.3e at block %d and %.3e over the tower against the host",
					spec, f.worst, f.at, fh)
			}
			if !(fq < aq) {
				t.Fatalf("%s: the binary16 twin reads %.3e against the f32-activation rows, no closer than the int8 arm's %.3e",
					spec, fq, aq)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

// g3nvStream is one encode of the golden's picture with the tower streamed
// through a device: the rows, the device's page-ins and page-outs over the
// encode, the host pager's evictions over it, and its ledger before placement, once placed, after the encode,
// at its limit and after release.
type g3nvStream struct {
	out                []float32
	placed             int
	resident           int    // blocks on the card when placement ends
	err                string // the encode's error, if it failed
	ins, outs          int
	evictions          int
	before, placedUsed uint64
	encodedUsed, after uint64
	limit              uint64
}

// g3nvStreamed encodes the golden's picture with every tower block placed on
// spec and streamed (Place.Stream) under a device budget of budget bytes (0
// is the device's own memory) and a host page budget of two vision pages, so
// the device pager evicts and re-uploads convolutional blocks mid-encode and
// every page-in re-reads a host page the host pager has reused. ok is false
// when spec is not present; a budget that places fewer than every block
// returns with placed short and nothing encoded, as does encode false.
func g3nvStreamed(t *testing.T, spec string, g *g3nvGolden, budget uint64, encode bool, extra ...tier.Option) (r g3nvStream, ok bool) {
	t.Helper()
	probe, _ := openGemma3nV(t)
	host := 2 * probe.container.PageBytes(int(probe.container.H.NBlocks))
	probe.Close()
	opts := []tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)}
	if budget > 0 {
		opts = append(opts, tier.WithBudget(budget))
	}
	gpu, err := openTier(t, append(opts, extra...)...)
	if err != nil || gpu == nil {
		return r, false
	}
	defer gpu.Close()
	m, _ := openGemma3nV(t, WithPageBudget(host), WithPlacement(Placement{Rest: &Place{On: spec, Stream: true}}))
	defer m.Close()
	tw := m.Tower()
	s := tw.testState()
	defer s.Close()
	r.before, r.limit = gpu.Budgets()[0].Used, gpu.Budgets()[0].Limit
	if err := s.SetDevice(gpu); err != nil {
		t.Fatal(err)
	}
	if r.placed = s.GPUBlocks(); r.placed != tw.Cfg.NLayer || !encode {
		return r, true
	}
	r.resident = r.placed - gpu.Stats().PageOuts
	r.placedUsed = gpu.Budgets()[0].Used
	px, err := s.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	c0 := gpu.Stats()
	_, ev := m.container.Faults()
	out, err := s.Encode(px)
	if err != nil {
		r.err = err.Error()
		return r, true
	}
	c1 := gpu.Stats()
	if n := c1.ConvBlocks - c0.ConvBlocks; n != int64(tw.Cfg.NLayer) {
		t.Fatalf("%s ran %d of %d tower blocks", spec, n, tw.Cfg.NLayer)
	}
	_, ev0 := m.container.Faults()
	r.out = append([]float32(nil), out...)
	r.ins, r.outs, r.evictions = c1.PageIns-c0.PageIns, c1.PageOuts-c0.PageOuts, int(ev0-ev)
	r.encodedUsed = gpu.Budgets()[0].Used
	if err := s.SetDevice(nil); err != nil {
		t.Fatal(err)
	}
	r.after = gpu.Budgets()[0].Used
	return r, true
}

// g3nvPaged is g3nvStreamed under the smallest budget that still places
// every block, found by bisection to a MiB, plus a MiB: room for a few of
// the tower's blocks at once beside its scratch, and not a budget on the
// knife edge of admission.
func g3nvPaged(t *testing.T, spec string, g *g3nvGolden, ref g3nvStream, extra ...tier.Option) (g3nvStream, uint64) {
	t.Helper()
	full := ref.placedUsed - ref.before
	lo, hi := uint64(0), full // lo places short, hi places every block
	for hi-lo > 1<<20 {
		mid := lo + (hi-lo)/2
		if r, _ := g3nvStreamed(t, spec, g, mid, false, extra...); r.placed == ref.placed {
			hi = mid
		} else {
			lo = mid
		}
	}
	hi += 1 << 20
	got, _ := g3nvStreamed(t, spec, g, hi, true, extra...)
	if got.err == "" && (got.placed != ref.placed || got.outs == 0 || got.ins == 0) {
		t.Fatalf("%s: under %d bytes the tower placed %d blocks and paged %d in and %d out",
			spec, hi, got.placed, got.ins, got.outs)
	}
	return got, hi
}

// TestGemma3nTowerPagesThroughTheDevice places the tower's 84 convolutional
// blocks on each device under a budget that holds a few of them, so the
// device pager evicts and re-uploads blocks during the encode, and holds the
// rows to the same tower fully resident on that device, bit for bit. The
// ledger stays under its limit and comes back to its figure before placement
// once the blocks are released. Its violation, a page-in that uploads stale
// host bytes, is TestGemma3nTowerPagesThroughTheDeviceStale (jitllmfault).
func TestGemma3nTowerPagesThroughTheDevice(t *testing.T) {
	_, g := openGemma3nV(t)
	ran := 0
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			ref, ok := g3nvStreamed(t, spec, g, 0, true)
			if !ok {
				t.Skipf("%s: not present", spec)
			}
			ran++
			if ref.err != "" {
				t.Fatal(ref.err)
			}
			if ref.placed != 84 || ref.outs != 0 {
				t.Fatalf("the resident arm placed %d blocks and paged %d out: it is not the resident tower", ref.placed, ref.outs)
			}
			got, budget := g3nvPaged(t, spec, g, ref)
			if got.err != "" {
				t.Fatalf("%s under %d bytes: %s", spec, budget, got.err)
			}
			nm, worst := nmse32(ref.out, got.out)
			t.Logf("%s: %d bytes against %d for the resident tower, %d of %d blocks on the card at once, "+
				"%d page-ins and %d page-outs and %d host evictions over the encode; NMSE %.3e, max|d| %.2e against resident",
				spec, budget, ref.placedUsed-ref.before, got.resident, got.placed, got.ins, got.outs, got.evictions, nm, worst)
			if got.resident*4 > got.placed {
				t.Fatalf("%s: %d of %d blocks stayed on the card: the budget is not below the tower", spec, got.resident, got.placed)
			}
			if got.evictions == 0 {
				t.Fatalf("%s: the host evicted nothing over the encode: no page-in re-read a reused host page", spec)
			}
			if worst != 0 {
				t.Fatalf("%s: paging the tower through the device changed its rows (NMSE %.3e)", spec, nm)
			}
			if got.encodedUsed > got.limit || got.after != got.before {
				t.Fatalf("%s: the ledger read %d before placement, %d after the encode against a %d limit, %d after release",
					spec, got.before, got.encodedUsed, got.limit, got.after)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
