package engine

import (
	"encoding/binary"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// The arms fp8Restated states.
const (
	fp8Verbatim = iota // NVFP4 and MXFP8 rows as stored
	fp8Dequant         // the oracle's values of the same codes, in f32
	fp8Swapped         // NVFP4's codes high nibble first, MXFP8's scales doubled
)

// fp8Restated is the edit giving every expert bank of a source NVFP4 rows
// (a global a sheet, an expert's) and every attention projection MXFP8
// rows, their codes and scales drawn from a seed by the tensor's block and
// role; arm fp8Dequant stores the oracle's dequantized values of the same
// codes in f32 instead, and fp8Swapped the violation.
func fp8Restated(arm int) func(*jlm.Source) {
	return func(s *jlm.Source) {
		for i := range s.Tensors {
			t := &s.Tensors[i]
			if t.Block < 0 || t.Role.Vision() || t.NDim < 2 {
				continue
			}
			k, rows := int(t.Dims[0]), int(t.Dims[1])
			sheets := 1
			if t.NDim >= 3 {
				sheets = int(t.Dims[2])
			}
			h := fnv.New64a()
			binary.Write(h, binary.LittleEndian, [2]int32{t.Block, int32(t.Role)})
			rng := rand.New(rand.NewPCG(h.Sum64(), 7))
			switch {
			case jlm.ExpertBank(t.Role) && k%16 == 0:
				codes := make([]byte, sheets*rows*k/2)
				for j := range codes {
					codes[j] = byte(rng.Uint32())
				}
				scales := make([]byte, sheets*rows*k/16)
				for j := range scales {
					scales[j] = byte(0x28 + rng.IntN(0x20))
				}
				global := make([]float32, sheets)
				for j := range global {
					global[j] = float32(0.004 + 0.008*rng.Float64())
				}
				want := make([]float32, sheets*rows*k)
				for j := range sheets {
					oracle.DequantNVFP4(want[j*rows*k:], codes[j*rows*k/2:], scales[j*rows*k/16:], global[j], rows, k)
				}
				if arm == fp8Swapped {
					for j := range codes {
						codes[j] = codes[j]<<4 | codes[j]>>4
					}
				}
				raw, err := jlm.NV4Rows(codes, scales, global, sheets, rows, k)
				fp8Set(t, arm, jlm.TypeNV4, raw, want, err)
			case (t.Role == jlm.RoleAttnQ || t.Role == jlm.RoleAttnK || t.Role == jlm.RoleAttnV || t.Role == jlm.RoleAttnOut) && k%32 == 0:
				n := sheets * rows
				vals := make([]byte, n*k)
				for j := range vals {
					vals[j] = byte(rng.IntN(2))<<7 | byte(4+rng.IntN(5))<<3 | byte(rng.IntN(8))
				}
				scales := make([]byte, n*k/32)
				for j := range scales {
					scales[j] = byte(121 + rng.IntN(3))
				}
				want := make([]float32, n*k)
				oracle.DequantMXFP8(want, vals, scales, n, k)
				if arm == fp8Swapped {
					for j := range scales {
						scales[j]++
					}
				}
				raw, err := jlm.MX8Rows(vals, scales, n, k)
				fp8Set(t, arm, jlm.TypeMX8, raw, want, err)
			}
		}
	}
}

// fp8Set gives t raw rows of type ty, or in arm fp8Dequant the f32 values
// want. A layout the edit cannot make leaves a tensor the writer refuses.
func fp8Set(t *jlm.Tensor, arm int, ty jlm.Type, raw []byte, want []float32, err error) {
	t.Load = nil
	if err != nil {
		t.Type, t.Data = ty, nil
		return
	}
	if arm != fp8Dequant {
		t.Type, t.Data = ty, raw
		return
	}
	b := make([]byte, 4*len(want))
	for i, v := range want {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	t.Type, t.Data = jlm.TypeF32, b
}

// TestFP8WeightsRunOnEveryDevice runs a mixture whose expert banks are
// NVFP4 rows in expert pages and whose attention projections are MXFP8
// rows, placed by model.Open, against the same model storing the oracle's
// values of the same codes in f32: on the CPU device within the reduction
// band, under a one-frame page budget equal to the resident run, and on a
// card (the tier's upload of the rows at their RawRowBytes strides) with
// every block placed and across a seam, against the CPU. The violation
// reads NVFP4's codes high nibble first and MXFP8's scales doubled, and
// must part from the f32 model on each device.
func TestFP8WeightsRunOnEveryDevice(t *testing.T) {
	src, ok := existingModel(testmodels.Path("synth-qwen2moe.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", testmodels.Path("synth-qwen2moe.gguf"))
	}
	open := func(name string, arm int) (*model.Model, string) {
		path := filepath.Join(t.TempDir(), name+model.Ext)
		if err := convertStated(src, path, fp8Restated(arm)); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		m, err := model.Open(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		t.Cleanup(func() { m.Close() })
		return m, path
	}
	fp8, fp8Path := open("fp8", fp8Verbatim)
	f32, _ := open("f32", fp8Dequant)
	bad, _ := open("swapped", fp8Swapped)

	var nv4, mx8, paged int
	for i := range fp8.Container.Entries() {
		e := &fp8.Container.Entries()[i]
		switch e.Type {
		case jlm.TypeNV4:
			nv4++
			if fp8.Container.ExpertPaged(e) {
				paged++
			}
		case jlm.TypeMX8:
			mx8++
		}
	}
	if nv4 == 0 || mx8 == 0 || paged != nv4 {
		t.Fatalf("%d NVFP4 banks (%d in expert pages), %d MXFP8 matrices: the model does not store what the gate runs", nv4, paged, mx8)
	}
	in := fp8.Vocab.Encode("The capital of France is", true)

	run := func(m *model.Model, opts ...Option) []float32 {
		st := testInstance(t, m, len(in)+8, opts...)
		defer st.Close()
		var out []float32
		var err error
		for _, id := range in {
			if out, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), out...)
	}
	want, got, viol := run(f32), run(fp8), run(bad)
	if !allFinite(want) || !allFinite(got) {
		t.Fatal("the logits are not finite")
	}
	d, v := maxAbsDiff(got, want), maxAbsDiff(viol, want)
	if d > 1e-3 || v < 100*max(d, 1e-4) {
		t.Fatalf("CPU: NVFP4/MXFP8 %.3g from the f32 model, the violation %.3g", d, v)
	}
	t.Logf("CPU: %d NVFP4 banks and %d MXFP8 matrices %.3g from the f32 model; the violation %.3g", nv4, mx8, d, v)

	// One frame: every block and expert page evicted and read again.
	tight, err := model.Open(fp8Path)
	if err != nil {
		t.Fatal(err)
	}
	defer tight.Close()
	if !budgetEvicts(tight, 1) {
		t.Fatal("a one-frame budget evicts nothing: the paging arm proved nothing")
	}
	if p := run(tight, WithPageBudget(1)); maxAbsDiff(p, got) != 0 {
		t.Fatalf("paged through one frame: %.3g from the resident run", maxAbsDiff(p, got))
	}

	for _, dev := range testdevices.Specs("cuda", "vulkan", "metal") {
		t.Run(dev, func(t *testing.T) { fp8OnACard(t, dev, fp8, bad, in, got) })
	}
}

// fp8OnACard is TestFP8WeightsRunOnEveryDevice's card arm on device dev:
// every block placed and a seam, fp8 against the CPU's logits cpu and the
// violation bad apart from them.
func fp8OnACard(t *testing.T, dev string, fp8, bad *model.Model, in []int32, cpu []float32) {
	g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	card := func(m *model.Model, seam int) []float32 {
		st := testInstance(t, m, len(in)+8)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if seam >= 0 {
			st.SetGPULayers(seam)
		}
		if n := st.GPULayers(); seam < 0 && n != m.Blocks() || seam >= 0 && n != seam {
			t.Fatalf("%d blocks on %s, want %d (%v)", n, dev, max(seam, m.Blocks()), g.Err())
		}
		var out []float32
		for _, id := range in {
			if out, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), out...)
	}
	for _, seam := range []int{-1, fp8.Blocks() / 2} {
		cg, cb := card(fp8, seam), card(bad, seam)
		d, v := maxAbsDiff(cg, cpu), maxAbsDiff(cb, cpu)
		if !allFinite(cg) || d > 0.05 || v < 10*max(d, 1e-3) {
			t.Fatalf("%s (%s), seam %d: %.3g from the CPU, the violation %.3g", dev, g.Name(), seam, d, v)
		}
		t.Logf("%s (%s), seam %d of %d blocks: %.3g from the CPU, the violation %.3g", dev, g.Name(), seam, fp8.Blocks(), d, v)
	}
}
