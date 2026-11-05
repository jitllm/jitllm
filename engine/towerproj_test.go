package engine

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// projectorRows is scripts/visiongold.py's patch_rows: a tower's last hidden
// state that is no picture, every row and channel distinct.
func projectorRows(n, width int) []float32 {
	out := make([]float32, n*width)
	for p := range n {
		for c := range width {
			fp, fc := float64(p), float64(c)
			out[p*width+c] = float32(math.Sin(0.37*fp+0.11*fc) + 0.5*math.Cos(0.013*fp*fc))
		}
	}
	return out
}

// readF32 is a little-endian float32 file.
func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v (scripts/visiongold.py writes it) -- this gate proved nothing", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

// projectorFrom is the tower's head from op from on -- the projector, where
// the ops before it are the vision model's own (Gemma 3's post norm) -- run
// on the CPU device over rows, the head's input grid tg.side wide.
func projectorFrom(t *testing.T, s *Instance, from int, rows []float32) []float32 {
	t.Helper()
	if err := s.towerReady(); err != nil {
		t.Fatal(err)
	}
	m, tr := s.m, s.tower
	ts, tg := tr.s, m.Tower
	h := m.Container.Vision().Tower.Head
	p := jlm.Program{NIn: 1, Out: h.Out - uint16(from)}
	for _, op := range h.Ops[from:] {
		op.In = append([]uint16(nil), op.In...)
		for i := range op.In {
			op.In[i] -= uint16(from)
		}
		p.Ops = append(p.Ops, op)
	}
	E := int(m.Container.Vision().NEmbd)
	n := len(rows) / E
	b := graph.NewBlock(-1)
	out, err := model.LowerProgram(b, m.Container.Vision().Tower, &p, []graph.ValueID{b.In("rows", E)}, n, m.TowerFacts(jlm.DenseBlock))
	if err != nil {
		t.Fatal(err)
	}
	b.Out(out)
	if err := graph.Verify(b.G, m.TowerFacts(jlm.DenseBlock)); err != nil {
		t.Fatal(err)
	}
	f, err := ts.formCPUGraph(b.G, n, nil)
	if err != nil {
		t.Fatal(err)
	}
	if why := ts.cpuWeights(f, -1, &m.TowerDense); why != nn.DeclineServes {
		t.Fatalf("the projector's weights: %v", why)
	}
	outRows := b.G.Values[out].Rows
	if outRows == 0 {
		outRows = n
	}
	tr.size(n, E)
	copy(tr.blk, rows)
	tr.req[composer.RequestGrid] = uint32(tg.Side)
	got := make([]float32, outRows*b.G.Values[out].Width)
	bd := &ts.cpu.bound
	*bd = cpuBound{li: -1, l: &m.TowerDense, own: ts.cpu.dev.Own(f), n: n, req: tr.req, pos: tr.posTab}
	bd.io[0], bd.io[1] = unsafe.Pointer(&tr.blk[0]), unsafe.Pointer(&got[0])
	ts.cpuRunForm(bd, f)
	return got
}

// TestProjectorMatchesTransformers holds the stated projector ALONE to
// transformers' on rows that are no picture (scripts/visiongold.py): Gemma
// 3's pool, RMSNorm and matrix, InternVL's shuffle, LayerNorm and MLP. With
// the tower out of it the only error is the projector's own, so a pool or a
// shuffle in the wrong order -- which a picture barely shows, its
// neighbouring patches alike after the blocks -- cannot hide: the rows'
// grid transposed is the violation.
func TestProjectorMatchesTransformers(t *testing.T) {
	fams := []struct {
		name, text, mm string
		// from is the first op of the head that is the projector's; lead
		// the class rows the head drops first.
		from, lead int
	}{
		{"gemma3", "gemma-3-4b-it-Q4_K_M.gguf", "vlm/mmproj-gemma-3-4b-it-f16.gguf", 1, 0},
		{"internvl", "vlm/InternVL3-1B-Instruct-Q8_0.gguf", "vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf", 0, 1},
	}
	for _, fam := range fams {
		t.Run(fam.name, func(t *testing.T) {
			text, mm := testmodels.Path(fam.text), testmodels.Path(fam.mm)
			if !sourcePresent(text, mm) {
				t.Skipf("MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
			}
			want := readF32(t, testmodels.Path(filepath.Join("vlm", "goldens", fam.name, "projector.features.f32")))
			m, err := model.Open(jlmOfPair(t, text, mm))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			s := testInstance(t, m, 64)
			defer s.Close()
			side, E := m.Tower.Side, int(m.Container.Vision().NEmbd)
			x := projectorRows(side*side, E)
			run := func(x []float32) []float32 {
				return projectorFrom(t, s, fam.from, append(make([]float32, fam.lead*E), x...))
			}
			got := run(x)
			if len(got) != len(want) {
				t.Fatalf("%d values, transformers %d", len(got), len(want))
			}
			clean, worst := nmseOf(got, want)
			t.Logf("projector alone: NMSE %.3e against transformers, max|d| %.3g", clean, worst)
			if !(clean < 1e-3) {
				t.Fatalf("the projector alone reads NMSE %.3e (bound 1e-3)", clean)
			}
			tx := make([]float32, len(x))
			for y := range side {
				for c := range side {
					copy(tx[(y*side+c)*E:][:E], x[(c*side+y)*E:][:E])
				}
			}
			v, _ := nmseOf(run(tx), want)
			t.Logf("violation (the grid transposed): NMSE %.3e", v)
			if !(v > 100*clean) {
				t.Fatalf("VIOLATION: the grid transposed reads NMSE %.3e against a clean %.3e: the gate cannot see it", v, clean)
			}
		})
	}
}
