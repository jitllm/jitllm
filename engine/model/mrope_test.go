package model

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// qvlFixtures are the Qwen-VL family fixtures scripts/qwenvlgold.py writes:
// transformers' own classes, converted by llama.cpp's own converter.
var qvlFixtures = []string{"synth-qwen2vl", "synth-qwen25vl", "synth-qwen3vl", "synth-qwen3vlmoe", "synth-qwen35vl",
	"synth-glm4v", "synth-glm4vmoe"}

// qvlGolden is scripts/qwenvlgold.py's record of transformers' own
// Qwen2-VL family on a prompt of text, one non-square image and text, then a
// continuation decoded token by token.
type qvlGolden struct {
	// Dtype is the reference's arithmetic, float32 unless stated.
	Dtype string  `json:"dtype"`
	Pre   []int32 `json:"pre"`
	Post  []int32 `json:"post"`
	Cont  []int32 `json:"cont"`
	Image struct {
		H    int       `json:"h"`
		W    int       `json:"w"`
		RGB  []byte    `json:"rgb"`
		Grid [3]int    `json:"grid"`
		Embd []float32 `json:"embd"`
		// EmbdQ8 is the same tower with its matrices at the converter's
		// Q8_0 (scripts/qwenvlgold.py quantize_tower).
		EmbdQ8 []float32 `json:"embd_q8"`
		// EmbdA8 is EmbdQ8 with every matmul's input at int8 per block of
		// 32 too (quantize_tower and qact): the arithmetic the engine's tower
		// is made of, so it is the reference the tower is held to tightly.
		EmbdA8 []float32 `json:"embd_a8"`
		// Deep, DeepQ8 and DeepA8 are Qwen3-VL's deepstack rows,
		// [taps][rows][width], at the same three precisions; empty for a
		// tower without them.
		Deep   []float32 `json:"deep"`
		DeepQ8 []float32 `json:"deep_q8"`
		DeepA8 []float32 `json:"deep_a8"`
	} `json:"image"`
	Prep struct {
		H    int       `json:"h"`
		W    int       `json:"w"`
		Min  int       `json:"min"`
		Max  int       `json:"max"`
		RGB  []byte    `json:"rgb"`
		OutH int       `json:"out_h"`
		OutW int       `json:"out_w"`
		PX   []float32 `json:"px"`
	} `json:"prep"`
	Pos [][3]int `json:"pos"`
	// Pos4 is XD-RoPE's four-axis positions (HunyuanVL), in place of Pos.
	Pos4   [][4]int `json:"pos4"`
	Delta  int      `json:"delta"`
	Logits []struct {
		Argmax int32     `json:"argmax"`
		Head   []float64 `json:"head"`
	} `json:"logits"`
}

func (g *qvlGolden) grid() ImageGrid {
	gr := ImageGrid{T: g.Image.Grid[0], H: g.Image.Grid[1], W: g.Image.Grid[2]}
	// HunyuanVL's picture carries a newline column and its begin and end rows.
	if len(g.Pos4) > 0 {
		gr.Line, gr.Ends = 1, 1
	}
	return gr
}

// openQVL opens a Qwen2-VL family fixture -- text GGUF and mmproj, one
// container -- and its golden. A missing fixture fails, naming the script.
func openQVL(t *testing.T, name string, opts ...Option) (*Model, *qvlGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "qwenvl", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &qvlGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(name+".gguf"), testmodels.Path("mmproj-"+name+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/qwenvlgold.py %s (RULE 11)",
			text, name)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	// The golden's picture went into the processor unresized (do_resize
	// False), so the tower's floor must not move it: Qwen3-VL's 256x256 floor
	// is above a toy tower's whole ceiling.
	if tw := m.Tower(); tw != nil && tw.Cfg.MinPixels > g.Image.H*g.Image.W {
		tw.Cfg.MinPixels = qwenMinPixels
	}
	return m, g
}

// qvlSpans is the golden's prompt as spans, the image's rows transformers'
// own tower output; grid false is the violation, the image laid out as text.
func qvlSpans(g *qvlGolden, grid bool) []Span {
	img := Span{Embd: g.Image.Embd}
	if len(g.Image.Deep) > 0 {
		img.Deep = g.Image.Deep
	}
	if grid {
		img.Grid = g.grid()
	}
	return []Span{{Tokens: g.Pre}, img, {Tokens: g.Post}}
}

// qvlRun prefills the golden's prompt into st and decodes its continuation,
// returning the logits at the prompt's last row and after each continuation
// token but the last, which is what the golden records.
func qvlRun(t *testing.T, st *State, g *qvlGolden, grid bool) [][]float32 {
	t.Helper()
	lg, err := st.PrefillMixed(qvlSpans(g, grid)...)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for _, id := range g.Cont[:len(g.Cont)-1] {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// qvlWorst is the worst NMSE of rows against the golden, and the argmax flips.
func (g *qvlGolden) dtypeName() string {
	if g.Dtype == "" {
		return "float32"
	}
	return g.Dtype
}

func qvlWorst(g *qvlGolden, rows [][]float32) (float64, int) {
	worst, flips := 0.0, 0
	for i, lg := range rows {
		nmse := llama4Cmp(g.Logits[i].Head, lg)
		if math.IsNaN(nmse) {
			return math.Inf(1), len(rows)
		}
		worst = math.Max(worst, nmse)
		if Greedy(lg) != g.Logits[i].Argmax {
			flips++
		}
	}
	return worst, flips
}

// TestMRopeLaysTheImageOutAsTransformersDoes holds spanRope to transformers'
// get_rope_index row for row -- the prompt's text, the image's (t, h, w) and
// the text after it -- and ropeOf's continuation to its rope delta.
func TestMRopeLaysTheImageOutAsTransformersDoes(t *testing.T) {
	for _, name := range qvlFixtures {
		t.Run(name, func(t *testing.T) { mRopeLaysTheImageOutAsTransformersDoes(t, name) })
	}
}

func mRopeLaysTheImageOutAsTransformersDoes(t *testing.T, name string) {
	m, g := openQVL(t, name)
	defer m.Close()
	st := m.NewState(64)
	defer st.Close()
	rows, next, err := st.spanRope(0, 0, qvlSpans(g, true))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows)+len(g.Cont) != len(g.Pos) {
		t.Fatalf("%d prompt rows and %d continuation, transformers lays out %d", len(rows), len(g.Cont), len(g.Pos))
	}
	for i, r := range rows {
		if [3]int(r[:3]) != g.Pos[i] {
			t.Fatalf("row %d at %v, transformers %v", i, r, g.Pos[i])
		}
	}
	end := len(rows)
	if d := next - end; d != g.Delta {
		t.Fatalf("the continuation resumes %d off its cache position, transformers' rope delta is %d", d, g.Delta)
	}
	for j := range g.Cont {
		want := g.Pos[end+j]
		if p := next + j; want != [3]int{p, p, p} {
			t.Fatalf("continuation row %d at %d, transformers %v", j, p, want)
		}
	}
	t.Logf("%d prompt rows, image grid %v, rope delta %d", len(rows), g.Image.Grid, g.Delta)
}

// TestMRopeMatchesTransformers is the text side of M-RoPE against transformers'
// Qwen2VLForConditionalGeneration: the prompt's image rows are the reference's
// own tower output, so what is compared is how the text model turns them and
// the rows around them, on decode, on a batch slot and on every device.
//
// The violation is the image laid out as text (sequential positions), the
// state this replaced: it must fail by orders of magnitude.
func TestMRopeMatchesTransformers(t *testing.T) {
	for _, name := range qvlFixtures {
		t.Run(name, func(t *testing.T) { mRopeMatchesTransformers(t, name) })
	}
}

func mRopeMatchesTransformers(t *testing.T, name string) {
	m, g := openQVL(t, name)
	defer m.Close()
	if len(m.rope.Runs) == 0 {
		t.Fatalf("the container's rotary is single-axis (sections %v): the gate would compare text to text",
			m.Cfg.RopeSections)
	}
	n := len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1
	check := func(what string, rows [][]float32) float64 {
		t.Helper()
		w, flips := qvlWorst(g, rows)
		if !(w < c6NMSE) || flips > 0 {
			t.Errorf("%s: worst NMSE %.3e against transformers (bound %.0e), %d argmax flips", what, w, c6NMSE, flips)
		}
		return w
	}
	st := m.NewState(n)
	host := check("host", qvlRun(t, st, g, true))
	st.Close()

	// A batch slot behind another sequence: the prompt into slot 1 and the
	// continuation through ForwardBatch, beside a text sequence in slot 0.
	b := m.NewBatch(2, n)
	if _, err := b.PrefillSeq(0, g.Pre); err != nil {
		t.Fatal(err)
	}
	lg, err := b.PrefillSeqMixed(1, qvlSpans(g, true)...)
	if err != nil {
		t.Fatal(err)
	}
	rows := [][]float32{append([]float32(nil), lg...)}
	nv := m.Cfg.NVocab
	for _, id := range g.Cont[:len(g.Cont)-1] {
		out, err := b.ForwardBatch([]int32{g.Pre[0], id})
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, append([]float32(nil), out[nv:2*nv]...))
	}
	b.Close()
	batch := check("batch slot", rows)

	// The violation.
	st = m.NewState(n)
	bad, _ := qvlWorst(g, qvlRun(t, st, g, false))
	st.Close()
	if !(bad > 1e4*math.Max(host, 1e-12)) {
		t.Errorf("the image at sequential positions reads %.3e against a clean %.3e: the gate cannot see M-RoPE",
			bad, host)
	}
	t.Logf("host %.3e, batch slot %.3e; sequential image positions (the violation) %.3e", host, batch, bad)

	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			run := func(grid bool) ([][]float32, error) {
				st := m.NewState(n)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				if got := st.GPULayers(); got != m.Cfg.NLayer || !st.HeadOnDevice() {
					return nil, fmt.Errorf("placed %d of %d blocks, head %v: %v %v", got, m.Cfg.NLayer,
						st.HeadOnDevice(), st.DeviceDeclines(), gpu.Err())
				}
				return qvlRun(t, st, g, grid), nil
			}
			rows, err := run(true)
			if err != nil {
				t.Fatal(err)
			}
			// The device's prefill chunk attends through f16 tiles where it has
			// them (c6dev_test's bound); decode is f32 throughout.
			w, flips := qvlWorst(g, rows)
			if !(w < 1e-5) || flips > 0 {
				t.Fatalf("worst NMSE %.3e against transformers, %d argmax flips", w, flips)
			}
			// The device must not build the table from a row's CACHE position,
			// which is the image at sequential positions: the host's goes up.
			if s := gpu.Stats(); s.RopeTableUploads == 0 || s.RopeTables != 0 {
				t.Fatalf("the device built %d rotary rows and took %d from the host (%q): an M-RoPE table "+
					"built from cache positions is the violation", s.RopeTables, s.RopeTableUploads, s.RopeTableWhy)
			}
			composedAfterPicture(t, m, g, spec, n)
			viol, err := run(false)
			if err != nil {
				t.Fatal(err)
			}
			v, _ := qvlWorst(g, viol)
			if !(v > 1e3*math.Max(w, 1e-12)) {
				t.Errorf("sequential image positions read %.3e on the device against %.3e", v, w)
			}

			// A batch slot on the device: the prompt's chunk as rows of the
			// slot, the continuation as ragged rows beside a text sequence.
			b := m.NewBatch(2, n)
			defer b.Close()
			b.SetDeviceLayers(gpu, -1)
			if got := b.GPULayers(); got != m.Cfg.NLayer {
				t.Fatalf("the batch placed %d of %d blocks: %v", got, m.Cfg.NLayer, gpu.Err())
			}
			if _, err := b.PrefillSeq(0, g.Pre); err != nil {
				t.Fatal(err)
			}
			lg, err := b.PrefillSeqMixed(1, qvlSpans(g, true)...)
			if err != nil {
				t.Fatal(err)
			}
			brows := [][]float32{append([]float32(nil), lg...)}
			for _, id := range g.Cont[:len(g.Cont)-1] {
				out, err := b.ForwardBatch([]int32{g.Pre[0], id})
				if err != nil {
					t.Fatal(err)
				}
				brows = append(brows, append([]float32(nil), out[nv:2*nv]...))
			}
			bw, bflips := qvlWorst(g, brows)
			if !(bw < 1e-5) || bflips > 0 {
				t.Errorf("batch slot on %s: worst NMSE %.3e, %d argmax flips", spec, bw, bflips)
			}

			// The continuation through StepRuns, one joint ragged step with a
			// second session decoding text beside it.
			a, c := m.NewState(n), m.NewState(n)
			defer a.Close()
			defer c.Close()
			a.SetDeviceLayers(gpu, -1)
			c.SetDeviceLayers(gpu, -1)
			if _, err := c.Prefill(g.Pre); err != nil {
				t.Fatal(err)
			}
			lg, err = a.PrefillMixed(qvlSpans(g, true)...)
			if err != nil {
				t.Fatal(err)
			}
			// Both sessions must take the JOINT arm, or this is decode twice.
			for _, s := range []*State{a, c} {
				if err := s.StepRefusal(); err != nil {
					t.Fatalf("a session is not steppable, so StepRuns would run it alone: %v", err)
				}
			}
			srows := [][]float32{append([]float32(nil), lg...)}
			for _, id := range g.Cont[:len(g.Cont)-1] {
				out, err := StepRuns([]Run{{State: c, Tokens: []int32{g.Pre[0]}, Logits: true},
					{State: a, Tokens: []int32{id}, Logits: true}})
				if err != nil {
					t.Fatal(err)
				}
				srows = append(srows, append([]float32(nil), out[1]...))
			}
			sw, sflips := qvlWorst(g, srows)
			if !(sw < 1e-5) || sflips > 0 {
				t.Errorf("StepRuns on %s: worst NMSE %.3e, %d argmax flips", spec, sw, sflips)
			}
			t.Logf("every block and the head on %s: decode %.3e, batch slot %.3e, StepRuns %.3e against "+
				"transformers; violation %.3e", spec, w, bw, sw, v)
		})
	}
	if ran == 0 {
		t.Log("no device on this host: the host and batch arms ran")
	}
}

// TestMRopePrefixCacheKeysOnPositions: a prompt with an image in it restores
// from the prefix cache and decodes on to transformers' continuation, and the
// same image under a transposed grid -- the same row values at different
// (t, h, w), with the text after it resuming elsewhere -- restores nothing
// past the text before the image.
func TestMRopePrefixCacheKeysOnPositions(t *testing.T) {
	for _, name := range qvlFixtures {
		t.Run(name, func(t *testing.T) { mRopePrefixCacheKeysOnPositions(t, name) })
	}
}

func mRopePrefixCacheKeysOnPositions(t *testing.T, name string) {
	m, g := openQVL(t, name)
	defer m.Close()
	n := len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1
	store := NewMemStore()
	open := func() *State {
		st := m.NewState(n)
		st.SetKVStore(store)
		mustKey(t, st, name+"/mrope")
		return st
	}
	run := func(st *State, spans []Span) [][]float32 {
		lg, err := st.PrefillCachedMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for _, id := range g.Cont[:len(g.Cont)-1] {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	spans := qvlSpans(g, true)
	// The golden's rows stand for one picture's: they need its name for the
	// pages that hold them -- and every page after -- to be named at all.
	key := ImageKey{0xab}
	spans[1].Key = &key
	first := open()
	w0, _ := qvlWorst(g, run(first, spans))
	first.Close()
	again := open()
	rows := run(again, spans)
	restored := again.KVRestored()
	again.Close()
	w1, flips := qvlWorst(g, rows)
	if restored <= len(g.Pre)+g.grid().Rows() {
		t.Fatalf("the second run restored %d positions: nothing past the image (it ends at %d), so the "+
			"pages after a picture are not named", restored, len(g.Pre)+g.grid().Rows())
	}
	if !(w0 < c6NMSE) || !(w1 < c6NMSE) || flips > 0 {
		t.Fatalf("against transformers: cold %.3e, restored %.3e (%d argmax flips)", w0, w1, flips)
	}

	// The transposed grid: the same rows, keyed apart only by their positions.
	// A picture with a newline column (HunyuanVL) holds H*(W+1) grid rows, so
	// its other grid is another factoring of that count.
	tg := g.grid()
	tg.H, tg.W = tg.W, tg.H
	if tg.Line > 0 {
		n := tg.W * (tg.H + tg.Line)
		for w := 1; w < n; w++ {
			if w != tg.H && n%(w+tg.Line) == 0 {
				tg.H, tg.W = n/(w+tg.Line), w
				break
			}
		}
	}
	swapped := []Span{spans[0], {Embd: spans[1].Embd, Grid: tg, Key: &key}, spans[2]}
	probe := m.NewState(n)
	rpA, _, _ := probe.spanRope(0, 0, spans)
	rpB, _, _ := probe.spanRope(0, 0, swapped)
	// The name the page keys hash for a whole prompt.
	named := func(sp []Span, rp [][4]int) string {
		k := probe.spanKey(0, sp, rp)
		probe.kv.seq = k
		h := sha256.New()
		probe.kv.writeSeq(h, 0, len(k))
		return string(h.Sum(nil))
	}
	// Without positions the two prompts are the same rows under the same
	// name, so a key without positions would hand one the other's pages; the
	// positions must part them.
	if named(spans, nil) != named(swapped, nil) {
		t.Fatal("the two grids' rows differ without their positions, so this arm proves nothing about positions")
	}
	if named(spans, rpA) == named(swapped, rpB) {
		t.Fatal("the two grids key the same with positions: the cache cannot tell them apart")
	}
	probe.Close()
	other := open()
	other.PrefillCachedMixed(swapped...)
	got := other.KVRestored()
	other.Close()
	if got > len(g.Pre) {
		t.Fatalf("a transposed grid restored %d positions of a prompt whose image starts at %d: the cache "+
			"handed one grid's history to the other", got, len(g.Pre))
	}
	t.Logf("cold %.3e, restored %d of %d rows then decoded %.3e against transformers; transposed grid restored %d",
		w0, restored, len(g.Pre)+g.grid().Rows()+len(g.Post), w1, got)
}

// TestMRopeKeepsTheFivePrinciples runs an image prompt through the principles'
// own conditions, each against transformers: a one-frame page budget (every
// block evicted and re-read every token), the blocks relocated onto a device
// and back home in the middle of the continuation, and a warm decode after
// the image at zero engine allocations on the host and every device.
func TestMRopeKeepsTheFivePrinciples(t *testing.T) {
	for _, name := range qvlFixtures {
		t.Run(name, func(t *testing.T) { mRopeKeepsTheFivePrinciples(t, name) })
	}
}

func mRopeKeepsTheFivePrinciples(t *testing.T, name string) {
	m, g := openQVL(t, name)
	defer m.Close()
	n := len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1

	t.Run("paging", func(t *testing.T) {
		m1, _ := openQVL(t, name)
		defer m1.Close()
		m1.SetPageBudget(m1.PageSize())
		st := m1.NewState(n)
		defer st.Close()
		w, flips := qvlWorst(g, qvlRun(t, st, g, true))
		frames, faults, _ := m1.PageStats()
		// A mixture's experts are pages of their own, smaller than a block's,
		// so a block page's bytes hold one block frame and some expert ones;
		// the page-ins are what say the blocks were evicted every token.
		if (m1.Cfg.NExpert == 0 && frames != 1) || faults < int64(len(g.Cont)*m1.Cfg.NLayer) {
			t.Fatalf("%d frame(s), %d page-in(s): the text blocks were not evicted every token", frames, faults)
		}
		if !(w < c6NMSE) || flips > 0 {
			t.Fatalf("one frame: worst NMSE %.3e against transformers, %d flips", w, flips)
		}
		t.Logf("one frame, %d page-in(s): worst NMSE %.3e against transformers", faults, w)
	})

	t.Run("relocation", func(t *testing.T) {
		ran := 0
		for _, spec := range stepDevices() {
			gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || gpu == nil {
				continue
			}
			ran++
			func() {
				defer gpu.Close()
				st := m.NewState(n)
				defer st.Close()
				lg, err := st.PrefillMixed(qvlSpans(g, true)...)
				if err != nil {
					t.Fatal(err)
				}
				rows := [][]float32{append([]float32(nil), lg...)}
				half := (len(g.Cont) - 1) / 2
				for i, id := range g.Cont[:len(g.Cont)-1] {
					// Onto the device after the image, home again later: the
					// history the image wrote moves with its blocks.
					switch i {
					case 1:
						st.SetDeviceLayers(gpu, -1)
						if st.GPULayers() != m.Cfg.NLayer {
							t.Fatalf("%s: moved %d of %d blocks: %v", spec, st.GPULayers(), m.Cfg.NLayer, gpu.Err())
						}
					case 1 + half:
						st.SetDeviceLayers(gpu, 0)
						if st.GPULayers() != 0 {
							t.Fatalf("%s: %d blocks stayed on the device", spec, st.GPULayers())
						}
					}
					if lg, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
					rows = append(rows, append([]float32(nil), lg...))
				}
				w, flips := qvlWorst(g, rows)
				if !(w < 1e-6) || flips > 0 {
					t.Fatalf("%s: relocated mid-continuation, worst NMSE %.3e against transformers, %d flips",
						spec, w, flips)
				}
				t.Logf("%s: onto the device after the image and home again: worst NMSE %.3e", spec, w)
			}()
		}
		if ran == 0 {
			t.Skip("no device on this host: nothing to relocate onto")
		}
	})

	t.Run("allocation", func(t *testing.T) {
		arm := func(t *testing.T, gpu *tier.GPU) {
			// decodeAllocs' window: positions 160..223, past the device's
			// first 128-position score grain (crossing one re-captures the
			// launch graph) and short of 256, where the host's KV cache
			// commits its next page -- both paid per span of context, not
			// per token. The warm-up starts where the prompt ends.
			const win = 64
			prompt := len(g.Pre) + g.grid().Rows() + len(g.Post)
			warm := 160 - prompt
			st := m.NewState(prompt + warm + win + 8)
			defer st.Close()
			if gpu != nil {
				if err := st.SetDevice(gpu); err != nil {
					t.Fatal(err)
				}
				if st.GPULayers() == 0 {
					t.Skip("no block placed on the device -- this arm proved nothing")
				}
			}
			if _, err := st.PrefillMixed(qvlSpans(g, true)...); err != nil {
				t.Fatal(err)
			}
			// A single-axis text model (Kimi-VL's) turns every row at its
			// cache position, so there is nothing to tell apart there; nor
			// does XD-RoPE (HunyuanVL), whose picture keeps its sequence
			// positions and moves only the other axes -- there the picture's
			// ordinal says decode follows one.
			switch {
			case m.Cfg.RopeXD:
				if len(st.ximg) == 0 || len(st.ximg[0]) == 0 {
					t.Fatal("the prompt recorded no picture: this measured text decode")
				}
			case len(m.rope.Runs) > 0 && st.ropeOf(0, st.Pos()) == st.Pos():
				t.Fatal("the image left the rotary on the cache position: this measured text decode")
			}
			for i := range warm {
				if _, err := st.Forward(int32(5 + i)); err != nil {
					t.Fatal(err)
				}
			}
			var c0 tier.Stats
			if gpu != nil {
				c0 = gpu.Stats()
			}
			r0 := m.container.Reads()
			w := countAllocs(func() {
				for i := range win {
					if _, err := st.Forward(int32(3 + i%64)); err != nil {
						t.Fatal(err)
					}
				}
			})
			captures := 0
			if gpu != nil {
				captures = gpu.Stats().Captures - c0.Captures
			}
			allocVerdict(t, fmt.Sprintf("decode after an image, %d of %d blocks placed", st.GPULayers(), m.Cfg.NLayer),
				w, win, m.container.Reads()-r0, captures)
		}
		t.Run("host", func(t *testing.T) { arm(t, nil) })
		for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
			t.Run(spec, func(t *testing.T) {
				gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
				if err != nil || gpu == nil {
					t.Skipf("%s: not present (%v)", spec, err)
				}
				defer gpu.Close()
				arm(t, gpu)
			})
		}
	})
}

// composedAfterPicture decodes the golden's continuation after its picture
// with the composer serving every block and the head the front end states
// (Config.Composed), held to transformers as the tier is. The blocks turn by
// main's table: a decode row after the picture turns at its own position,
// off its cache position, and on a text row every axis's position is the
// same. Each step serves every stated block once and declines every other by
// name, every step run without a recording so that each counts; the reasons
// are logged.
func composedAfterPicture(t *testing.T, m *Model, g *qvlGolden, spec string, n int) {
	t.Helper()
	cg, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.NoGraph = tier.ComposedBlock|tier.ComposedHead, true }))
	if err != nil || cg == nil {
		t.Fatalf("%s did not open for the composed decode: %v", spec, err)
	}
	defer cg.Close()
	st := m.NewState(n)
	defer st.Close()
	st.SetDeviceLayers(cg, -1)
	if got := st.GPULayers(); got != m.Cfg.NLayer {
		t.Fatalf("composed: placed %d of %d blocks: %v", got, m.Cfg.NLayer, cg.Err())
	}
	rows := qvlRun(t, st, g, true)
	w, flips := qvlWorst(g, rows)
	if !(w < 1e-5) || flips > 0 {
		t.Fatalf("composed: worst NMSE %.3e against transformers, %d argmax flips", w, flips)
	}
	stated, why := 0, map[string]int{}
	for li := range m.Cfg.NLayer {
		if u := m.Uncovered(li); u == Covered {
			stated++
		} else {
			why[u.String()]++
		}
	}
	steps := len(g.Cont) - 1
	nb, db, _ := cg.Stats().ServedPart(tier.ComposedBlock)
	if nb != steps*stated || db != steps*(m.Cfg.NLayer-stated) {
		t.Fatalf("composed: %d block steps served and %d declined, where %d steps of %d stated and %d other blocks are %d and %d",
			nb, db, steps, stated, m.Cfg.NLayer-stated, steps*stated, steps*(m.Cfg.NLayer-stated))
	}
	nh, dh, _ := cg.Stats().ServedPart(tier.ComposedHead)
	t.Logf("composed after the picture: worst %.3e; %d of %d blocks stated, %d block steps served, %d declined %v; head served %d, declined %d",
		w, stated, m.Cfg.NLayer, nb, db, why, nh, dh)
}
