package model

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// Gemma 3's vision half: a SigLIP-so400m tower (27 blocks, 1152 wide, 4096
// patches of 14px at 896), a 4x4 average pool to 256 tokens, an RMSNorm and
// one matrix into the text model's 2560.

var gemma3Text = testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")
var gemma3Proj = testmodels.Path("vlm/mmproj-gemma-3-4b-it-f16.gguf")

// openGemma3 converts the pair into one container and opens it. A missing
// file fails, naming what to fetch: a skipped vision gate is a green line that
// proves nothing.
func openGemma3(t testing.TB, opts ...Option) (*Model, *Tower) {
	t.Helper()
	if !vlmAvailable(gemma3Text, gemma3Proj) {
		testmodels.Missing(t, "%s and its mmproj %s (or the merged container) are missing. RULE 11: fetch "+
			"ggml-org/gemma-3-4b-it-GGUF's mmproj-model-f16.gguf into %s and re-run",
			gemma3Text, gemma3Proj, testmodels.Dir())
	}
	m, err := Open(jlmOfPair(t, gemma3Text, gemma3Proj), opts...)
	if err != nil {
		t.Fatal(err)
	}
	tw := m.Tower()
	if tw == nil {
		m.Close()
		t.Fatal("the merged container carries no tower, so this gate proves nothing")
	}
	return m, tw
}

// TestGemma3TowerLoads asserts every number that decides the graph, each read
// off the mmproj first.
func TestGemma3TowerLoads(t *testing.T) {
	m, tw := openGemma3(t)
	defer m.Close()
	c := tw.Cfg
	for _, k := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 27}, {"NEmbd", c.NEmbd, 1152}, {"NHead", c.NHead, 16},
		{"HeadDim", c.HeadDim, 72},
		// 4304 in the file, padded by the converter to a whole Q8_0 block with
		// zero rows and columns (convert.padFFN).
		{"NFFN", c.NFFN, 4320},
		{"ImageSz", c.ImageSz, 896}, {"PatchSz", c.PatchSz, 14}, {"Scale", c.Scale, 4},
		{"Patches", c.Patches(), 4096}, {"Seq", c.Seq(), 4096}, {"Tokens", c.Tokens(), 256},
		{"ProjDim", c.ProjDim, 2560}, {"projK", c.projK(), 1152},
	} {
		if k.got != k.want {
			t.Errorf("%s = %d, want %d", k.name, k.got, k.want)
		}
	}
	if c.Projector != "gemma3" || c.CLS || c.PreLN || c.Rope {
		t.Errorf("projector %q CLS %v PreLN %v Rope %v; gemma3 is SigLIP: no class token, "+
			"no pre-norm, a learned table", c.Projector, c.CLS, c.PreLN, c.Rope)
	}
	if tw.postLnW == nil || tw.projNorm == nil || tw.projW2.e != nil || tw.projB != nil {
		t.Errorf("post-norm %v, projector norm %v, second matrix %v, bias %v: gemma3's "+
			"projector is a norm and ONE unbiased matrix after a post-norm",
			tw.postLnW != nil, tw.projNorm != nil, tw.projW2.e != nil, tw.projB != nil)
	}
	if c.Mean != [3]float64{0.5, 0.5, 0.5} || c.Std != [3]float64{0.5, 0.5, 0.5} {
		t.Errorf("mean %v std %v, want 0.5 each", c.Mean, c.Std)
	}
}

// towerNodePair names one point of the tower in llama.cpp's dump and in
// jitllm's trace. norm marks a LayerNorm's output, which divides by its row's
// spread and so magnifies the 36 printed values' noise: it takes twice the
// bound.
type towerNodePair struct {
	dump, ours string
	norm       bool
}

// holdTowerToDump compares every pair and fails past bound (relative RMS of
// the printed values, and of the sum). Every pair must be found: a pair that
// silently matched nothing is a gate that proves nothing.
func holdTowerToDump(t *testing.T, got map[string][]float32, dump map[string]*mtmdNode,
	pairs []towerNodePair, rmsBound, sumBound float64) {
	t.Helper()
	worst := 0.0
	for _, p := range pairs {
		nd, ok := dump[p.dump]
		if !ok {
			t.Fatalf("the dump has no 2-D node %q", p.dump)
		}
		x, ok := got[p.ours]
		if !ok {
			t.Fatalf("the trace has no %q", p.ours)
		}
		rms, sr, err := nd.against(x)
		if err != nil {
			t.Fatalf("%s: %v", p.dump, err)
		}
		t.Logf("%-16s rel RMS %.4f  sum %.4e vs %.4e (%.2e)", p.dump, rms, sumOf(x), nd.sum, sr)
		worst = max(worst, rms)
		b := rmsBound
		if p.norm {
			b *= 2
		}
		if rms > b || sr > sumBound {
			t.Errorf("%s: rel RMS %.4f (bound %.2f), sum off %.2e (bound %.2e)", p.dump, rms, b, sr, sumBound)
		}
	}
	t.Logf("worst rel RMS over %d nodes: %.4f", len(pairs), worst)
}

func sumOf(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v)
	}
	return s
}

// gemma3Pairs is the tower's checkpoints in b10825's gemma3 dump: block 0's
// input, every block's output, the post-norm, the pooled-and-normed rows and
// the projector's output. The two node_ names are the dump's own for the
// soft_emb_norm multiply and the projection.
func gemma3Pairs(nLayer int) []towerNodePair {
	ps := []towerNodePair{{dump: "pos_embed", ours: "pos_embed"}}
	for i := 0; i < nLayer; i++ {
		n := fmt.Sprintf("layer_out-%d", i)
		ps = append(ps, towerNodePair{dump: n, ours: n})
	}
	return append(ps, towerNodePair{"norm_b", "post_ln", true}, towerNodePair{"node_863", "proj_in", true},
		towerNodePair{dump: "node_864", ours: "out"})
}

// TestGemma3TowerMatchesLlamaCpp holds the tower to llama-mtmd-debug node by
// node on the rainbow picture: every block's output, the pool and the
// projection. llama.cpp runs the tower in f16/f32 where jitllm quantizes
// activations to int8 per matmul (~0.4% each, RULE 7i), so the bound is
// percent: the trajectory climbs from 0.3% after block 0 to 4-6% by block 24
// and reads 3.3% at the projection, on amd64 and on arm64.
func TestGemma3TowerMatchesLlamaCpp(t *testing.T) {
	m, tw := openGemma3(t)
	defer m.Close()
	dump := parseMtmdDump(t, testmodels.Path("vlm/oracle/gemma3-rainbow.txt"))
	s := tw.testState()
	defer s.Close()
	got := towerTrace(t, s, rainbow(tw.Cfg.ImageSz))
	holdTowerToDump(t, got, dump, gemma3Pairs(tw.Cfg.NLayer), 0.1, 0.05)
	towerDumpViolation(t, tw, dump, "node_864", "pooling skipped", towerFaultPoolCorner, 0.1)
}

// towerDumpViolation runs the tower with one fault and demands the projector
// output leave the dump by more than twice the clean bound.
func towerDumpViolation(t *testing.T, tw *Tower, dump map[string]*mtmdNode, node, name string,
	f towerFault, bound float64) {
	t.Helper()
	defer func(was towerFault) { tw.Cfg.fault = was }(tw.Cfg.fault)
	tw.Cfg.fault = f
	s := tw.testState()
	defer s.Close()
	got := towerTrace(t, s, rainbow(tw.Cfg.ImageSz))
	rms, _, err := dump[node].against(got["out"])
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("violation %q: projector output rel RMS %.4f", name, rms)
	if rms < 2*bound {
		t.Errorf("violation %q reads rel RMS %.4f: the gate cannot see it", name, rms)
	}
}

// TestBidirRunIsNeverCut prefills a prompt whose image (a bidirectional run of
// 256 rows) starts mid-chunk, at a 64-row chunk width that would cut it, and
// demands the logits of the 512-row width that takes it whole. A cut leaves
// the run's earlier rows attending to keys the next chunk has not written; the
// violation lets the chunk end inside the run and must fail.
func TestBidirRunIsNeverCut(t *testing.T) {
	var emb []float32
	logits := func(chunk int, cut bool) []float32 {
		m, tw := openGemma3(t, WithJITOptions(nn.WithPrefillChunk(chunk), nn.WithTune(nn.TuneOff)))
		defer m.Close()
		if emb == nil {
			ts := tw.testState()
			e, err := ts.Encode(rainbow(tw.Cfg.ImageSz))
			if err != nil {
				t.Fatal(err)
			}
			emb = append([]float32(nil), e...)
			ts.Close()
		}
		pre := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 9), true)
		spans := []Span{{Tokens: pre}, {Embd: emb, Bidir: true}, {Tokens: m.Vocab.Encode(" It shows", false)}}
		if len(pre)%64 == 0 {
			t.Fatalf("the image starts on a chunk boundary (%d): nothing would cut it", len(pre))
		}
		s := m.NewInstance(SpanPositions(spans, m.Cfg.NEmbd) + 1)
		defer s.Close()
		s.bidirCutFault = cut
		lg, err := s.PrefillMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	whole := logits(512, false)
	small := logits(64, false)
	nm, worst := nmseOf(small, whole)
	t.Logf("64-row chunks against 512: logit NMSE %.3e, max|d| %.4f", nm, worst)
	if math.IsNaN(nm) || nm > 1e-4 {
		t.Errorf("64-row chunks read NMSE %.3e against the whole run", nm)
	}
	vn, vw := nmseOf(logits(64, true), whole)
	t.Logf("violation, a run cut at the chunk: NMSE %.3e, max|d| %.4f", vn, vw)
	if vn < 10*max(nm, 1e-6) {
		t.Errorf("a cut run reads NMSE %.3e against a clean %.3e: the gate cannot see the cut", vn, nm)
	}
}
