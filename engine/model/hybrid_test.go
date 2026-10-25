package model

import (
	"encoding/binary"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"math"
	"math/rand"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"

	"github.com/jitllm/jitllm/engine/nn"
)

// A hybrid, built out of ordinary Go values: qwen3next's geometry (16 key heads
// against 32 value heads, a fused q|k|v projection, a gated double-width query,
// a short convolution) at a tiny size, so the plumbing gates run in a second.
//
// It is not an arithmetic oracle for the architecture -- only llama.cpp on the
// real weights can say that. It gates that the three entry points compute the
// same thing, which is what the recurrent state makes fragile.
const (
	hyEmbd   = 32
	hyLayer  = 4
	hyHead   = 2
	hyKVHead = 1
	hyHeadD  = 32
	hyRot    = 8 // partial rotary: a QUARTER of head_dim, the ratio qwen3next has (64 of 256)
	hyFFN    = 32
	hyVocab  = 16
	hyConv   = 4
	hyKHeads = 2
	hyVHeads = 4
	hyKDim   = 8
	hyVDim   = 8
	hyInner  = hyVHeads * hyVDim
	hyChans  = hyInner + 2*hyKHeads*hyKDim

	// The mixture half: qwen3next runs a routed mixture and a shared expert in
	// every block.
	hyExperts = 4
	hyExpUsed = 2
	hyFFNExp  = 32
	hyShExp   = 32
)

func hyF32(rnd *rand.Rand, n int) []byte {
	b := make([]byte, n*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(rnd.NormFloat64()*0.3)))
	}
	return b
}

// hyQ4 is n elements of Q4_0: an f16 scale then 16 nibble pairs per 32.
//
// The matrices are quantized and the vectors are not, as in a real file. With
// everything F32 nothing packs, and every gate asking whether a weight resolved
// its packed spans would skip silently.
func hyQ4(rnd *rand.Rand, n int) []byte {
	if n%32 != 0 {
		panic("hyQ4: not a whole number of blocks")
	}
	b := make([]byte, n/32*18)
	// One Uint64 per eight payload bytes: the big fixture asks for tens of
	// millions of these and a call per byte is most of its runtime.
	for i := 0; i < len(b); i += 18 {
		binary.LittleEndian.PutUint16(b[i:], 0x2C00) // 0.0625, finite and small
		for j := 2; j < 18; j += 8 {
			binary.LittleEndian.PutUint64(b[i+j:i+j+8:i+j+8], rnd.Uint64())
		}
	}
	return b
}

// hybridModel writes a synthetic hybrid container and opens it.
func hybridModel(t *testing.T) *Model { return hybridModelOpt(t, hyOpt{}) }

type hyOpt struct {
	// moe replaces the dense feed-forward with the routed mixture and shared
	// expert qwen3next actually carries.
	moe bool
	// big is the mixture at Qwen3-Next-80B's own head geometry. The small
	// fixture's heads fit one 32-lane subgroup; 16 heads of 256 do not, so the
	// per-head norm, rope and score walk take a different path.
	big bool
	// wideMoE is the 80B's mixture at the small head geometry: 512 experts of
	// which 10 run, each 512 wide. It cannot be combined with big (a bank would
	// be half a billion elements); it covers a shared scratch sized off the
	// widest consumer rather than NEmbd.
	wideMoE bool
	deep    bool
	// solo, when > 0, makes exactly one layer (index solo-1) full attention and
	// every other linear, placing a single device block at a chosen depth.
	solo int
	// layers overrides the block count. 8 puts full attention at 3 and 7.
	layers int
	// budget caps the page pool, so blocks are evicted and frames reused.
	budget uint64
	// manyExp gives the mixture far more experts than a short decode routes,
	// so most of the bank is never ensured by moe().
	manyExp bool
	// tiled writes the fixture's value heads in TILED order and says so
	// (jlm.FlagDeltaKeyTiled): Qwen3.5's pairing, value head vh reading key
	// head vh % kHeads. The fixture has two value heads per key head, so the
	// pairing is observable.
	tiled bool
	// ctx overrides the context length (64), for a gate whose prompt crosses
	// a device prefill chunk (nn.MaxDevicePrefillChunk).
	ctx int
	// chunk overrides the pager's fill granularity (WithChunk).
	chunk uint64
	// seed draws different weights at the same shape (7 when zero): two
	// models a device could mistake for one another by block index.
	seed int64
}

func hybridModelOpt(t *testing.T, o hyOpt) *Model {
	t.Helper()
	seed := o.seed
	if seed == 0 {
		seed = 7
	}
	rnd := rand.New(rand.NewSource(seed))
	moe := o.moe
	nLayer := hyLayer
	if o.deep {
		nLayer = 48
	}
	if o.layers > 0 {
		nLayer = o.layers
	}
	embd, head, kvHead, headD, rot := hyEmbd, hyHead, hyKVHead, hyHeadD, hyRot
	ffn, nExp, expUsed, ffnExp, shExp := hyFFN, hyExperts, hyExpUsed, hyFFNExp, hyShExp
	if o.big {
		// Qwen3-Next-80B's attention geometry with a narrowed mixture: the
		// expert count is not what any attention kernel reads, and NFFN
		// stays 5120 because it sizes the shared activation buffer.
		embd, head, kvHead, headD, rot = 2048, 16, 2, 256, 64
		ffn, nExp, expUsed, ffnExp, shExp = 5120, 8, 4, 128, 512
	}
	if o.wideMoE {
		nExp, expUsed, ffnExp, shExp = 512, 10, 512, 512
	}
	if o.manyExp {
		// The bank has to cross a 1 MiB residency chunk, or every expert
		// arrives with the attention weights and a partially-read bank
		// cannot exist.
		nExp, expUsed, ffnExp = 4096, 1, 32
	}

	var tensors []jlm.Tensor
	addT := func(ty jlm.Type, role jlm.Role, block int32, dims ...int) {
		n := 1
		d64 := make([]uint64, len(dims))
		for i, d := range dims {
			n *= d
			d64[i] = uint64(d)
		}
		data := hyF32(rnd, n)
		if ty == jlm.TypeQ4 {
			data = hyQ4(rnd, n)
		}
		e := jlm.Tensor{Role: role, Block: block, Index: -1, Type: ty,
			NDim: uint8(len(dims)), Data: data, Name: role.String()}
		copy(e.Dims[:], d64)
		tensors = append(tensors, e)
	}
	// add is the vector form (norms, biases, the conv kernel): F32, read
	// expanded rather than driven as a matvec.
	add := func(role jlm.Role, block int32, dims ...int) {
		addT(jlm.TypeF32, role, block, dims...)
	}
	// mat is the MATRIX form, quantized like a real file's.
	mat := func(role jlm.Role, block int32, k, rows int) {
		addT(jlm.TypeQ4, role, block, k, rows)
	}
	add(jlm.RoleTokenEmbd, jlm.DenseBlock, embd, hyVocab)
	add(jlm.RoleOutputNorm, jlm.DenseBlock, embd)

	kinds := make([]jlm.LayerKind, nLayer)
	for i := int32(0); i < int32(nLayer); i++ {
		add(jlm.RoleAttnNorm, i, embd)
		add(jlm.RoleFFNNorm, i, embd)
		if moe {
			// A bank is [k][rows per expert][expert]: each expert is its own
			// contiguous sheet.
			add(jlm.RoleRouter, i, embd, nExp)
			addT(jlm.TypeQ4, jlm.RoleExpGateBank, i, embd, ffnExp, nExp)
			addT(jlm.TypeQ4, jlm.RoleExpUpBank, i, embd, ffnExp, nExp)
			addT(jlm.TypeQ4, jlm.RoleExpDownBank, i, ffnExp, embd, nExp)
			add(jlm.RoleShRouter, i, embd)
			mat(jlm.RoleShExpGate, i, embd, shExp)
			mat(jlm.RoleShExpUp, i, embd, shExp)
			mat(jlm.RoleShExpDown, i, shExp, embd)
		} else {
			mat(jlm.RoleFFNGate, i, embd, ffn)
			mat(jlm.RoleFFNUp, i, embd, ffn)
			mat(jlm.RoleFFNDown, i, ffn, embd)
		}
		// Every fourth layer is full attention, the 80B's pattern, written per
		// layer because the container says so per layer.
		attn := (i+1)%4 == 0
		if o.solo > 0 {
			attn = int(i) == o.solo-1
		}
		if attn {
			kinds[i] = jlm.LayerFullAttn
			mat(jlm.RoleAttnQ, i, embd, 2*head*headD) // query and its gate
			mat(jlm.RoleAttnK, i, embd, kvHead*headD)
			mat(jlm.RoleAttnV, i, embd, kvHead*headD)
			mat(jlm.RoleAttnOut, i, head*headD, embd)
			add(jlm.RoleAttnQNorm, i, headD)
			add(jlm.RoleAttnKNorm, i, headD)
			continue
		}
		kinds[i] = jlm.LayerLinearAttn
		mat(jlm.RoleAttnQKV, i, embd, hyChans)
		mat(jlm.RoleAttnGate, i, embd, hyInner)
		mat(jlm.RoleSSMBA, i, embd, 2*hyVHeads)
		mat(jlm.RoleSSMOut, i, hyInner, embd)
		add(jlm.RoleSSMConv1d, i, hyConv, hyChans)
		if o.ctx > 0 {
			// A decay under one, as every real file has (ssm_a is -exp(A_log)).
			// A positive random value makes the gate a growth factor, which
			// overflows over a prompt that crosses a device chunk.
			addT(jlm.TypeF32, jlm.RoleSSMA, i, hyVHeads)
			d := tensors[len(tensors)-1].Data
			for j := 0; j < len(d); j += 4 {
				d[j+3] |= 0x80 // the sign bit of a little-endian f32
			}
		} else {
			add(jlm.RoleSSMA, i, hyVHeads)
		}
		add(jlm.RoleSSMDtBias, i, hyVHeads)
		add(jlm.RoleSSMNorm, i, hyVDim)
	}

	toks := make([]string, hyVocab)
	kindsTok := make([]jlm.TokenKind, hyVocab)
	scores := make([]float32, hyVocab)
	for i := range toks {
		toks[i] = string(rune('a' + i))
		kindsTok[i] = jlm.TokenNormal
	}
	cfg := &jlm.Config{
		Arch: jlm.ArchQwen3Next, NLayer: uint32(nLayer), NEmbd: uint32(embd),
		NHead: uint32(head), NKVHead: uint32(kvHead), HeadDim: uint32(headD), NRot: uint32(rot),
		NFFN: uint32(ffn), NVocab: hyVocab, NCtx: 64, RMSEps: 1e-6,
		RopeBase: 10000, EmbdScale: 1, AttnFactor: 1,
		Flags: jlm.FlagTiedEmbd | jlm.FlagRopeNeox | jlm.FlagQKNorm |
			jlm.FlagAttnOutGate,
		LayerKinds: kinds,
		SSM: jlm.SSMConfig{ConvKernel: hyConv, Groups: hyKHeads,
			Inner: hyInner, StateSize: hyKDim, NHeadV: hyVHeads},
	}
	if o.tiled {
		cfg.Flags |= jlm.FlagDeltaKeyTiled
	}
	if o.ctx > 0 {
		cfg.NCtx = uint32(o.ctx)
	}
	if moe {
		cfg.NExpert, cfg.NExpertUsed = uint32(nExp), uint32(expUsed)
		cfg.NFFNExp, cfg.NFFNShExp = uint32(ffnExp), uint32(shExp)
	}
	src := &jlm.Source{
		Config: cfg,
		Vocab: &jlm.Vocab{
			Kind: jlm.VocabBPE, Tokens: toks, Scores: scores, Kinds: kindsTok,
			BOS: 0, EOS: 1, Unk: -1, Pad: -1, Sep: -1, Mask: -1,
		},
		Tensors: tensors,
	}
	dst := filepath.Join(t.TempDir(), "hybrid"+jlm.Ext)
	if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("jlm.Write: %v", err)
	}
	opts := []Option{noTune}
	if o.chunk > 0 {
		opts = append(opts, WithChunk(o.chunk))
	}
	if o.budget > 0 {
		opts = append(opts, WithPageBudget(o.budget))
	}
	m, err := Open(dst, opts...)
	if err != nil {
		t.Fatalf("Open a hybrid container: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestHybridRuns is the load gate: every tensor the architecture needs is
// bound, every shape agrees, a token comes out finite, and the recurrent state
// moved (a linear layer that never ran still yields finite logits).
func TestHybridRuns(t *testing.T) {
	m := hybridModel(t)
	if !m.Cfg.Hybrid() {
		t.Fatal("the container's layer map did not survive the load")
	}
	s := m.NewState(16)
	defer s.Close()
	before := append([]float32(nil), s.rstate[0]...)
	lg, err := s.Forward(3)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range lg {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is %v", i, v)
		}
	}
	same := true
	for i, v := range s.rstate[0] {
		if v != before[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the recurrent state did not change: the delta rule never ran")
	}
}

// TestHybridStateIsConstantInContext is the architecture's whole claim, stated
// as a test: the memory a linear layer keeps does not grow with the
// conversation. A KV cache fails this on the second token.
func TestHybridStateIsConstantInContext(t *testing.T) {
	m := hybridModel(t)
	s := m.NewState(32)
	defer s.Close()
	want := len(s.rstate[0]) + len(s.rconv[0])
	for i := 0; i < 8; i++ {
		if _, err := s.Forward(int32(i % hyVocab)); err != nil {
			t.Fatal(err)
		}
		if got := len(s.rstate[0]) + len(s.rconv[0]); got != want {
			t.Fatalf("after %d tokens the recurrent state is %d floats, want %d", i+1, got, want)
		}
	}
}

// TestHybridPrefillMatchesForward and TestHybridBatchMatchesForward: the
// recurrent state has to be carried identically by all three entry points.
func TestHybridPrefillMatchesForward(t *testing.T) {
	m := hybridModel(t)
	ids := []int32{3, 9, 1, 14, 6, 2, 11, 4}

	a := m.NewState(32)
	defer a.Close()
	var want []float32
	for _, id := range ids {
		lg, err := a.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		want = lg
	}
	b := m.NewState(32)
	defer b.Close()
	got, err := b.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	hyClose(t, "prefill", want, got)
	hyCloseState(t, a, b)
}

func TestHybridBatchMatchesForward(t *testing.T) {
	m := hybridModel(t)
	const nseq, steps = 3, 6
	streams := make([][]int32, nseq)
	for i := range streams {
		streams[i] = make([]int32, steps)
		for j := range streams[i] {
			streams[i][j] = int32((3 + i*7 + j*5) % hyVocab)
		}
	}
	want := make([][]float32, nseq)
	for i := range streams {
		s := m.NewState(steps + 1)
		for _, id := range streams[i] {
			lg, err := s.Forward(id)
			if err != nil {
				s.Close()
				t.Fatal(err)
			}
			want[i] = lg
		}
		s.Close()
	}
	b := m.NewBatch(nseq, steps+1)
	defer b.Close()
	var out []float32
	for j := 0; j < steps; j++ {
		row := make([]int32, nseq)
		for i := range streams {
			row[i] = streams[i][j]
		}
		lg, err := b.ForwardBatch(row)
		if err != nil {
			t.Fatal(err)
		}
		out = lg
	}
	for i := 0; i < nseq; i++ {
		hyClose(t, "batch row", want[i], out[i*hyVocab:(i+1)*hyVocab])
	}
}

func hyClose(t *testing.T, what string, want, got []float32) {
	t.Helper()
	if Greedy(want) != Greedy(got) {
		t.Errorf("%s: greedy token %d, want %d", what, Greedy(got), Greedy(want))
	}
	var worst float64
	for i := range want {
		if d := math.Abs(float64(want[i] - got[i])); d > worst {
			worst = d
		}
	}
	if worst > 1e-3 {
		t.Errorf("%s: worst logit difference %.3e", what, worst)
	}
}

// hyCloseState compares the recurrent state itself, which is what the logits
// only see through one more layer of mixing.
func hyCloseState(t *testing.T, a, b *State) {
	t.Helper()
	for li := range a.rstate {
		if a.rstate[li] == nil {
			continue
		}
		var worst float64
		for i := range a.rstate[li] {
			if d := math.Abs(float64(a.rstate[li][i] - b.rstate[li][i])); d > worst {
				worst = d
			}
		}
		if worst > 1e-4 {
			t.Errorf("layer %d recurrent state differs by %.3e", li, worst)
		}
		for i := range a.rconv[li] {
			if a.rconv[li][i] != b.rconv[li][i] {
				t.Errorf("layer %d convolution state differs at %d: %v vs %v",
					li, i, a.rconv[li][i], b.rconv[li][i])
				break
			}
		}
	}
}

// TestEveryLayerTensorIsBound walks the layer struct by reflection and asserts
// every tensor field resolved its packed spans. A tensor missing from
// bindPacked's list otherwise fails only at the first matvec of a real model.
// Reflection covers a new field of layer the day it is added.
func TestEveryLayerTensorIsBound(t *testing.T) {
	m := hybridModel(t)
	tt := reflect.TypeOf(tensor{})
	checked := 0
	for li := range m.layers {
		// Open touches no page, so fault the block in first: pageIn is where a
		// weight meets its bytes.
		if err := m.pageIn(li); err != nil {
			t.Fatalf("page in block %d: %v", li, err)
		}
		v := reflect.ValueOf(&m.layers[li]).Elem()
		for f := 0; f < v.NumField(); f++ {
			fv := v.Field(f)
			if fv.Type() != tt {
				continue
			}
			// Reading unexported fields needs the unsafe view; the test is in
			// the package, so the addresses are legitimate.
			te := (*tensor)(v.Field(f).Addr().UnsafePointer())
			if te.e == nil {
				continue // legitimately absent for this layer kind
			}
			checked++
			if te.packed == nil {
				continue // an unpacked type reads its bytes directly
			}
			// QS is the signal: `get` fills t.data whether or not bindOne ran,
			// and only bindOne sets Packed.QS.
			if len(te.packed.QS) == 0 {
				t.Errorf("layer %d field %q (%v) has an unresolved packed span: "+
					"bindPacked does not know about it",
					li, v.Type().Field(f).Name, te.e.Role)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no tensor fields were examined; this gate proves nothing")
	}
	t.Logf("%d bound tensors across %d layers", checked, len(m.layers))
}

// TestLinearBlockIsDeclinedForItsOwnReason: a linear block must reach the device
// as a linear block. One plan built from model-wide config carries AttnOutGate
// onto linear blocks too, and declineReason would refuse block 0 for a gate it
// does not have. Asserts the plan, since the defect is in what the model offers.
func TestLinearBlockIsDeclinedForItsOwnReason(t *testing.T) {
	m := hybridModel(t)
	defer m.Close()
	c := m.Cfg
	if !c.AttnOutGate {
		t.Fatal("the fixture has no attention out-gate, so this gate is about nothing")
	}
	if c.LayerKind(0) != jlm.LayerLinearAttn {
		t.Fatalf("block 0 is %v; this gate needs a linear block first, which is also the "+
			"arrangement that makes offerRange stop at it", c.LayerKind(0))
	}
	s := m.NewState(8)
	defer s.Close()

	// The plan the linear arm of offerRange builds, reproduced from the same
	// fields it reshapes.
	plan := s.plan
	g := c.delta()
	plan.Recurrent = nn.RecurrentPlan{Conv: g.conv, Chans: g.chans, KHeads: g.kHeads,
		VHeads: g.vHeads, KDim: g.kDim, VDim: g.vDim,
		StateLen: g.deltaStateLen(), ConvState: g.convStateLen()}
	plan.AttnOutGate = false

	if plan.Recurrent.Conv == 0 {
		t.Fatal("the linear plan carries no Recurrent.Conv, so declineReason would not " +
			"reach the recurrent case at all")
	}
	if plan.AttnOutGate {
		t.Fatal("a linear block's plan still carries AttnOutGate: a device that learns the " +
			"delta rule would still decline block 0 for a gate the block does not have")
	}
	// A full block of the same model keeps it.
	if c.LayerKind(3) == jlm.LayerLinearAttn {
		t.Fatal("block 3 is linear; this fixture cannot show the gate is kept where it belongs")
	}
	// The model-wide flag is untouched, so a full-attention block still carries
	// the gate. (s.plan is only populated by SetDeviceLayers, which is why this
	// asserts the source it is built from.)
	if !c.AttnOutGate {
		t.Fatal("Config.AttnOutGate was cleared: a full-attention block would lose its gate")
	}
}

// TestHybridAttentionBlockOnDevice is the fast end-to-end gate for a
// qwen3next-shaped attention block: partial rotary (a quarter of the head, as
// qwen3next rotates 64 of 256), a double-width query with its gate interleaved
// per head, and QK-norm. The head is 32 wide because the device's per-head norm
// is a 32-lane kernel. It runs with and without the mixture, since the dense arm
// cannot see a fault in the routed or shared expert.
func TestHybridAttentionBlockOnDevice(t *testing.T) {
	for _, c := range []struct {
		name string
		opt  hyOpt
	}{
		{"dense", hyOpt{}}, {"moe", hyOpt{moe: true}}, {"tiled", hyOpt{tiled: true}},
		{"big", hyOpt{moe: true, big: true}},
		{"widemoe", hyOpt{moe: true, wideMoE: true}}} {
		// Every backend, not "gpu:0". See hybridDeviceNMSEOn.
		for _, spec := range stepDevices() {
			t.Run(c.name+"/"+spec, func(t *testing.T) {
				hybridDeviceBlockOn(t, c.opt, spec, false)
				if composedByDefault {
					// The composer's counts are of the steps emitted, which a
					// recording replays: the arm that counts runs without one.
					hybridDeviceBlockOn(t, c.opt, spec, true)
				}
			})
		}
	}
}

// hybridDeviceBound is what a placed block may differ from the host by. It
// bounds the block, not the stack, so exceeding it triggers a control rather
// than a verdict -- see hybridDeviceBlockOn.
const hybridDeviceBound = 5e-3

// hybridDeviceBlockOn runs every block of the fixture on the named backend and
// compares the logits against the host, position by position.
//
// Exceeding the bound runs a shallow control rather than failing: with every
// block placed, block 0's int8 rounding travels several recurrences, and that
// amplification grows steeply with the depth below the seam. The control is the
// same geometry at two layers with the seam one block from the bottom. A fault
// in the block moves both arms; amplification moves only the deep one. See
// docs/engineering-history/placement.md 15i.
func hybridDeviceBlockOn(t *testing.T, o hyOpt, spec string, noGraph bool) {
	worst, at := hybridDeviceNMSEOn(t, o, spec, noGraph)
	if worst <= hybridDeviceBound {
		return
	}
	so := o
	so.layers, so.solo = 2, 2
	ctl, cat := hybridDeviceNMSEOn(t, so, spec, noGraph)
	if ctl > hybridDeviceBound {
		t.Fatalf("pos %d: logit NMSE %.3e, and the SHALLOW control is %.3e at "+
			"pos %d -- both arms moved, so the device block does not compute "+
			"what the host does", at, worst, ctl, cat)
	}
	t.Logf("pos %d: logit NMSE %.3e is AMPLIFICATION: the same block with one "+
		"host block below it reads %.3e (%.0fx over two more recurrences)",
		at, worst, ctl, worst/ctl)
}

// hybridDeviceNMSEOn places every block of the fixture on a named backend and
// returns the worst per-position logit NMSE against the host, with the position
// it was at. "gpu:0" resolves to CUDA wherever it exists, so a gate asking only for
// "gpu:0" never ran the gated delta rule on SPIR-V.
func hybridDeviceNMSEOn(t *testing.T, o hyOpt, spec string, noGraph bool) (float64, int) {
	m := hybridModelOpt(t, o)
	defer m.Close()
	if !m.Cfg.AttnOutGate || m.Cfg.NRot == m.Cfg.HeadDim {
		t.Fatal("the fixture lost the out-gate or the partial rotary; this gate is about both")
	}

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	host := m.NewState(16)
	defer host.Close()
	want := make([][]float32, len(ids))
	for i, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want[i] = append([]float32(nil), l...)
	}

	// Pinned, because every gate here is a comparison: tuneSplit measures a
	// k-split per placement and a split is the f32 reduction order, so two
	// unpinned placements disagree in the last bits, which a deep hybrid
	// amplifies (TestHybridDeviceErrorIsAmplification).
	opts := []tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)}
	if noGraph {
		opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))
	}
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()

	dev := m.NewState(16)
	defer dev.Close()
	// Every block of the fixture, linear ones included.
	dev.SetDeviceLayers(g, m.Cfg.NLayer)
	if dev.GPULayers() == 0 {
		t.Skipf("the device took no block of this fixture: %s", g.Err())
	}
	t.Logf("%d of %d blocks placed", dev.GPULayers(), m.Cfg.NLayer)
	// The census: "N of M placed" does not say which or why.
	for _, d := range dev.DeviceDeclines() {
		t.Logf("  %d declined: %s", d.Blocks, d.Why)
	}
	if e := g.Err(); e != "" {
		t.Logf("  tier says: %s", e)
	}

	worst, at := 0.0, -1
	// Every position, not the last one: a cache written wrong diverges once and
	// stays diverged, a scratch read before it is written alternates, and
	// position 0 is the only one whose attention reads a key the same call wrote.
	for p, id := range ids {
		l, err := dev.Forward(id)
		if err != nil {
			t.Fatalf("device: %v", err)
		}
		got, w := l, want[p]
		// Non-finite first and unconditionally: `NaN > bound` is false, so an
		// NMSE alone reports a NaN row as a perfect match.
		var num, den float64
		for i := range w {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("pos %d: logit %d is %g on the device and %g on the host: "+
					"not finite, which no tolerance can see", p, i, got[i], w[i])
			}
			d := float64(got[i] - w[i])
			num += d * d
			den += float64(w[i]) * float64(w[i])
		}
		nmse := num / den
		t.Logf("pos %d: logit NMSE %.3e", p, nmse)
		if nmse > worst {
			worst, at = nmse, p
		}
	}
	if noGraph {
		composedCovers(t, m, len(ids), dev.GPULayers(), true, g)
	}
	return worst, at
}

// TestHybridDeviceErrorIsAmplification prices how far a device block's own
// rounding travels through the host blocks below it. Both tiers quantize
// activations to int8 and round differently, and a gated delta rule is a
// recurrence, so the error should track the depth below the seam. Both arms
// place exactly one block of the same shape; only its depth changes, so a bug
// in the block would move both together.
func TestHybridDeviceErrorIsAmplification(t *testing.T) {
	ids := []int32{1, 2, 3, 4}
	// The seam moves by the model's depth, not the block's index. solo=1 makes
	// block 0 the attention one and every block after it linear, and offering one
	// places exactly that block on both arms. hybridModelOpt seeds its generator
	// at 7 and emits block 0's tensors first, so the placed block is
	// byte-identical between the two depths.
	run := func(nLayer int) float64 {
		m := hybridModelOpt(t, hyOpt{moe: true, layers: nLayer, solo: 1})
		defer m.Close()
		// A tier per arm, because a tier is one model: its blocks are keyed by
		// index and its lone matvecs by pointer, so the second arm on the
		// first's tier ran against the first model's state.
		// Pinned: see hybridDeviceNMSEOn.
		g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		host := m.NewState(16)
		defer host.Close()
		dev := m.NewState(16)
		defer dev.Close()
		dev.SetDeviceLayers(g, 1)
		if dev.GPULayers() != 1 {
			t.Skipf("wanted one block over %d layers, got %d: %s",
				nLayer, dev.GPULayers(), g.Err())
		}
		var num, den float64
		for _, id := range ids {
			w, err := host.Forward(id)
			if err != nil {
				t.Fatalf("host: %v", err)
			}
			want := append([]float32(nil), w...)
			got, err := dev.Forward(id)
			if err != nil {
				t.Fatalf("device: %v", err)
			}
			for i := range want {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
					t.Fatalf("%d layers: logit %d is %g, not finite", nLayer, i, got[i])
				}
				d := float64(got[i] - want[i])
				num += d * d
				den += float64(want[i]) * float64(want[i])
			}
		}
		return num / den
	}
	// 47 host blocks below the seam against 3.
	deep, shallow := run(48), run(4)
	t.Logf("one device block, 47 host blocks below it: NMSE %.3e", deep)
	t.Logf("one device block,  3 host blocks below it: NMSE %.3e", shallow)
	if shallow == 0 {
		t.Fatal("the shallow arm is bit-identical, so this measures nothing: " +
			"the device block did not run")
	}
	// The ordering is asserted: if the error does not track the depth below the
	// seam, the arms measure the block itself and every "that is just
	// amplification" reading in this file loses its evidence.
	if deep <= shallow {
		t.Fatalf("44 more recurrences below the seam did not amplify: %.3e at "+
			"48 layers against %.3e at 4 -- the error does not track the depth, "+
			"so it is not amplification", deep, shallow)
	}
	t.Logf("amplification over 44 more blocks: %.1fx", deep/shallow)
}

// TestPagedWeightsAreNeverServedStale runs a mixture whose host pages share
// one frame, with a device attached for single matvecs, and requires the
// logits of the same model with nothing paged. The device keeps its copy of a
// lone matvec's weight keyed on the pointer, and every block's router sits at
// the same offset of the same frame: unless the pager tells it the frame was
// refilled, the device answers every block's routing with the first block's
// router (NMSE ~0.9 here; placement.md 15v). It is also what made
// TestHybridDeviceErrorIsAmplification fail one run in ten, through a closed
// model's frames reused by the next model's.
//
// The page weights must actually reach the device, or the equality is the
// host against itself: both arms serve the same matvecs, and the paged arm
// forgets copies, which only a page weight on the device can give it.
func TestPagedWeightsAreNeverServedStale(t *testing.T) {
	ids := []int32{1, 2, 3, 4}
	run := func(budget uint64) ([][]float32, tier.Stats) {
		m := hybridModelOpt(t, hyOpt{moe: true, layers: 8, budget: budget})
		defer m.Close()
		if budget > 0 {
			var all uint64
			for i := 0; i < m.container.NPages(); i++ {
				all += m.container.PageBytes(i)
			}
			// A zero budget is unlimited; one that holds every page evicts
			// nothing and the two arms would be the same configuration.
			if got := m.container.Budget(); got == 0 || got >= all {
				t.Fatalf("budget %d against %d of pages: nothing is evicted", got, all)
			}
		}
		// Pinned: see hybridDeviceNMSEOn.
		g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		// Zero blocks: the device serves lone matvecs and nothing else, so every
		// block runs on the host and pages through the one frame.
		if err := s.SetDeviceLayers(g, 0); err != nil {
			t.Fatal(err)
		}
		if !s.forgets {
			t.Fatal("the tier is not told when a frame is refilled, so a page's weights are not offered")
		}
		var out [][]float32
		for _, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out, g.Stats()
	}
	want, resident := run(0)
	got, paged := run(1)
	if paged.Served == 0 || paged.Served != resident.Served {
		t.Fatalf("the device served %d matvecs paged and %d resident: the arms did not "+
			"run the same configuration", paged.Served, resident.Served)
	}
	for p := range want {
		var num, den float64
		for i := range want[p] {
			d := float64(got[p][i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		if num != 0 {
			t.Fatalf("pos %d: paged logits differ from resident ones, NMSE %.3e: "+
				"the device answered for a refilled frame (%d copies forgotten)",
				p, num/den, paged.Forgotten)
		}
	}
	if paged.Forgotten == 0 {
		t.Fatalf("no copy was forgotten in %d served matvecs: no page weight reached the "+
			"device, so this measures nothing", paged.Served)
	}
	t.Logf("%d matvecs served on the device, %d copies forgotten as frames were refilled; "+
		"paged and resident logits identical", paged.Served, paged.Forgotten)
}

// TestHybridPrefillOnDeviceDoesNotDemote is the gate on a hybrid's prompt. A
// batch the device cannot prepare must not demote the placement: the host
// restart re-runs linear blocks whose gated delta rule has already advanced,
// which is a wrong answer, not a slowdown. The assertion is the demotion count,
// because a demotion on the first chunk computes the right answer anyway. It
// runs dense and tiled as well as mixture: prepBatch refuses a recurrent block
// by name, since emitLinear steps one token of the delta rule per launch.
func TestHybridPrefillOnDeviceDoesNotDemote(t *testing.T) {
	for _, c := range []struct {
		name string
		o    hyOpt
	}{{"moe", hyOpt{moe: true}}, {"dense", hyOpt{}}, {"tiled", hyOpt{tiled: true}}} {
		t.Run(c.name, func(t *testing.T) { hybridPrefillOnDevice(t, c.o) })
	}
}

func hybridPrefillOnDevice(t *testing.T, o hyOpt) {
	m := hybridModelOpt(t, o)
	defer m.Close()

	// Pinned: see hybridDeviceNMSEOn.
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	host := m.NewState(32)
	defer host.Close()
	want, err := host.Prefill(ids)
	if err != nil {
		t.Fatalf("host prefill: %v", err)
	}
	want = append([]float32(nil), want...)

	dev := m.NewState(32)
	defer dev.Close()
	dev.SetDeviceLayers(g, 4)
	if dev.GPULayers() == 0 {
		t.Skipf("the device took no block of this fixture: %s", g.Err())
	}
	got, err := dev.Prefill(ids)
	if err != nil {
		t.Fatalf("device prefill: %v", err)
	}
	if d := dev.DeviceDemotions(); d != 0 {
		t.Fatalf("the device demoted %d time(s) prefilling a mixture: %s", d, g.Err())
	}
	if dev.GPULayers() == 0 {
		t.Fatal("the placement is empty after a prefill that reported no demotion")
	}
	var num, den float64
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("logit %d is %g on the device, %g on the host: not finite",
				i, got[i], want[i])
		}
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	nmse := num / den
	t.Logf("prefill logit NMSE %.3e over %d tokens", nmse, len(ids))
	if nmse > 5e-3 {
		t.Fatalf("prefill logit NMSE %.3e: the device chunk does not compute "+
			"what the host does", nmse)
	}
}

// TestHybridSecondSessionIsClean runs two device sessions over one tier the way
// `jitllm verify` does: one State generates and is closed, then a second
// attaches and is diffed against the host. A placement that leaks or a graph
// that outlives its recorder shows up only on the second.
func TestHybridSecondSessionIsClean(t *testing.T) {
	for _, blocks := range []int{4, 8} {
		t.Run(itoa(blocks)+"-block", func(t *testing.T) {
			hybridSecondSession(t, blocks)
		})
	}
}

// hybridSecondSession offers all `offered` blocks and expects every one placed:
// the device runs the linear blocks as well as the attention ones.
func hybridSecondSession(t *testing.T, offered int) {
	m := hybridModelOpt(t, hyOpt{moe: true, layers: offered})
	defer m.Close()

	// Pinned: see hybridDeviceNMSEOn.
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	host := m.NewState(16)
	defer host.Close()
	want := make([][]float32, len(ids))
	for i, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want[i] = append([]float32(nil), l...)
	}

	first := m.NewState(16)
	first.SetDeviceLayers(g, offered)
	if first.GPULayers() != offered {
		n := first.GPULayers()
		first.Close()
		t.Fatalf("the device took %d of %d blocks of this fixture: %v", n, offered, g.Err())
	}
	for _, id := range ids {
		if _, err := first.Forward(id); err != nil {
			t.Fatalf("first session: %v", err)
		}
	}
	first.Close()

	second := m.NewState(16)
	defer second.Close()
	second.SetDeviceLayers(g, offered)
	if n := second.GPULayers(); n != offered {
		t.Fatalf("the second session placed %d block(s) where the first placed %d: %v",
			n, offered, g.Err())
	}
	for p, id := range ids {
		got, err := second.Forward(id)
		if err != nil {
			t.Fatalf("second session: %v", err)
		}
		var num, den float64
		for i := range want[p] {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("second session pos %d: logit %d is %g, not finite", p, i, got[i])
			}
			d := float64(got[i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		if nmse := num / den; nmse > 5e-3 {
			t.Fatalf("second session pos %d: logit NMSE %.3e", p, nmse)
		}
	}
	// A demotion to the host would pass the NMSE above trivially.
	if n := second.GPULayers(); n != offered {
		t.Fatalf("the second session fell to the host after its tokens (%d of %d blocks left): %v",
			n, offered, g.Err())
	}
}

// TestHybridSecondSessionOnAFullCard is the gate on a device with no room left
// for another session. A second session that cannot get a KV cache must run on
// the host, not launch against nil buffers (NaN logits, no error). The budget is
// set from what one session actually used, so the second is squeezed by
// construction.
func TestHybridSecondSessionOnAFullCard(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{moe: true, layers: 8})
	defer m.Close()

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	host := m.NewState(16)
	defer host.Close()
	want := make([][]float32, len(ids))
	for i, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want[i] = append([]float32(nil), l...)
	}

	// What one session costs, measured rather than guessed.
	probe, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || probe == nil {
		noDevice(t, "device", err)
	}
	warm := m.NewState(16)
	warm.SetDeviceLayers(probe, 8)
	placed := warm.GPULayers()
	used := probe.Bytes()
	warm.Close()
	probe.Close()
	if placed == 0 || used == 0 {
		t.Skipf("the device took no block of this fixture: %d placed, %d bytes", placed, used)
	}

	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithBudget(used),
		tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	first := m.NewState(16)
	first.SetDeviceLayers(g, 8)
	if first.GPULayers() == 0 {
		first.Close()
		t.Skipf("the budget of %d bytes held no block: %s", used, g.Err())
	}
	second := m.NewState(16)
	defer second.Close()
	second.SetDeviceLayers(g, 8)
	t.Logf("budget %d B: first session placed %d block(s), second placed %d",
		used, first.GPULayers(), second.GPULayers())
	first.Close()

	for p, id := range ids {
		got, err := second.Forward(id)
		if err != nil {
			// A refusal is a correct outcome here; silent NaN is not.
			t.Fatalf("second session pos %d: %v", p, err)
		}
		var num, den float64
		for i := range want[p] {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("second session pos %d: logit %d is %g -- the block ran "+
					"without a KV cache of its own", p, i, got[i])
			}
			d := float64(got[i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		if nmse := num / den; nmse > 5e-3 {
			t.Fatalf("second session pos %d: logit NMSE %.3e", p, nmse)
		}
	}
}

// TestHybridDeviceErrorGrowsWithPosition is the other half of the amplification
// control: the same perturbation compounding through positions. A linear
// block's state is carried forward, so the seam's rounding accumulates rather
// than being re-seeded each token. It prints the trajectory and asserts only
// that the arm stays finite: accumulated rounding moves an argmax, it does not
// reach Inf.
func TestHybridDeviceErrorGrowsWithPosition(t *testing.T) {
	// One block and two: a second seam is a second source of error.
	t.Run("1-block", func(t *testing.T) {
		hybridGrowth(t, hyOpt{moe: true, deep: true, solo: 4}, 1)
	})
	t.Run("2-block", func(t *testing.T) {
		hybridGrowth(t, hyOpt{moe: true, deep: true}, 2)
	})
}

func hybridGrowth(t *testing.T, o hyOpt, want int) {
	// Pinned: see hybridDeviceNMSEOn.
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	m := hybridModelOpt(t, o)
	defer m.Close()
	host := m.NewState(48)
	defer host.Close()
	dev := m.NewState(48)
	defer dev.Close()
	// Offer exactly what this gate wants rather than trusting a budget to refuse
	// the rest: the seam is a contiguous prefix, so the placement does not
	// depend on a card's spare bytes.
	dev.SetDeviceLayers(g, want)
	if got := dev.GPULayers(); got != want {
		t.Skipf("this card holds %d of the %d block(s) this gate needs: %s",
			got, want, g.Err())
	}

	// The same token to both arms at every position, as `jitllm verify -n` does:
	// free-running chains would diverge on the first argmax flip.
	tok, nonFinite := int32(1), -1
	for p := 0; p < 24; p++ {
		w, err := host.Forward(tok)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want := append([]float32(nil), w...)
		got, err := dev.Forward(tok)
		if err != nil {
			t.Fatalf("device: %v", err)
		}
		var num, den float64
		bad := false
		for i := range want {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				bad = true
				break
			}
			d := float64(got[i] - want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		if bad {
			if nonFinite < 0 {
				nonFinite = p
			}
			t.Logf("pos %2d: NON-FINITE on the device", p)
		} else {
			t.Logf("pos %2d: logit NMSE %.3e", p, num/den)
		}
		best, bi := want[0], int32(0)
		for i, v := range want {
			if v > best {
				best, bi = v, int32(i)
			}
		}
		tok = bi
	}
	if nonFinite >= 0 {
		t.Fatalf("the device arm went non-finite at position %d with %d block(s) "+
			"placed: accumulating rounding moves an argmax, it does not reach "+
			"Inf", nonFinite, want)
	}
}

// TestHybridSecondSessionMatchesTheFirst compares a second device session
// against the first one rather than against the host: a deep seam moves the
// logits far more than a session bug needs to, so the host is the wrong
// reference. One session runs exactly this arithmetic, so the answer has to be
// bit equality. Two blocks with forty host blocks below is Qwen3-Next-80B's
// arrangement on a 4 GiB card.
func TestHybridSecondSessionMatchesTheFirst(t *testing.T) {
	m := hybridModelOpt(t, hyOpt{moe: true, deep: true})
	defer m.Close()
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	run := func(sessions int) [][]float32 {
		// The tuner is pinned: the k-split is the f32 reduction order, the one
		// knob this equality depends on.
		g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		if sessions > 1 {
			// A session that exists before the one under test, as benchRun is
			// to forcedDiff.
			warm := m.NewState(16)
			warm.SetDeviceLayers(g, 2)
			if warm.GPULayers() != 2 {
				n := warm.GPULayers()
				warm.Close()
				t.Fatalf("wanted two blocks, got %d: %v", n, g.Err())
			}
			for _, id := range ids {
				if _, err := warm.Forward(id); err != nil {
					t.Fatalf("warm session: %v", err)
				}
			}
			warm.Close()
		}
		s := m.NewState(16)
		defer s.Close()
		// Two, asked for by count.
		s.SetDeviceLayers(g, 2)
		if s.GPULayers() != 2 {
			t.Fatalf("wanted two blocks, got %d: %v", s.GPULayers(), g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatalf("session %d: %v", sessions, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		// Still on the device, or the arms are two tiers: a demotion to the host
		// could match to the ulp by accident.
		if s.GPULayers() != 2 {
			t.Fatalf("session %d fell to the host after its tokens (%d blocks left): %v",
				sessions, s.GPULayers(), g.Err())
		}
		return out
	}

	alone, second := run(1), run(2)
	for p := range alone {
		for i := range alone[p] {
			if math.IsNaN(float64(second[p][i])) || math.IsInf(float64(second[p][i]), 0) {
				t.Fatalf("pos %d: logit %d is %g on the second session", p, i, second[p][i])
			}
			if alone[p][i] != second[p][i] {
				t.Fatalf("pos %d: logit %d is %g on the second session and %g on the "+
					"first: same tier, same blocks, same arithmetic -- a second "+
					"session is disturbing the first's",
					p, i, second[p][i], alone[p][i])
			}
		}
	}
}

// TestMoEDeviceUploadReadsTheWholeBank is the gate on what a device is handed.
// pageIn reads a block without its routed experts on purpose, but PrepLayer
// uploads the whole bank once, so whatever the frame held then is what the card
// runs: zeros on a fresh frame (fluent, wrong), another block's bytes on a
// reused one (NaN). So the gate runs a host decode first under a page budget
// tight enough to evict, and attaches the device afterwards.
func TestMoEDeviceUploadReadsTheWholeBank(t *testing.T) {
	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	// Three frames for eight blocks, sized from the container's own page: a
	// budget that holds every block allocates no pool and never evicts.
	probe := hybridModelOpt(t, hyOpt{moe: true, layers: 8, manyExp: true})
	page := probe.container.H.PageSize
	probe.Close()
	m := hybridModelOpt(t, hyOpt{moe: true, layers: 8, manyExp: true, budget: 3 * page})
	defer m.Close()
	t.Logf("page %d bytes (%.2f MiB), chunk %d", page, float64(page)/(1<<20), jlm.Chunk)
	if b := m.container.Budget(); b == 0 || b >= 8*page {
		t.Skipf("the budget is %d bytes against 8 pages of %d, so nothing is evicted", b, page)
	}

	// Two tokens, not eight: residency is tracked in 1 MiB chunks, so the bank is
	// only partly read while expert selections are far fewer than chunks. This
	// page is ~7 chunks and one expert runs per token.
	cycle := m.NewState(16)
	for _, id := range []int32{1, 2} {
		if _, err := cycle.Forward(id); err != nil {
			t.Fatalf("host: %v", err)
		}
	}
	resident, in, out := m.PageStats()
	cycle.Close()
	t.Logf("after the host decode: %d page(s) resident, %d in, %d out", resident, in, out)
	if out == 0 {
		t.Skip("no page was evicted, so no frame was reused and this proves nothing")
	}

	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	want := make([][]float32, len(ids))
	host := m.NewState(16)
	for i, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want[i] = append([]float32(nil), l...)
	}
	host.Close()

	// The device attaches after that decode; one attached first reads every
	// frame fresh and cannot show this.
	dev := m.NewState(16)
	defer dev.Close()
	dev.SetDeviceLayers(g, 8)
	if dev.GPULayers() == 0 {
		t.Skipf("the device took no block of this fixture: %s", g.Err())
	}
	t.Logf("%d block(s) placed after a host decode", dev.GPULayers())
	for p, id := range ids {
		got, err := dev.Forward(id)
		if err != nil {
			t.Fatalf("device pos %d: %v", p, err)
		}
		var num, den float64
		for i := range want[p] {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("pos %d: logit %d is %g -- the expert bank the device "+
					"uploaded is not the model's", p, i, got[i])
			}
			d := float64(got[i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		if nmse := num / den; nmse > 5e-3 {
			t.Fatalf("pos %d: logit NMSE %.3e -- the expert bank the device "+
				"uploaded is not the model's", p, nmse)
		}
	}
}

// TestMoEDeviceReadsOnlyTheBanksItTakes is the other half of
// TestMoEDeviceUploadReadsTheWholeBank: the bank is read only for a block the
// device actually admits, not for every block offered (a mixture's bank is most
// of its page). Self-calibrating: under a budget admitting one block of two the
// read must be strictly smaller than under one admitting both; if it followed
// the offer the two would be equal.
func TestMoEDeviceReadsOnlyTheBanksItTakes(t *testing.T) {
	opt := hyOpt{moe: true, layers: 8, manyExp: true}
	// The prompt's scratch is not reserved on either side: its width is halved
	// until it fits beside the first block (scratch.go), so a budget measured
	// on an open card holds a wider reservation than the same budget buys, and
	// the bytes it frees admit blocks the measurement did not count -- the
	// budgets measured for one and two blocks placed three and four.
	noReserve := tier.WithConfig(func(c *tier.Config) { c.NoScratchReserve = true })
	// Measure what n blocks cost rather than relying on what the default budget
	// happens to admit.
	cost := func(n int) (int, uint64) {
		g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff), noReserve)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		m := hybridModelOpt(t, opt)
		defer m.Close()
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, n)
		return s.GPULayers(), g.Bytes()
	}
	nOneP, bOne := cost(1)
	nTwoP, used := cost(2)
	if nOneP != 1 || nTwoP != 2 || bOne == 0 || used <= bOne {
		t.Skipf("this card places %d and %d block(s) holding %d and %d bytes, so "+
			"one and two blocks cannot be separated here", nOneP, nTwoP, bOne, used)
	}

	// chunks reports what SetDeviceLayers read off the file under `budget`
	// bytes of card, and how many blocks it placed.
	chunks := func(budget uint64) (int64, int) {
		m := hybridModelOpt(t, opt)
		defer m.Close()
		g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithBudget(budget),
			tier.WithDeviceTune(tier.TuneOff), noReserve)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		before := m.container.ChunksIn()
		s.SetDeviceLayers(g, 8)
		return m.container.ChunksIn() - before, s.GPULayers()
	}

	two, nTwo := chunks(used)
	one, nOne := chunks(bOne)
	t.Logf("budget %d B: %d block(s), %d chunk(s) read", used, nTwo, two)
	t.Logf("budget %d B: %d block(s), %d chunk(s) read", bOne, nOne, one)
	if nTwo != 2 || nOne != 1 {
		t.Skipf("wanted 2 and 1 blocks placed, got %d and %d: the budgets do not "+
			"separate the two cases on this device", nTwo, nOne)
	}
	if one >= two {
		t.Fatalf("placing one block of two read %d chunk(s) against %d for both: "+
			"the read follows the OFFER rather than the admission, so every block "+
			"the budget refuses still costs a whole expert bank", one, two)
	}
}

// TestDevicePagingSurvivesTheHostPager is the gate on a block being uploaded
// more than once. The tier's own paging gates run on fake buffers, which copy
// wrong host bytes faithfully; against a real backend a page-in once read a host
// page the host pager had already evicted and reused (CUDA faulted, Vulkan
// produced garbage silently).
//
// Both pagers have to be live: the host page budget is three pages of eight
// blocks, and manyExp puts a routed bank in each page so a re-read has
// something to get wrong.
func TestDevicePagingSurvivesTheHostPager(t *testing.T) {
	// Dense and MoE: the refresh concerns the host page, which every block has,
	// not only a mixture's routed bank.
	for _, c := range []struct {
		name string
		opt  hyOpt
	}{
		{"dense", hyOpt{layers: 8, budget: 3 << 20}},
		{"moe", hyOpt{moe: true, layers: 8, manyExp: true, budget: 3 << 20}},
	} {
		t.Run(c.name, func(t *testing.T) { devicePagingUnderHostPager(t, c.opt) })
	}
}

func devicePagingUnderHostPager(t *testing.T, o hyOpt) {
	m := hybridModelOpt(t, o)
	defer m.Close()
	if b := m.container.Budget(); b == 0 {
		t.Skip("the host page budget did not apply, so only one pager is live")
	}
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	host := m.NewState(16)
	defer host.Close()
	want := make([][]float32, len(ids))
	for i, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatalf("host: %v", err)
		}
		want[i] = append([]float32(nil), l...)
	}

	g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	for li := 0; li < m.Cfg.NLayer; li++ {
		g.Stream(li, true)
	}
	dev := m.NewState(16)
	defer dev.Close()
	dev.SetDeviceLayers(g, m.Cfg.NLayer)
	if dev.GPULayers() == 0 {
		t.Skipf("the device took no block: %s", g.Err())
	}
	// A slot is what is left after the scratch and KV cache, which on the tiny
	// dense fixture are most of the card: its eight blocks are 33 KB beside a
	// 3.3 MB scratch on Metal, so no flat fraction of the card leaves one slot
	// and not none. Search for the tightest budget that still leaves one: the
	// slot count only falls as the budget does.
	full := g.Bytes()
	g.SetBudget(full)
	if g.Stats().Slots < 1 {
		t.Fatalf("the budget the placement charged (%d bytes) leaves no slot: %s", full, g.Err())
	}
	lo, chosen := uint64(0), full // no slot at lo, one at chosen
	for chosen-lo > 1 {
		b := lo + (chosen-lo)/2
		g.SetBudget(b)
		if g.Stats().Slots >= 1 {
			chosen = b
		} else {
			lo = b
		}
	}
	g.SetBudget(chosen)

	for p, id := range ids {
		l, err := dev.Forward(id)
		if err != nil {
			t.Fatalf("pos %d: %v", p, err)
		}
		var num, den float64
		for i := range want[p] {
			if math.IsNaN(float64(l[i])) || math.IsInf(float64(l[i]), 0) {
				t.Fatalf("pos %d: logit %d is %g, not finite", p, i, l[i])
			}
			d := float64(l[i] - want[p][i])
			num += d * d
			den += float64(want[p][i]) * float64(want[p][i])
		}
		if nmse := num / den; nmse > hybridDeviceBound {
			t.Fatalf("pos %d: logit NMSE %.3e after %d page-in(s): the pager "+
				"uploaded bytes that are not this block's", p, nmse, g.Stats().PageIns)
		}
	}
	st := g.Stats()
	// The configuration has to have been selected. PageIns > 0 is not enough: a
	// failed device demotes to the host, which computes the right logits. What
	// separates them is blocks executed on the device, NLayer per position.
	if want := int(m.Cfg.NLayer) * len(ids); st.Blocks != want {
		t.Fatalf("%d block-run(s) on the device over %d positions, want %d: the "+
			"blocks were demoted to the host, so the logits above are the "+
			"host's and prove nothing (%d page-in(s), %d page-out(s)): %s",
			st.Blocks, len(ids), want, st.PageIns, st.PageOuts, g.Err())
	}
	if st.PageIns == 0 {
		t.Fatalf("%d slot(s) and ZERO page-ins: nothing swapped, so this gate "+
			"proved nothing", st.Slots)
	}
	t.Logf("%d slot(s), %d page-in(s), %d page-out(s) with the host pager also "+
		"evicting: every position within %.0e", st.Slots, st.PageIns, st.PageOuts,
		hybridDeviceBound)
}

// TestPlacedBlockSurvivesAKVCacheRoundTrip is the gate on a block's history
// crossing the seam in both directions, for both kinds of block. A placed
// linear block keeps its summary in the tier's recPair, so the KV store must
// migrate it home (MigrateRec, the twin of MigrateKV) or it stores a summary
// that never saw the prompt. The attention arm is the control.
//
// It generates eight tokens because a wrong history agrees on the first token
// and parts on the next. It opens and closes one tier per run: several tiers
// sharing the card at once move the tokens on their own.
func TestPlacedBlockSurvivesAKVCacheRoundTrip(t *testing.T) {
	prompt := make([]int32, 0, 24)
	for i := 0; i < 24; i++ {
		prompt = append(prompt, int32(1+i%14))
	}
	const gen = 8
	for _, c := range []struct {
		name string
		opt  hyOpt
	}{
		// solo=1 puts the one attention block at index 0, so offering one
		// block places attention and the recurrence never crosses the seam.
		{"attention-on-device", hyOpt{moe: true, layers: 8, solo: 1}},
		{"linear-on-device", hyOpt{moe: true, layers: 8}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := hybridModelOpt(t, c.opt)
			defer m.Close()
			fs, err := NewFileStore(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			run := func(cached bool) ([]int32, int) {
				g, err := tier.OpenWith(tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || g == nil {
					noDevice(t, "device", err)
				}
				defer g.Close()
				s := m.NewState(64)
				defer s.Close()
				s.SetDeviceLayers(g, 1)
				if s.GPULayers() != 1 {
					t.Skipf("wanted one placed block, got %d: %s", s.GPULayers(), g.Err())
				}
				if cached {
					s.SetKVStore(fs)
					mustKey(t, s, "seam/"+c.name)
				}
				var lg []float32
				if cached {
					lg, err = s.PrefillCached(prompt)
				} else {
					lg, err = s.Prefill(prompt)
				}
				if err != nil {
					t.Fatal(err)
				}
				out := make([]int32, 0, gen)
				for i := 0; i < gen; i++ {
					id := int32(argmax(lg))
					out = append(out, id)
					if lg, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				return out, s.KVRestored()
			}
			want, _ := run(false)
			if _, n := run(true); n != 0 {
				t.Fatalf("the sealing pass restored %d positions from an empty store", n)
			}
			got, reused := run(true)
			// The cached arm has to have used the cache.
			if reused == 0 {
				t.Fatalf("the cached pass restored nothing, so this gate compared "+
					"two ordinary runs (%d-token prompt)", len(prompt))
			}
			sameTokens(t, c.name, want, got)
			t.Logf("%d of %d positions restored with a block placed; tokens identical",
				reused, len(prompt))
		})
	}
}
