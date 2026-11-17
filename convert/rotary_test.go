package convert

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// llama3Freqs is llama3's per-pair factors as a GGUF of the model ships them,
// every step in float32 as torch tensors run it: freq = 1/base^(i/dim) over
// even i, then 1 where the wavelength is under orig/high, factor where it is
// over orig/low, and the smooth blend between. It is the Go transcription the
// converter computed them by before rotary.json stated them, kept as the
// statement's reference.
func llama3Freqs(headDim uint32, base, factor, low, high, orig float64) []float32 {
	dim := float32(headDim)
	lowWave, highWave := orig/low, orig/high
	twoPi := float32(2 * math.Pi)
	var out []float32
	for i := uint32(0); i < headDim; i += 2 {
		e := float32(float32(i) / dim)
		p := float32(math.Pow(float64(float32(base)), float64(e)))
		freq := float32(1 / p)
		wave := float32(twoPi / freq)
		switch {
		case float64(wave) < highWave:
			out = append(out, 1)
		case float64(wave) > lowWave:
			out = append(out, float32(factor))
		default:
			s := float32(float32(float32(float32(orig)/wave)-float32(low)) / float32(high-low))
			out = append(out, float32(1/float32(float32(float32(1-s)/float32(factor))+s)))
		}
	}
	return out
}

// yarnFreqs is YaRN's per-pair divisors as the engine computed them at load
// before rotary.json stated them, kept as the statement's reference: pair
// p's angle divided by 1/((1-e)/factor + e), e one below the correction
// range, zero above it and a linear ramp between, the range's ends
// nrot*ln(origCtx/(2*pi*b))/(2*ln(base)) at b = betaFast and betaSlow,
// floored and ceiled unless exact.
func yarnFreqs(nrot int, base, factor float64, origCtx int, betaFast, betaSlow float64, exact bool) []float32 {
	corr := func(b float64) float64 {
		return float64(nrot) * math.Log(float64(origCtx)/(2*math.Pi*b)) / (2 * math.Log(base))
	}
	lo, hi := corr(betaFast), corr(betaSlow)
	if !exact {
		lo, hi = math.Floor(lo), math.Ceil(hi)
	}
	lo, hi = math.Max(lo, 0), math.Min(hi, float64(nrot-1))
	if hi == lo {
		hi += 0.001
	}
	out := make([]float32, nrot/2)
	for p := range out {
		ramp := math.Min(math.Max((float64(p)-lo)/(hi-lo), 0), 1)
		e := 1 - ramp
		out[p] = float32(1 / ((1-e)/factor + e))
	}
	return out
}

func yarnParams(factor float32, orig uint32, fast, slow float32, truncate bool) map[string]any {
	return map[string]any{"factor": float64(factor), "original_context": float64(orig),
		"beta_fast": float64(fast), "beta_slow": float64(slow), "truncate": truncate}
}

// TestRotaryStatementsMatchTheirTranscription holds rotary.json's yarn,
// linear and llama3 statements to the Go each replaced, bit for bit, over
// the widths, bases, factors and contexts the shipped models state: a
// statement that rounds once differently moves a rotary table's bytes.
func TestRotaryStatementsMatchTheirTranscription(t *testing.T) {
	n := 0
	for _, nrot := range []uint32{16, 32, 64, 80, 96, 128, 256} {
		for _, base := range []float32{10000, 150000, 500000, 1e6, 5e6, 1e7, 75e6} {
			for _, factor := range []float32{1.5, 4, 8, 16, 32, 40, 64} {
				for _, orig := range []uint32{64, 2048, 4096, 8192, 32768, 65536, 262144} {
					for _, truncate := range []bool{true, false} {
						got, err := rotaryDivisors("yarn", int(nrot)/2, float64(nrot), float64(base),
							yarnParams(factor, orig, 32, 1, truncate))
						if err != nil {
							t.Fatal(err)
						}
						want := yarnFreqs(int(nrot), float64(base), float64(factor), int(orig), 32, 1, !truncate)
						if !slices.Equal(f32Bits(got), f32Bits(want)) {
							t.Fatalf("yarn nrot %d base %g factor %g orig %d truncate %v: %v, transcription %v",
								nrot, base, factor, orig, truncate, got, want)
						}
						n++
					}
					got, err := rotaryDivisors("llama3", int(nrot)/2, float64(nrot), float64(base),
						map[string]any{"factor": float64(factor), "low_freq_factor": 1.0, "high_freq_factor": 4.0,
							"original_context": float64(orig)})
					if err != nil {
						t.Fatal(err)
					}
					want := llama3Freqs(nrot, float64(base), float64(factor), 1, 4, float64(orig))
					if !slices.Equal(f32Bits(got), f32Bits(want)) {
						t.Fatalf("llama3 nrot %d base %g factor %g orig %d: %v, transcription %v", nrot, base, factor, orig, got, want)
					}
					n++
				}
				got, err := rotaryDivisors("linear", int(nrot)/2, float64(nrot), float64(base), map[string]any{"factor": float64(factor)})
				if err != nil {
					t.Fatal(err)
				}
				for p, d := range got {
					if d != factor {
						t.Fatalf("linear %g: pair %d divides by %g", factor, p, d)
					}
				}
			}
		}
	}
	t.Logf("%d shapes bit for bit", n)
}

func f32Bits(f []float32) []uint32 {
	out := make([]uint32, len(f))
	for i, x := range f {
		out[i] = math.Float32bits(x)
	}
	return out
}

// TestRotaryYarnMatchesTransformers holds the yarn statement to transformers'
// own _compute_yarn_parameters (scripts/yarngold.py) on gpt-oss's rotary and
// a second shape, with the correction range truncated and not: the inverse
// frequency base^(-2p/d)/divisor against transformers' inv_freq.
func TestRotaryYarnMatchesTransformers(t *testing.T) {
	raw, err := os.ReadFile("testdata/yarn.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		HeadDim  int `json:"head_dim"`
		Base     float64
		Factor   float64
		Orig     int
		Truncate bool
		InvFreq  []float64 `json:"inv_freq"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("no golden cases")
	}
	truncDiffers := false
	for _, c := range cases {
		params := func(truncate bool) map[string]any {
			return map[string]any{"factor": c.Factor, "original_context": float64(c.Orig), "beta_fast": 32.0,
				"beta_slow": 1.0, "truncate": truncate}
		}
		f, err := rotaryDivisors("yarn", c.HeadDim/2, float64(c.HeadDim), c.Base, params(c.Truncate))
		if err != nil {
			t.Fatal(err)
		}
		if len(f) != len(c.InvFreq) {
			t.Fatalf("%+v: %d pairs, golden has %d", c, len(f), len(c.InvFreq))
		}
		other, err := rotaryDivisors("yarn", c.HeadDim/2, float64(c.HeadDim), c.Base, params(!c.Truncate))
		if err != nil {
			t.Fatal(err)
		}
		for p, want := range c.InvFreq {
			got := math.Pow(c.Base, -2*float64(p)/float64(c.HeadDim)) / float64(f[p])
			if math.Abs(got-want) > 2e-6*want {
				t.Fatalf("d=%d truncate=%v pair %d: inv_freq %.9g, transformers %.9g", c.HeadDim, c.Truncate, p, got, want)
			}
			truncDiffers = truncDiffers || other[p] != f[p]
		}
	}
	// The control: truncating must change something, or the parameter is untested.
	if !truncDiffers {
		t.Fatal("truncated and exact ranges give identical frequencies -- the golden cannot tell them apart")
	}
}

// TestStateRotaryStatesThePairs states a YaRN and a linear factor over the
// factors a file ships: the container carries the product per pair, short
// and long, and no method; stating again changes nothing, and a method's
// statement that disagrees with its caller is refused.
func TestStateRotaryStatesThePairs(t *testing.T) {
	ones := func(role jlm.Role, v float32) jlm.Tensor {
		f := make([]float32, 8)
		for i := range f {
			f[i] = v + float32(i)
		}
		t := ropeFreqsTensor(f)
		t.Role = role
		return t
	}
	read := func(t *testing.T, tn jlm.Tensor) []float32 {
		t.Helper()
		f, err := floatsOf(tn.Type, tn.Data, int(tn.Dims[0]))
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	yarn := yarnFreqs(16, 10000, 8, 64, 32, 1, false)
	for _, r := range []struct {
		name   string
		cfg    jlm.Config
		have   []jlm.Tensor
		scaled []float32
	}{
		{"yarn alone", jlm.Config{NRot: 16, RopeBase: 10000, YarnFactor: 8, YarnOrigCtx: 64, YarnBetaFast: 32, YarnBetaSlow: 1}, nil, yarn},
		{"yarn over shipped factors", jlm.Config{NRot: 16, RopeBase: 10000, YarnFactor: 8, YarnOrigCtx: 64, YarnBetaFast: 32,
			YarnBetaSlow: 1}, []jlm.Tensor{ones(jlm.RoleRopeFreqs, 1)}, yarn},
		{"linear over short and long", jlm.Config{NRot: 16, RopeBase: 10000, RopeLinear: 8},
			[]jlm.Tensor{ones(jlm.RoleRopeFactorsShort, 1), ones(jlm.RoleRopeFactorsLong, 2)}, slices.Repeat([]float32{8}, 8)},
	} {
		t.Run(r.name, func(t *testing.T) {
			cfg := r.cfg
			s := &jlm.Source{Config: &cfg, Tensors: slices.Clone(r.have)}
			if err := StateRotary(&cfg, s); err != nil {
				t.Fatal(err)
			}
			if cfg.RopeLinear != 0 || cfg.YarnBetaFast != 0 || cfg.YarnBetaSlow != 0 || cfg.YarnFactor != 0 {
				t.Fatalf("the config still states a method: %+v", cfg)
			}
			if len(r.have) == 0 && (len(s.Tensors) != 1 || s.Tensors[0].Role != jlm.RoleRopeFreqs) {
				t.Fatalf("%d tensors, want rope_freqs alone", len(s.Tensors))
			}
			for i, tn := range s.Tensors {
				got := read(t, tn)
				for p := range got {
					shipped := float32(1)
					if len(r.have) > 0 {
						shipped = read(t, r.have[i])[p]
					}
					if want := r.scaled[p] * shipped; got[p] != want {
						t.Fatalf("%s pair %d: %g, want %g", tn.Role, p, got[p], want)
					}
				}
			}
			before := slices.Clone(s.Tensors[0].Data)
			if err := StateRotary(&cfg, s); err != nil || string(before) != string(s.Tensors[0].Data) {
				t.Fatalf("stating again: %v, or the factors moved", err)
			}
		})
	}
	cfg := jlm.Config{NRot: 16, RopeBase: 10000, YarnFactor: 8, YarnOrigCtx: 64, YarnBetaFast: 32}
	if err := StateRotary(&cfg, &jlm.Source{Config: &cfg}); err == nil || !strings.Contains(err.Error(), "correction range") {
		t.Errorf("a YaRN with no slow beta: %v", err)
	}
	if _, err := rotaryDivisors("yarn", 8, 16, 10000, map[string]any{"factor": 8.0}); err == nil {
		t.Error("a yarn given one parameter of five was evaluated")
	}
	if _, err := rotaryDivisors("linear", 8, 16, 10000, map[string]any{"factor": 8.0, "beta_fast": 32.0}); err == nil {
		t.Error("a linear given a parameter it does not read was evaluated")
	}
	if _, err := rotaryDivisors("ntk", 8, 16, 10000, nil); err == nil {
		t.Error("a method rotary.json does not state was evaluated")
	}
	short := ropeFreqsTensor(make([]float32, 4))
	cfg = jlm.Config{NRot: 16, RopeBase: 10000, RopeLinear: 2}
	if err := StateRotary(&cfg, &jlm.Source{Config: &cfg, Tensors: []jlm.Tensor{short}}); err == nil {
		t.Error("four shipped factors scaled over eight pairs")
	}
}

// TestRotaryCallsAreTheStatements evaluates a statement's calls and
// refuses what is not one.
func TestRotaryCallsAreTheStatements(t *testing.T) {
	env := func(p string) any { return map[string]any{"x": 2.5, "t": true}[p] }
	for src, want := range map[string]any{
		"floor(x) + ceil(x)":               5.0,
		"max(x, 3) - min(x, 3)":            0.5,
		"t ? log(x) : log(0 - x)":          math.Log(2.5),
		"f32(0.1) == 0.1":                  false,
		"sqrt(x * x) == x && floor(x) < x": true,
	} {
		n, err := parseExpr(src)
		if err != nil {
			t.Fatal(err)
		}
		got, err := evalRotary(n, env)
		if err != nil || got != want {
			t.Errorf("%s: %v (%v), want %v", src, got, err, want)
		}
	}
	for _, src := range []string{"tanh(x)", "max(x)", "log(0 - x)"} {
		n, err := parseExpr(src)
		if err != nil {
			t.Fatal(err)
		}
		if v, err := evalRotary(n, env); err == nil {
			t.Errorf("%s evaluated to %v", src, v)
		}
	}
}

// statedRopeFreqs is the per-pair rotary factors s states, rope_freqs.weight.
func statedRopeFreqs(t testing.TB, s *jlm.Source) []float32 {
	t.Helper()
	i := slices.IndexFunc(s.Tensors, func(tn jlm.Tensor) bool { return tn.Role == jlm.RoleRopeFreqs && tn.Block == jlm.DenseBlock })
	if i < 0 {
		t.Fatal("the source states no rope_freqs")
	}
	b, err := s.Tensors[i].Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := floatsOf(s.Tensors[i].Type, b, int(s.Tensors[i].Dims[0]))
	if err != nil {
		t.Fatal(err)
	}
	return f
}
