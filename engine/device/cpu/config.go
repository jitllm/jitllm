package cpu

// Config is the emitter's knobs. Zero is the shipping default, and every field
// is a measurement instrument rather than a tuning dial.
//
// The package reads no environment: an embedded caller must be able to set
// these, and a stray export must not silently change the generated code.
// It is package-level because the emitters are package-level functions taking
// a Spec, and Spec is capped (RULE 6). What remains here is process-wide by
// nature: a profiler's perf map and the arm64 measurement probes, set by
// cmd/jitllm. The prefetch distances are per JIT (EmitOpts, EmittersWith).
type Config struct {
	// Pack pins the row-interleave width for every quant type; 0 takes the
	// measured per-type default from BestPack.
	Pack int
	// Accs pins the number of independent accumulator chains (1..4); 0 takes
	// the per-type default from BestAccs.
	Accs int
	// PerfMap writes /tmp/perf-<pid>.map so a profiler can name generated code.
	PerfMap bool
	// ScalarProbe, LoadProbe and VectorProbe are arm64 measurement scaffolding:
	// they inject N extra scalar / load / vector operations into the kernel so
	// one can be priced against the others. 0 disables. Never set outside a
	// measurement -- they make the kernel deliberately slower.
	ScalarProbe, LoadProbe, VectorProbe int
}

var cfg Config

// Configure sets the measurement knobs -- PerfMap and the three arm64 probes. It leaves Pack and Accs alone, which SetWidths owns; two setters over
// one struct keep either caller from clobbering the other.
func Configure(c Config) {
	c.Pack, c.Accs = cfg.Pack, cfg.Accs
	cfg = c
}

// SetWidths pins the interleave width and the accumulator-chain count for the
// process-wide reporters BestPack and BestAccs, or takes the measured per-type
// defaults at 0.
//
// nn does not call it: a JIT carries its own pin (PackWidth, AccChains,
// PackWidthNative), so two models in one process can hold different widths
// and concurrent JIT construction does not race here. The remaining caller is
// `jitllm hardware`, which prints what a run would emit.
func SetWidths(pack, accs int) { cfg.Pack, cfg.Accs = pack, accs }
