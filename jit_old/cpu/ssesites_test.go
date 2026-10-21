//go:build amd64

package cpu

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestEveryMatvecEmitterHasAnSSEAnswer is TestEveryVPDPBUSDSiteIsCoveredOrRefused
// one tier down: every file that emits the matvec family's arithmetic for the
// AVX2 tier -- an int8 dot (VPDPBUSD, VPMADDUBSW, or a dotEmitter built to emit
// one) or the float matvec -- must have a declared answer for a host with no
// AVX at all, and each answer is checked against the SSE table rather than
// trusted:
//
//	twin     the SSE table emits a declared SSE-tier kernel for it
//	refused  the SSE table refuses it by decision (not ErrNoSSEKernel)
//	declined the SSE table says it has no such family, so nn never builds it
//
// It greps the source so a new emitter with no SSE answer fails here rather
// than as a SIGILL (or an unexplained refusal) on an SSE host.
func TestEveryMatvecEmitterHasAnSSEAnswer(t *testing.T) {
	type answer struct {
		kind  string // twin, refused, declined
		check func(t *testing.T)
	}
	em := EmittersFor(TierSSE)
	packedTwin := answer{"twin", func(t *testing.T) {
		for _, g := range quant.PackedTypes {
			for _, emit := range []func() ([]byte, error){
				func() ([]byte, error) { return em.PackedMatVec(g, PackedRows) },
				func() ([]byte, error) { return em.PackedMatVec(g, PackedTail) },
				func() ([]byte, error) { return em.PackedFused(g) },
			} {
				code, err := emit()
				if err != nil || KernelTier(code) != TierSSE {
					t.Errorf("%s: the SSE table's packed kernel: %v (tier %v)", g, err, KernelTier(code))
				}
			}
		}
	}}
	refused := func(what string, emit func(quant.Type) error) answer {
		return answer{"refused", func(t *testing.T) {
			for _, g := range quant.PackedTypes {
				if err := emit(g); err == nil || errors.Is(err, ErrNoSSEKernel) {
					t.Errorf("%s %s: want a decided refusal, got %v", what, g, err)
				}
			}
		}}
	}
	declinedGGUF := answer{"declined", func(t *testing.T) {
		if em.RowMajorGGUF {
			t.Error("the SSE table claims the GGUF row-major machinery (GEMM, interleaved pack width)")
		}
		for _, g := range quant.PackedTypes {
			if em.RowMajorSupported(g) {
				t.Errorf("%s: the SSE table claims a quantized row-major kernel", g)
			}
		}
	}}
	answers := map[string]answer{
		// The dot sequences themselves: sseDot is their twin, and the violation
		// hook's twin is armed by the same switch.
		"prevnni.go":            packedTwin,
		"prevnni_hook_amd64.go": packedTwin,
		"packed.go":             packedTwin,
		"packedfused.go":        packedTwin,
		"packedtiled.go": refused("token-tiled", func(g quant.Type) error {
			_, err := em.PackedTiled(g, 2048, 512, 2)
			if em.MaxTiledTokens(g) != 0 {
				return nil
			}
			return err
		}),
		"packedwide.go": refused("wide", func(g quant.Type) error { _, err := em.PackedWide(g); return err }),
		"packedgemm.go": refused("weight-stationary GEMM", func(g quant.Type) error {
			_, err := em.PackedGEMM(g, 2048, 512, 256)
			return err
		}),
		// matvec.go carries both halves of the row-major family: the float
		// matvec (a twin, EmitRowMajorSSE) and the quantized GGUF kernels
		// (declined -- a container never decodes through them).
		"matvec.go": {"twin+declined", func(t *testing.T) {
			for _, ft := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
				code, err := em.RowMajor(Spec{W: ft, Rows: 1, Accs: 1, Cols: 1})
				if err != nil || KernelTier(code) != TierSSE {
					t.Errorf("%s: the SSE table's float matvec: %v", ft, err)
				}
			}
			declinedGGUF.check(t)
		}},
		"gemm.go":  declinedGGUF,
		"gemmk.go": declinedGGUF,
	}

	if _, err := os.Stat("prevnni.go"); err != nil {
		t.Skipf("NO PACKAGE SOURCE HERE (%v) -- this gate reads jit_old/cpu/*.go, which a "+
			"shipped test binary does not carry; it is not arch-dependent, run it in the repo", err)
	}
	site := regexp.MustCompile(`\.(VPDPBUSD|VPDPBUSDMem|VPMADDUBSW)\(|dotEmitter\{|^func emitFloatMatVec\(`)
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, e := range ents {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") || strings.HasPrefix(name, "sse") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(".", name))
		if err != nil {
			t.Fatal(err)
		}
		hit := false
		for _, line := range strings.Split(string(b), "\n") {
			if i := strings.Index(line, "//"); i >= 0 {
				line = line[:i]
			}
			hit = hit || site.MatchString(line)
		}
		if !hit {
			continue
		}
		seen[name] = true
		a, ok := answers[name]
		if !ok {
			t.Errorf("%s emits the matvec family's arithmetic for the AVX2 tier and has NO declared "+
				"SSE answer. Give it a twin in the SSE table, refuse it there by decision, or "+
				"decline its family -- and record which in this gate's map. Without one, an Atom "+
				"either SIGILLs or cannot run whatever reaches it.", name)
			continue
		}
		t.Run(name+"/"+a.kind, a.check)
	}
	for name := range answers {
		if !seen[name] {
			t.Errorf("the map declares %s but it emits none of the matvec family's arithmetic -- "+
				"delete the entry, or this gate covers a file that no longer needs it", name)
		}
	}
	if len(seen) == 0 {
		t.Fatal("found no matvec emitters at all; the regexp or the cwd is wrong and this gate proved nothing")
	}
}
