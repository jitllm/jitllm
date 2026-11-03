//go:build amd64

package cpu

import (
	"reflect"
	"strings"
	"testing"
)

// TestSSEProbesCoverEveryField: SSEPending asks one probe per table field,
// and a field with no probe would be a pending op Baseline cannot see -- the
// engine passes its floor and then panics naming the kernel mid-Open. Every
// function field that returns an error must have exactly one probe, and every
// probe must name a real field.
func TestSSEProbesCoverEveryField(t *testing.T) {
	typ := reflect.TypeOf(Emitters{})
	errType := reflect.TypeOf((*error)(nil)).Elem()
	fields := map[string]bool{}
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if f.Type.Kind() == reflect.Func && f.Type.NumOut() > 0 && f.Type.Out(f.Type.NumOut()-1) == errType {
			fields[f.Name] = true
		}
	}
	if len(fields) < 25 {
		t.Fatalf("found %d error-returning fields; the reflection walked the wrong thing", len(fields))
	}
	probed := map[string]int{}
	for _, p := range sseProbes() {
		probed[p.field]++
		if !fields[p.field] {
			t.Errorf("probe %q names no error-returning field of Emitters", p.field)
		}
	}
	for f := range fields {
		if probed[f] != 1 {
			t.Errorf("Emitters.%s has %d probes in sseProbes, want exactly 1", f, probed[f])
		}
	}
}

// TestSSEProbesReportAPanickingEmitter: an emitter that panics at an engine
// shape is as absent as a stub, and Baseline must say which one rather than
// take the process down.
func TestSSEProbesReportAPanickingEmitter(t *testing.T) {
	em := EmittersFor(TierSSE)
	em.Conv1d = func(int, int) ([]byte, error) { panic("synthetic emitter bug") }
	var hit string
	for _, p := range ssePendingIn(em) {
		if strings.HasPrefix(p, "Conv1d") {
			hit = p
		}
	}
	if !strings.Contains(hit, "panicked: synthetic emitter bug") {
		t.Fatalf("a panicking Conv1d emitter reads as %q in SSEPending", hit)
	}
}

// TestTierReportSaysWhatARunDoes is `jitllm hardware`'s tier line on the three
// amd64 hosts: the tier the probe chose, the extensions it chose from, and --
// on the SSE tier -- every op still missing, or that none is.
func TestTierReportSaysWhatARunDoes(t *testing.T) {
	if HostTier() == TierAVX2 {
		if r := TierReport(); !strings.HasPrefix(r, "avx2 ") {
			t.Errorf("an AVX2 host reports %q", r)
		}
		if l := HostDotLabel(); l != HostDotKind().String() {
			t.Errorf("an AVX2 host's dot label is %q, want HostDotKind's %q", l, HostDotKind())
		}
	}
	t.Run("sse", func(t *testing.T) {
		withTier(t, TierSSE)
		r := TierReport()
		if !strings.HasPrefix(r, "sse ") || !strings.Contains(r, "no usable avx2, fma, f16c") {
			t.Errorf("an SSE host reports %q", r)
		}
		if p := SSEPending(); len(p) > 0 {
			for _, op := range p {
				if !strings.Contains(r, op) {
					t.Errorf("the SSE report does not name pending op %s: %q", op, r)
				}
			}
		} else if !strings.Contains(r, "every op is generated") {
			t.Errorf("a complete SSE tier reports %q", r)
		}
		if l := HostDotLabel(); l != "sse" {
			t.Errorf("an SSE host's dot label is %q: HostDotKind's %q is a VEX sequence it cannot run", l, HostDotKind())
		}
	})
	t.Run("below the floor", func(t *testing.T) {
		withTier(t, TierNone)
		if r := TierReport(); !strings.HasPrefix(r, "none ") || !strings.Contains(r, "missing ssse3, sse4.1") {
			t.Errorf("a below-floor host reports %q", r)
		}
		if l := HostDotLabel(); l != "none" {
			t.Errorf("a below-floor host's dot label is %q", l)
		}
	})
}
