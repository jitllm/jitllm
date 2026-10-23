package jit

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

func keyRequest() Request {
	return Request{
		Graph:   graph.MatVec(jlm.RoleOutput, 64, 256),
		Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jlm.TypeQ4},
		Step:    Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
		Device:  Device{Kind: DeviceGPU, Slots: 32, Lanes: 32},
	}
}

// TestFormIsCachedByKey holds the composer's one cache: a request formed
// twice is formed once and keyed alike, a change to anything it is formed
// from moves its Key, and a hit whose words differ is an error.
func TestFormIsCachedByKey(t *testing.T) {
	r := Default()
	k1, c1, err := r.Form(keyRequest())
	if err != nil {
		t.Fatal(err)
	}
	k2, c2, err := r.Form(keyRequest())
	if err != nil {
		t.Fatal(err)
	}
	if k1 != k2 || len(c1) == 0 || &c1[0] != &c2[0] {
		t.Fatalf("one request formed twice: keys %#x %#x, the second not the cached codes", k1, k2)
	}
	if n := len(r.cache.m); n != 1 {
		t.Fatalf("%d cache entries after one request twice", n)
	}

	moved := map[string]func(*Request, *Registry){
		"a step field": func(q *Request, _ *Registry) { q.Step.ActWindow = 64 },
		"a format":     func(q *Request, _ *Registry) { q.Formats[jlm.RoleOutput] = jlm.TypeQ8 },
		"an attribute": func(q *Request, _ *Registry) { q.Graph.Ops[0].Attr.Eps = 1e-6 },
		"the device":   func(q *Request, _ *Registry) { q.Device.Lanes = 64 },
		"the int8 form": func(_ *Request, x *Registry) {
			p := x.Profiles[DeviceGPU]
			p.Int8 = nil
			x.Profiles[DeviceGPU] = p
		},
		"the lowering": func(_ *Request, x *Registry) { x.Lowerings[DeviceGPU] = gpuLowering{Poison: true} },
	}
	for what, change := range moved {
		q, x := keyRequest(), Default()
		change(&q, x)
		k, _, err := x.keyOf(q)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if k == k1 {
			t.Errorf("changing %s left the key at %#x", what, k)
		}
	}

	// A forged entry under the request's key with other words.
	f := r.cache.m[k1]
	f.words = append([]uint64{1}, f.words...)
	r.cache.m[k1] = f
	if _, _, err := r.Form(keyRequest()); err == nil || !strings.Contains(err.Error(), "share the key") {
		t.Fatalf("a hit with other words gave %v, want the collision refused", err)
	}

	// A request that does not form is remembered: the second answer is the
	// cached error, with nothing formed again.
	bad := keyRequest()
	bad.Graph.Ops[0].Kind = graph.OpInvalid
	r2 := Default()
	_, _, e1 := r2.Form(bad)
	_, _, e2 := r2.Form(bad)
	if e1 == nil || e2 != e1 {
		t.Fatalf("an unformable request: %v, then %v; want one error, remembered", e1, e2)
	}

	// A registry with no cache forms every request afresh.
	plain := *Default()
	plain.cache = nil
	_, a, err := plain.Form(keyRequest())
	if err != nil {
		t.Fatal(err)
	}
	_, b, _ := plain.Form(keyRequest())
	if &a[0] == &b[0] {
		t.Fatal("a registry with no cache returned the same codes twice")
	}
}
