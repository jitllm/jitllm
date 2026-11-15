package graph

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

// TestStoredGraphReadsBack stores a block's graph and reads it back as
// another block's: the same graph, the same encoding, and each corruption
// refused.
func TestStoredGraphReadsBack(t *testing.T) {
	g, f := block(8, 16)
	g.Routed = Region{Lo: 1, Hi: 3}
	b, err := Encode(g)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Decode(b, 5, f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Block != 5 {
		t.Fatalf("read back as block %d, want 5", got.Block)
	}
	g.Block = 5
	if got.String() != g.String() || got.Routed != g.Routed {
		t.Fatalf("read back\n%s\nwant\n%s", got, g)
	}
	again, err := Encode(got)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(again, b) {
		t.Fatal("a graph read back encodes to other bytes")
	}
	for _, c := range []struct {
		name, want string
		b          []byte
		f          Facts
	}{
		{"truncated", "stored graph", b[:len(b)/2], f},
		{"an op this build does not know", "does not know", encodeRenamed(t, g, OpMatVec, "matvex"), f},
		{"a tensor the container does not store", "does not store", b, Facts{}},
	} {
		if _, err := Decode(c.b, 0, c.f); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error naming %q", c.name, err, c.want)
		}
	}
	far := *g
	far.Routed = Region{Lo: 2, Hi: len(g.Ops) + 1}
	fb, err := Encode(&far)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(fb, 0, f); err == nil || !strings.Contains(err.Error(), "routed") {
		t.Errorf("routed ops past the graph: %v", err)
	}
}

// encodeRenamed is g stored with kind k under another name.
func encodeRenamed(t *testing.T, g *Graph, k OpKind, name string) []byte {
	t.Helper()
	b, err := Encode(g)
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.Replace(b, []byte(k.String()), []byte(name), 1)
	if bytes.Equal(r, b) || len(name) != len(k.String()) {
		t.Fatalf("no %q in the encoding to rename", k)
	}
	return r
}

// TestEveryOpKindHasItsOwnName: a stored op is its kind's name, so each
// kind's must be its own.
func TestEveryOpKindHasItsOwnName(t *testing.T) {
	seen := map[string]OpKind{}
	for k := OpInvalid + 1; k < numOps; k++ {
		n := k.String()
		if strings.HasPrefix(n, "op") && strings.TrimLeft(n[2:], "0123456789") == "" {
			t.Errorf("op kind %d has no name", k)
		}
		if o, ok := seen[n]; ok {
			t.Errorf("op kinds %d and %d share the name %q", o, k, n)
		}
		seen[n] = k
		if KindNamed(n) != k {
			t.Errorf("%q reads back as %d, want %d", n, KindNamed(n), k)
		}
	}
}

// storedFields are the field names a stored graph's types held when the
// encoding was set. A stored graph reads a field by its name, and one whose
// name is gone reads as zero, silently; so a name here never disappears
// without a new Version. A field added is safe (zero is what every graph
// stored before it meant) and joins this list.
func storedFields() map[string][]string {
	return map[string][]string{
		"storedGraph": {"Version", "Values", "Ops", "In", "Out", "Routed"},
		"storedOp":    {"Kind", "In", "Out", "Weight", "Reads", "Writes", "Attr"},
		"Value":       {"Name", "Width", "Elem", "Rows"},
		"Binding":     {"Role", "K", "Rows", "Sheets", "From"},
		"StateRef":    {"State", "From", "Version", "Hist"},
		"Region":      {"Lo", "Hi"},
		"Term":        {"Fn", "Args"},
		"Arg":         {"Kind", "Index", "Elem", "Const"},
	}
}

// attrsAtEncoding are Attrs' fields when the encoding was set.
func attrsAtEncoding() []string {
	return []string{
		"Eps", "HeadDim", "NRot", "Base", "Layout", "Freqs", "Sections", "Interleaved", "Cycle", "CycleAxes",
		"Mscale", "NHead", "NKVHead", "Scale", "Window", "Chunked", "Bidir", "Softcap", "Expr", "Experts", "TopK",
		"Alpha", "Taps", "Dil", "GridW", "Stride", "Positional", "Request", "Pooling", "Block", "BF16", "Rotary",
		"Grid", "Chain", "Filter", "Tiled", "Additive", "RowWindows", "Bound", "Offset", "Width", "Origin", "Local",
		"Pooled", "Firsts", "Inverse", "Keeps", "Fn", "Order", "Span", "Const", "Words", "Reset", "Exact", "Sound",
		"Pixels", "PicW", "PicH", "PicArea", "PicSide", "PicLive", "PicCanvas", "PicFill", "PicTorch", "PicBytes",
		"PicFrame", "PicPass", "PicColumns", "PicBF16", "Mean", "Std",
	}
}

// TestStoredFieldsOnlyGrow holds each stored type to the names it had, and
// Attrs to every field it has now: a name renamed or removed fails here
// before a container reads it as zero.
func TestStoredFieldsOnlyGrow(t *testing.T) {
	for _, v := range []any{storedGraph{}, storedOp{}, Value{}, Binding{}, StateRef{}, Region{}, Term{}, Arg{}} {
		ty := reflect.TypeOf(v)
		have := map[string]bool{}
		for i := range ty.NumField() {
			have[ty.Field(i).Name] = true
		}
		for _, n := range storedFields()[ty.Name()] {
			if !have[n] {
				t.Errorf("%s.%s is gone: a stored graph reads it as zero; bump Version", ty.Name(), n)
			}
		}
		if len(have) != len(storedFields()[ty.Name()]) {
			t.Errorf("%s has %d fields, the list %d: add the new one to storedFields", ty.Name(), len(have), len(storedFields()[ty.Name()]))
		}
	}
	at := reflect.TypeOf(Attrs{})
	for i := range at.NumField() {
		if !at.Field(i).IsExported() {
			t.Errorf("Attrs.%s is unexported, so a stored graph drops it", at.Field(i).Name)
		}
	}
	if at.NumField() < len(attrsAtEncoding()) {
		t.Errorf("Attrs has %d fields, fewer than the %d it had", at.NumField(), len(attrsAtEncoding()))
	}
	for _, n := range attrsAtEncoding() {
		if _, ok := at.FieldByName(n); !ok {
			t.Errorf("Attrs.%s is gone: a stored graph reads it as zero; bump Version", n)
		}
	}
}
