package jinja

import (
	"math"
	"strings"
	"testing"
)

// TestEvaluationIsBounded: a template or an expression that would loop or
// allocate without end is refused, by its steps or by the size it asks for,
// and a real one is not.
func TestEvaluationIsBounded(t *testing.T) {
	for _, c := range []struct{ name, src, want string }{
		{"a range past the bound", `{{ range(10000000000) | length }}`, "range"},
		{"a string repeated past the bound", `{{ "ab" * 100000000 }}`, "repeated"},
		{"loops past the budget", `{% for i in range(1000000) %}{% for j in range(1000000) %}{% endfor %}{% endfor %}`, "budget"},
	} {
		tp, err := Compile(c.src)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if _, err := tp.Render(nil); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error naming %q", c.name, err, c.want)
		}
	}
	tp, err := Compile(`{% for i in range(1000) %}{{ i }}{% endfor %}`)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := tp.Render(nil); err != nil || !strings.HasSuffix(out, "999") {
		t.Fatalf("a bounded loop: %q, %v", out, err)
	}
}

// TestExpressionsEvaluateToValues evaluates expressions over variables to
// their values, a call among them, and refuses one past its steps.
func TestExpressionsEvaluateToValues(t *testing.T) {
	called := NewCallable("stored", func(args []Value, _ map[string]Value) (Value, error) {
		return NewBool(len(args) == 1 && args[0].IsString() && args[0].AsString() == "attn_q.bias"), nil
	})
	vars := map[string]Value{"heads": NewInt(8), "head_dim": NewInt(64), "inf": NewFloat(math.Inf(1)), "stored": called}
	for _, c := range []struct {
		src  string
		want Value
	}{
		{"heads * head_dim", NewInt(512)},
		{"[2, heads // 4] | min", NewInt(2)},
		{"-inf", NewFloat(math.Inf(-1))},
		{"stored('attn_q.bias') and not stored('attn_k.bias')", NewBool(true)},
		{"'add' if heads > 4 else 'mul'", NewString("add")},
		{"1e-20", NewFloat(1e-20)},
	} {
		x, err := CompileExpr(c.src)
		if err != nil {
			t.Fatalf("%s: %v", c.src, err)
		}
		got, err := x.Eval(vars, 1000)
		if err != nil || !got.Equals(c.want) {
			t.Errorf("%s = %v (%v), want %v", c.src, got, err, c.want)
		}
	}
	x, err := CompileExpr("heads * head_dim + heads")
	if err != nil {
		t.Fatal(err)
	}
	if n := x.Names(); len(n) != 2 || n[0] != "heads" || n[1] != "head_dim" {
		t.Fatalf("names %v, want heads and head_dim once each", n)
	}
	if y, err := CompileExpr("[stored('a'), rows['b'] | default(c)] | min if d else -e"); err != nil {
		t.Fatal(err)
	} else if n := y.Names(); len(n) != 5 {
		t.Fatalf("names %v, want stored, rows, c, d and e", n)
	}
	if _, err := x.Eval(vars, 2); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("an expression past its steps: %v", err)
	}
}
