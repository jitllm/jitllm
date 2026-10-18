package jinja

import (
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkScope renders a real template through the inline-slot scope. The
// map-only arm it was once measured against is gone with its package switch.
func BenchmarkScope(b *testing.B) {
	src, err := os.ReadFile(filepath.Join("testdata", "templates", "qwen3.jinja"))
	if err != nil {
		b.Fatal(err)
	}
	tpl, err := Compile(string(src))
	if err != nil {
		b.Fatal(err)
	}
	data := map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": "write a hello world http server in rust"}},
		"add_generation_prompt": true,
		"bos_token":             "<|begin_of_text|>",
		"eos_token":             "<|eot_id|>",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := tpl.Render(data); err != nil {
			b.Fatal(err)
		}
	}
}
