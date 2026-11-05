package engine

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/tok/jinja"
)

// TestChatTemplateCompilesOncePerModel: a request renders its model's chat
// template and does not compile it -- the compile is hundreds of allocations
// a request used to pay every time -- and the one compiled template renders
// for every request at once.
func TestChatTemplateCompilesOncePerModel(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "tok", "jinja", "testdata", "templates", "qwen3.jinja"))
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	m := &model.Model{}
	tpl, err := m.CompiledTemplate(src)
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := m.CompiledTemplate(src); again != tpl {
		t.Fatal("a second request compiled the template again")
	}
	saved := testing.AllocsPerRun(5, func() { jinja.Compile(src) })
	if n := testing.AllocsPerRun(100, func() { m.CompiledTemplate(src) }); n != 0 {
		t.Fatalf("finding the compiled template makes %.0f allocations, want 0 (a compile makes %.0f)", n, saved)
	}
	t.Logf("a compile, which a request no longer makes: %.0f allocations", saved)

	// One template, rendering for several requests at once (-race sees a
	// render that writes to it).
	msgs := []model.ChatMessage{{Role: "user", Content: "Name a prime."}}
	want, err := model.RenderCompiled(tpl, src, "", "", nil, msgs, nil, model.ToolChoice{}, true, false, "")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan string, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				got, err := model.RenderCompiled(tpl, src, "", "", nil, msgs, nil, model.ToolChoice{}, true, false, "")
				if err != nil || got != want {
					errs <- got
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for got := range errs {
		t.Fatalf("a concurrent render gave %q, want %q", got, want)
	}
}
