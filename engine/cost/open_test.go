package cost_test

import (
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"github.com/jitllm/jitllm/jit"
)

func init() {
	cost.SetComposeOpen(func(t *testing.T, name string) jit.Model {
		t.Helper()
		fresh.Must(t, testmodels.Path(name))
		m, err := model.Open(testmodels.Path(name))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m.Composing()
	})
}
