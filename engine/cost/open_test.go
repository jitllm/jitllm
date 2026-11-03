package cost_test

import (
	"flag"
	"os"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// fixtureOpener opens a fixture with the model; the tests in package cost
// reach it through the flag it is registered as.
type fixtureOpener struct{}

func (fixtureOpener) String() string   { return "" }
func (fixtureOpener) Set(string) error { return nil }

func (fixtureOpener) Open(t *testing.T, name string) composer.Model {
	t.Helper()
	fresh.Must(t, testmodels.Path(name))
	m, err := model.Open(testmodels.Path(name))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m.Composing()
}

func TestMain(m *testing.M) {
	flag.String("compose-model", "synth-ernie45", "the fixture the estimates price")
	flag.Var(fixtureOpener{}, "compose-opener", "set by TestMain: opens a fixture as a composer model")
	os.Exit(m.Run())
}
