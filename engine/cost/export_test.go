package cost

import (
	"flag"
	"testing"

	"github.com/jitllm/jitllm/composer"
)

// FixtureOpener is what the external test package registers, as the
// "compose-opener" flag's value in its TestMain, for the tests here to open a
// fixture as a composer model: this package cannot import the model.
type FixtureOpener interface {
	Open(t *testing.T, name string) composer.Model
}

func composing(t *testing.T, name string) composer.Model {
	t.Helper()
	return flag.Lookup("compose-opener").Value.(FixtureOpener).Open(t, name)
}

// composeModel is the fixture the estimates price.
func composeModel() string { return flag.Lookup("compose-model").Value.String() }
