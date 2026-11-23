package convert

import (
	"errors"
	"strings"
	"testing"
)

func TestAnUnstatedArchitectureNamesTheMissingStatement(t *testing.T) {
	_, err := configOf(headerOnly("ollmo", baseKeys(4)))
	if !errors.Is(err, ErrUnstated) || !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("an unstated architecture is not ErrUnstated and ErrNotImplemented: %v", err)
	}
	if !strings.Contains(err.Error(), "convert/models") {
		t.Errorf("the refusal does not name where the statement goes: %v", err)
	}
	if errors.Is(ErrNotImplemented, ErrUnstated) {
		t.Error("every ErrNotImplemented reads as a missing statement")
	}
}
