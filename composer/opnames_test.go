package composer

import (
	"fmt"
	"testing"
)

func TestEveryOpHasAName(t *testing.T) {
	for op := Op(0); op < numOps; op++ {
		if op.String() == fmt.Sprintf("op%d", op) {
			t.Errorf("op %d has no name", op)
		}
	}
}
