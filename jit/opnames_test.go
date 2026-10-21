package jit

import "testing"

func TestEveryOpHasAName(t *testing.T) {
	for op := Op(0); op < numOps; op++ {
		if opNames[op] == "" {
			t.Errorf("op %d has no name", op)
		}
	}
}
