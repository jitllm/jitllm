package composer

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestEveryOpKindDecomposes holds the composer to refusing no op a graph can
// hold: every named op kind has a decomposition. The violation, a registry
// with one decomposition taken out, is seen.
func TestEveryOpKindDecomposes(t *testing.T) {
	d := decompositions()
	missing, named := undecomposed(d)
	if named < 40 {
		t.Fatalf("read %d named op kinds: the gate read nothing", named)
	}
	for _, k := range missing {
		t.Errorf("op kind %s has no decomposition: a graph holding it is refused at composition", k)
	}
	delete(d, graph.OpMap)
	if got, _ := undecomposed(d); len(got) != len(missing)+1 {
		t.Errorf("a registry without the map's decomposition gave %v", got)
	}
}

// undecomposed is the named op kinds d holds no decomposition of, and how
// many kinds are named; a kind past the last named one prints as its number.
func undecomposed(d map[graph.OpKind]Decomposition) (missing []graph.OpKind, named int) {
	for i := 1; i < 256; i++ {
		k := graph.OpKind(i)
		if k.String() == fmt.Sprintf("op%d", i) {
			continue
		}
		named++
		if _, ok := d[k]; !ok {
			missing = append(missing, k)
		}
	}
	return missing, named
}
