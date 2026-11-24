// Command genmodels writes docs/models.md, the supported-models list, from the
// converter's own tables (internal/modelsdoc):
//
//	go run ./scripts/genmodels docs/models.md
//
// With no argument it prints the page. It writes nothing when a name in the
// code has no describe line in its library file.
package main

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/internal/modelsdoc"
)

func main() {
	b, err := modelsdoc.Render()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(os.Args) < 2 {
		os.Stdout.Write(b)
		return
	}
	if err := os.WriteFile(os.Args[1], b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
