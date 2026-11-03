package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/jitllm/jitllm/convert/library"
)

// libraryArgs lets convert take a library name: `jitllm convert qwen3-8b`
// becomes the model's pinned hf:// reference and its vision tower's, and every
// other argument passes through.
//
// A name only when it is nothing else. A path that exists, anything with a
// slash or a scheme, is what it looks like; the library is consulted for a bare
// word that names no file, so no existing command line changes meaning.
func libraryArgs(args []string) []string {
	if len(args) == 0 || strings.ContainsAny(args[0], "/:\\") {
		return args
	}
	if _, err := os.Stat(args[0]); err == nil {
		return args
	}
	m, ok := library.Find(args[0])
	if !ok {
		return args
	}
	out := []string{m.Ref().String()}
	if t, ok := m.TowerRef(); ok {
		out = append(out, t.String())
	}
	return append(out, args[1:]...)
}

// libraryCmd lists the models jitllm can fetch by name.
func libraryCmd(args []string) error {
	fs := flag.NewFlagSet("library", flag.ContinueOnError)
	refs := fs.Bool("refs", false, "print each name, its size in bytes and its hf:// references, for scripts")
	if err := fs.Parse(args); err != nil {
		return err
	}
	for _, m := range library.Models() {
		if *refs {
			line := fmt.Sprintf("%s %d %s", m.Name, m.Bytes, m.Ref())
			if t, ok := m.TowerRef(); ok {
				line += " " + t.String()
			}
			fmt.Println(line)
			continue
		}
		vision := ""
		if m.Vision() {
			vision = "  sees images"
		}
		fmt.Printf("%-30s %-8s %-9s %6.1f GB%s\n", m.Name, m.Params, m.Arch, float64(m.Bytes)/1e9, vision)
	}
	return nil
}
