// Command shots plays every scenario in package mock and writes each frame as
// a PNG, with what stage found wrong with it.
//
//	go run ./cmd/shots -out shots
//
// The frames are the running app's: the window is mounted empty and filled by
// the mock engine through the same Store writes the real engine posts, so a
// widget that does not re-measure when its data arrives is visible here.
// stage.TestEveryScenarioRendersClean asserts what this prints.
package main

import (
	"flag"
	"fmt"
	"image/png"
	"os"
	"path/filepath"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/mock"
	"github.com/jitllm/jitllm/ui/stage"
)

func main() {
	out := flag.String("out", "/tmp/shots", "directory to write the PNGs into")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fail(err)
	}
	problems := 0
	for _, sc := range mock.Scenarios() {
		for _, size := range []struct {
			suffix string
			w, h   int
		}{{"", 1440, 900}, {"-narrow", 1000, 700}} {
			for _, dark := range []bool{false, true} {
				cfg := app.DefaultConfig(app.Env{})
				cfg.Light = !dark
				cfg.ModelDirs = []string{os.TempDir()}
				theme := ""
				if dark {
					theme = "-dark"
				}
				failed := stage.Play(sc, cfg, size.w, size.h, true, func(step string, f stage.Frame) {
					name := fmt.Sprintf("%s-%s%s%s.png", sc.Name, step, size.suffix, theme)
					write(filepath.Join(*out, name), f)
					for _, p := range f.Problems() {
						problems++
						fmt.Printf("%s: %s\n", name, p)
					}
				})
				for _, why := range failed {
					problems++
					fmt.Printf("%s: %s\n", sc.Name, why)
				}
			}
		}
	}
	fmt.Printf("wrote %s, %d problem(s)\n", *out, problems)
	if problems > 0 {
		os.Exit(1)
	}
}

func write(path string, f stage.Frame) {
	file, err := os.Create(path)
	if err != nil {
		fail(err)
	}
	defer file.Close()
	if err := png.Encode(file, f.Image); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
