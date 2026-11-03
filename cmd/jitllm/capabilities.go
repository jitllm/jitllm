package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/jitllm/jitllm/internal/capabilities"
)

// `jitllm capabilities [-json]` prints the capability manifest for this build:
// docs/capabilities.json's content with the binary's version and revision.
// The same data jitllmd serves at GET /v1/capabilities.
func capabilitiesCmd(args []string, w io.Writer) error {
	set := flag.NewFlagSet("capabilities", flag.ContinueOnError)
	set.SetOutput(os.Stderr)
	asJSON := set.Bool("json", false, "print the manifest as JSON")
	if err := set.Parse(args); err != nil {
		return err
	}
	if set.NArg() > 0 {
		return fmt.Errorf("jitllm capabilities takes no argument, got %q", set.Args())
	}
	m, err := capabilities.Build(buildVersion())
	if err != nil {
		return err
	}
	if *asJSON {
		b, err := capabilities.Render(m)
		if err != nil {
			return err
		}
		_, err = w.Write(b)
		return err
	}
	rev := m.Build.Revision
	if m.Build.Modified {
		rev += "+dirty"
	}
	fmt.Fprintf(w, "build          %s (%s) %s %s/%s\n", m.Build.Version, rev, m.Build.Go, m.Build.OS, m.Build.Arch)
	fmt.Fprintf(w, "container      version %d\n", m.Container.Version)
	var fs []string
	for _, f := range m.InputFormats {
		fs = append(fs, f.Name)
	}
	fmt.Fprintf(w, "inputs         %s\n", strings.Join(fs, ", "))
	names := func(gs []capabilities.Graph) string {
		var out []string
		for _, g := range gs {
			out = append(out, g.Name)
		}
		return strings.Join(out, ", ")
	}
	fmt.Fprintf(w, "architectures  %d: %s\n", len(m.Architectures), names(m.Architectures))
	fmt.Fprintf(w, "safetensors    %d: %s\n", len(m.SafetensorsClasses), names(m.SafetensorsClasses))
	fmt.Fprintf(w, "projectors     %d: %s\n", len(m.Projectors), strings.Join(m.Projectors, ", "))
	fmt.Fprintf(w, "quantization   %s (packed kernels: %s)\n", strings.Join(m.Quantization.Dequantable, ", "),
		strings.Join(m.Quantization.Packed, ", "))
	for _, b := range m.Backends {
		state := "implemented, not verified on hardware"
		if len(b.Verified) > 0 {
			var hw []string
			for _, v := range b.Verified {
				hw = append(hw, v.Hardware)
			}
			state = "verified on " + strings.Join(hw, "; ")
		}
		fmt.Fprintf(w, "backend        %-15s %s\n", b.Name, state)
	}
	for _, e := range m.API.Endpoints {
		fmt.Fprintf(w, "endpoint       %-7s %-22s %s\n", strings.Join(e.Methods, ","), e.Path, e.Surface)
	}
	fmt.Fprintf(w, "refusals       %d combinations refused rather than ignored (-json lists them)\n", len(m.API.Refusals))
	return nil
}
