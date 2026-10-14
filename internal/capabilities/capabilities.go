// Package capabilities is the machine-readable answer to "does this build
// support my workload": the architectures, Hugging Face classes and projectors
// the converter takes, the input formats, the quantization types, the backends
// (implemented, and verified on which hardware, with the gate that is the
// evidence), the HTTP API's endpoints, request fields, limits and refused
// combinations, and the build's version and revision.
//
// Everything derivable is derived: the lists from convert, quant and jlm at
// call time, and the API section from the server's route table and request
// structs by reflection (server.API, written to api.json beside this file
// because this module cannot import the server's). What no code can derive --
// on which hardware a backend has been verified, and the input formats -- is
// maintained.json, held by TestMaintainedEvidenceExists to name gates and
// files that exist.
//
// docs/capabilities.json is Doc() rendered; `jitllm capabilities` and
// jitllmd's GET /v1/capabilities serve Build() for the running build.
// internal/srcgate.TestCapabilitiesDocIsGenerated fails when the file and the
// code part, and server.TestAPIManifestIsGenerated when api.json and the
// server part.
package capabilities

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"runtime"
	"runtime/debug"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// Path is where the rendered manifest lives, from the module root.
const Path = "docs/capabilities.json"

// APIPath is where the server's half lives, from the module root.
const APIPath = "internal/capabilities/api.json"

// Command regenerates both files. It runs in the server module, the only one
// that sees both halves.
const Command = "cd server && go run ./cmd/gencapabilities"

// Schema is the manifest's own version: bumped when a field changes meaning or
// goes away, not when one is added.
const Schema = 1

// Manifest is the whole answer.
type Manifest struct {
	Schema int `json:"schema"`
	// Build is the running binary's identity. It is absent from
	// docs/capabilities.json, which describes the source tree, and filled by
	// Build().
	Build *BuildInfo `json:"build,omitempty"`
	// ServerAPI is the serving jitllmd's control-plane compatibility version
	// (localserver.APIVersion), absent outside a running server.
	ServerAPI int `json:"server_api,omitempty"`
	// Container is the .jlm format version this build reads; it reads no
	// other (there is no backward compatibility, on purpose).
	Container          ContainerInfo `json:"container"`
	InputFormats       []Format      `json:"input_formats"`
	Architectures      []Graph       `json:"architectures"`
	SafetensorsClasses []Graph       `json:"safetensors_classes"`
	Projectors         []string      `json:"projectors"`
	Quantization       Quantization  `json:"quantization"`
	Backends           []Backend     `json:"backends"`
	API                API           `json:"api"`
}

// BuildInfo is what binary this is.
type BuildInfo struct {
	Version  string `json:"version"`
	Revision string `json:"revision"`
	Modified bool   `json:"modified,omitempty"`
	Go       string `json:"go"`
	OS       string `json:"os"`
	Arch     string `json:"arch"`
}

// ContainerInfo is the container format.
type ContainerInfo struct {
	Version uint32 `json:"version"`
	Note    string `json:"note"`
}

// Graph is a source name and the graph it converts to.
type Graph struct {
	Name  string `json:"name"`
	Graph string `json:"graph"`
}

// Quantization is the weight formats: what decodes, and what has packed
// generated kernels on the container path.
type Quantization struct {
	Dequantable []string `json:"dequantable"`
	Packed      []string `json:"packed"`
}

// Format is one input the converter takes (maintained.json).
type Format struct {
	Name     string   `json:"name"`
	Example  string   `json:"example"`
	Note     string   `json:"note,omitempty"`
	Evidence []string `json:"evidence"`
}

// Backend is one compute backend (maintained.json): implemented means the code
// is in this tree and built; Verified lists the hardware it has run its gates
// on, each with the gate that is the evidence. Implemented with no Verified
// entry is a claim about code, not about a card.
type Backend struct {
	Name        string         `json:"name"`
	Kind        string         `json:"kind"`
	Package     string         `json:"package"`
	Implemented bool           `json:"implemented"`
	Verified    []Verification `json:"verified"`
	Note        string         `json:"note,omitempty"`
}

// Verification is one piece of hardware a backend ran on.
type Verification struct {
	Hardware string   `json:"hardware"`
	Evidence []string `json:"evidence"`
}

// API is the server's surface (server.API).
type API struct {
	Endpoints []Endpoint `json:"endpoints"`
	Refusals  []Refusal  `json:"refusals"`
	// Control is the ConnectRPC control plane's services and methods, from
	// the generated descriptors.
	Control []Service `json:"control"`
}

// Endpoint is one HTTP route.
type Endpoint struct {
	Path    string   `json:"path"`
	Methods []string `json:"methods"`
	Surface string   `json:"surface"`
	Note    string   `json:"note,omitempty"`
	Fields  []Field  `json:"fields,omitempty"`
}

// Field is one request field: its wire name, its Go shape, and what it does
// and its limits.
type Field struct {
	Name string `json:"name"`
	Type string `json:"type"`
	Note string `json:"note"`
}

// Refusal is a combination the server refuses rather than ignores.
type Refusal struct {
	ID        string   `json:"id"`
	Endpoints []string `json:"endpoints"`
	When      string   `json:"when"`
	Message   string   `json:"message"`
}

// Service is one ConnectRPC service.
type Service struct {
	Name    string   `json:"name"`
	Methods []string `json:"methods"`
}

//go:embed maintained.json
var maintainedJSON []byte

//go:embed api.json
var apiJSON []byte

// maintained is the hand-kept half.
type maintained struct {
	InputFormats []Format  `json:"input_formats"`
	Backends     []Backend `json:"backends"`
}

func loadMaintained() (maintained, error) {
	var m maintained
	d := json.NewDecoder(bytes.NewReader(maintainedJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&m); err != nil {
		return m, fmt.Errorf("capabilities: maintained.json: %w", err)
	}
	return m, nil
}

// EmbeddedAPI is the server's half as api.json holds it.
func EmbeddedAPI() (API, error) {
	var a API
	d := json.NewDecoder(bytes.NewReader(apiJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&a); err != nil {
		return a, fmt.Errorf("capabilities: api.json: %w; run: %s", err, Command)
	}
	return a, nil
}

// Doc is the manifest of the source tree: everything but the build.
func Doc() (Manifest, error) {
	a, err := EmbeddedAPI()
	if err != nil {
		return Manifest{}, err
	}
	return DocWith(a)
}

// DocWith is Doc with the server's half given rather than read from api.json.
func DocWith(a API) (Manifest, error) { return assemble(a) }

// Build is Doc with the running binary's identity. version is the release
// version the binary was linked with ("dev" when none).
func Build(version string) (Manifest, error) {
	m, err := Doc()
	if err != nil {
		return m, err
	}
	b := Identity(version)
	m.Build = &b
	return m, nil
}

// WithAPI is Build with the server's half given rather than read from
// api.json: jitllmd serves what its own route table says.
func WithAPI(version string, a API) (Manifest, error) {
	m, err := assemble(a)
	if err != nil {
		return m, err
	}
	b := Identity(version)
	m.Build = &b
	return m, nil
}

func assemble(a API) (Manifest, error) {
	mt, err := loadMaintained()
	if err != nil {
		return Manifest{}, err
	}
	m := Manifest{
		Schema: Schema,
		Container: ContainerInfo{Version: jlm.Version,
			Note: "the engine reads only .jlm containers of exactly this version; anything else is converted with `jitllm convert`"},
		InputFormats: mt.InputFormats,
		Backends:     mt.Backends,
		API:          a,
	}
	for _, n := range convert.Architectures() {
		g, _ := convert.ArchitectureGraph(n)
		m.Architectures = append(m.Architectures, Graph{Name: n, Graph: g.String()})
	}
	for _, c := range convert.HFClasses() {
		g, _ := convert.HFClassGraph(c)
		m.SafetensorsClasses = append(m.SafetensorsClasses, Graph{Name: c, Graph: g.String()})
	}
	m.Projectors = convert.Projectors()
	m.Quantization = quantization()
	return m, nil
}

// quantization walks the type space rather than copying a list, so a type
// added to quant.Dequantable appears here without an edit.
func quantization() Quantization {
	q := Quantization{Dequantable: []string{}, Packed: []string{}}
	for t := quant.Type(0); t < 256; t++ {
		if quant.Dequantable(t) {
			q.Dequantable = append(q.Dequantable, t.String())
		}
	}
	for _, t := range quant.PackedTypes {
		q.Packed = append(q.Packed, t.String())
		// An engine type (TQ2_F32) sits above the GGUF type space the walk
		// covers.
		if t.Engine() && quant.Dequantable(t) {
			q.Dequantable = append(q.Dequantable, t.String())
		}
	}
	return q
}

// Identity is the running binary's version and revision, from the build
// info the Go toolchain embeds.
func Identity(version string) BuildInfo {
	b := BuildInfo{Version: version, Revision: "unknown", Go: runtime.Version(), OS: runtime.GOOS, Arch: runtime.GOARCH}
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return b
	}
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			b.Revision = s.Value
		case "vcs.modified":
			b.Modified = s.Value == "true"
		}
	}
	if b.Version == "" || b.Version == "dev" {
		if v := bi.Main.Version; v != "" && v != "(devel)" {
			b.Version = v
		}
	}
	return b
}
