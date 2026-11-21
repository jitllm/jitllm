package convert

import (
	"encoding/json"
	"fmt"
	"slices"
)

// Described is the line docs/models.md gives each name the converter lists,
// as the library states it: a library file's describe for its GGUF
// architectures and classes, projectors.json's for the vision tower, each
// projector and each audio tower.
type Described struct {
	GGUF, Classes, Projectors, Audio map[string]string
}

// Descriptions is every name's line, by list.
func Descriptions() (Described, error) {
	d := Described{GGUF: map[string]string{}, Classes: map[string]string{}, Projectors: map[string]string{},
		Audio: map[string]string{}}
	files, err := libraryFiles()
	if err != nil {
		return d, err
	}
	for _, mj := range files {
		for n, s := range mj.Describe {
			if slices.Contains(mj.Classes, n) {
				d.Classes[n] = s
			} else {
				d.GGUF[n] = s
			}
		}
	}
	var lib projectorFile
	if err := json.Unmarshal(projectorsJSON, &lib); err != nil {
		return d, fmt.Errorf("convert: projectors.json: %w", err)
	}
	d.GGUF[towerArch] = lib.Describe
	for n, p := range lib.Projectors {
		d.Projectors[n] = p.Describe
	}
	for n, a := range lib.Audio {
		d.Audio[n] = a.Describe
	}
	return d, nil
}
