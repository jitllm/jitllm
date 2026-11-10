package main

import (
	"fmt"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
)

// upgradeCmd brings each container to this build's version through the
// format's migrations (jlm.Upgrade): in place where a version changed only
// the version word, rewritten from its own pages where it changed tensors.
// A container written before the converter stated its activations has them
// stated from its own config and tensors (convert.StateStored). The source
// the container was converted from is not needed.
func upgradeCmd(args []string) error {
	for _, path := range args {
		from, err := jlm.Upgrade(path, convert.LegacyStated)
		if err != nil {
			return err
		}
		stated, err := jlm.Restate(path, convert.StateStored)
		if err != nil {
			return fmt.Errorf("stating %s's activations: %w", path, err)
		}
		switch {
		case from != jlm.Version:
			fmt.Printf("%s upgraded from version %d to %d\n", path, from, jlm.Version)
		case !stated:
			fmt.Printf("%s is already version %d\n", path, jlm.Version)
		}
		if stated {
			fmt.Printf("%s states its activations\n", path)
		}
	}
	return nil
}
