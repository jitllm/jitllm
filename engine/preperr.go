package engine

import "github.com/jitllm/jitllm/nn"

// prepErr is the first error a device of the instance's routes met preparing
// what it was offered (nn.PrepReporter), nil for none.
func (s *Instance) prepErr() error {
	for _, p := range s.places {
		if r, ok := p.(nn.PrepReporter); ok {
			if err := r.PrepErr(); err != nil {
				return err
			}
		}
	}
	return nil
}
