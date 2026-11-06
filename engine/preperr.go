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

// notePrep tells the device whose memory holds the history that preparing it
// failed: the history's arenas are the device's that runs the lanes.
func (s *Instance) notePrep(err error) {
	if n, ok := s.places[routeLanes].(nn.PrepNoter); ok {
		n.NotePrep(err)
	}
}

// storage is the history storage of the device whose memory holds the
// sessions' history: the lanes' device's.
func (s *Instance) storage() nn.HistoryStorage {
	return s.places[routeLanes].(nn.HistoryStorage)
}

// setSpinning tells each device whether the step read from storage.
func (s *Instance) setSpinning(on bool) {
	for _, p := range s.places {
		if sp, ok := p.(nn.Spinner); ok {
			sp.SetSpinning(on)
		}
	}
}
