package convert

import "github.com/jitllm/jitllm/format/jlm"

// stateTower states s's tower's conventions (jlm.VisionConv): the facts its
// tensors do not show, which its projector's reference decides by name. The
// container is where a name becomes a fact (jlm.ConventionsOf); past it, a
// tower is read by what it states and what it stores.
func stateTower(s *jlm.Source) error {
	if s.Vision == nil {
		return nil
	}
	c, err := jlm.ConventionsOf(s.Vision)
	if err != nil {
		return err
	}
	s.Vision.Conv = c
	return nil
}
