package convert

// ErrNotImplemented marks a refusal that is a scope decision rather than a
// defect: an architecture or a projector this engine does not implement.
// A caller sweeping a directory needs to tell it from a genuine conversion
// break (a skip against a red gate) without matching the message text. It
// wraps rather than replaces, so the message a person reads is unchanged.
const ErrNotImplemented constError = "not implemented by this engine"

type constError string

func (e constError) Error() string { return string(e) }
