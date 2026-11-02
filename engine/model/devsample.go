package model

// Where a sampled token is drawn is decided, not measured: on the device
// wherever it serves the draw (nn.Head.SampleK, the composed walk), which
// reads one token back where the alternative reads the vocabulary's row and
// draws on the CPU after it. The modes are the caller's overrides.

// DeviceSampleMode is WithDeviceSample's setting.
type DeviceSampleMode int

const (
	// DeviceSampleAuto draws on the device wherever it serves the draw.
	DeviceSampleAuto DeviceSampleMode = iota
	// DeviceSampleOn always draws on the device where it can.
	DeviceSampleOn
	// DeviceSampleOff always reads the row back.
	DeviceSampleOff
)

// WithDeviceSample forces where a sampled token is drawn.
func WithDeviceSample(m DeviceSampleMode) Option {
	return func(l *loadOpts) { l.opt.devSample = m }
}
