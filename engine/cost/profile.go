package cost

// Profile is what is known of a hardware model, as a probe measured it on
// one or as its specification approximates it: data any node can price that
// hardware with, ahead of sending it work, without the device, its runtime
// or its compiler. Source says which: "probe" for a measurement.
type Profile struct {
	Name   string
	API    string
	Source string
	Device Device
}
