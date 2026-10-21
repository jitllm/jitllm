package cost

import (
	"encoding/json"
	"fmt"
)

// ProfileVersion is the layout of a Profile's measurements. A reader
// refuses a profile of another version rather than price a device with
// numbers it would read wrong.
const ProfileVersion = 1

// Profile is what a node knows of one of its devices, as its probe measured
// it there: data another node can price that device with, ahead of sending
// it work, without the device, its runtime or its compiler.
type Profile struct {
	Version int    `json:"version"`
	Name    string `json:"name"`
	API     string `json:"api"`
	Device  Device `json:"device"`
}

// Encode is the profile as it travels between nodes.
func (p Profile) Encode() ([]byte, error) {
	p.Version = ProfileVersion
	return json.Marshal(p)
}

// DecodeProfile is the profile b carries, refused where it was written for
// another version of the measurements.
func DecodeProfile(b []byte) (Profile, error) {
	var p Profile
	if err := json.Unmarshal(b, &p); err != nil {
		return Profile{}, err
	}
	if p.Version != ProfileVersion {
		return Profile{}, fmt.Errorf("cost: a profile of version %d; this build reads %d", p.Version, ProfileVersion)
	}
	return p, nil
}
