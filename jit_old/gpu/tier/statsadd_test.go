package tier

import (
	"reflect"
	"testing"
	"time"
)

// Stats.add is a hand-written list of every counter, and a counter missing from
// it reads as zero with confidence (StreamOverlaps once printed 0 while the
// pipeline was demonstrably running). The gate is reflective: it walks every
// field Stats has and demands add() moved it.
func TestStatsAddSumsEveryCounter(t *testing.T) {
	var o Stats
	v := reflect.ValueOf(&o).Elem()
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		if !f.CanSet() {
			continue
		}
		switch f.Kind() {
		case reflect.Int, reflect.Int64:
			if f.Type() == reflect.TypeOf(time.Duration(0)) {
				f.SetInt(int64(time.Second))
			} else {
				f.SetInt(7)
			}
		case reflect.Uint, reflect.Uint64:
			f.SetUint(7)
		case reflect.Float64:
			f.SetFloat(7)
		}
	}

	var s Stats
	s.add(o)

	got := reflect.ValueOf(&s).Elem()
	missed := 0
	for i := 0; i < v.NumField(); i++ {
		name := v.Type().Field(i).Name
		src, dst := v.Field(i), got.Field(i)
		if !src.CanSet() {
			continue
		}
		var zeroSrc, zeroDst bool
		switch src.Kind() {
		case reflect.Int, reflect.Int64:
			zeroSrc, zeroDst = src.Int() == 0, dst.Int() == 0
		case reflect.Uint, reflect.Uint64:
			zeroSrc, zeroDst = src.Uint() == 0, dst.Uint() == 0
		case reflect.Float64:
			zeroSrc, zeroDst = src.Float() == 0, dst.Float() == 0
		default:
			continue // not a counter: bool, slice, string
		}
		if zeroSrc {
			t.Fatalf("%s could not be seeded, so this gate proved nothing about it", name)
		}
		if zeroDst {
			t.Errorf("Stats.add does not sum %s: it is a hand-written list and "+
				"this field fell off it, so every report of it reads zero", name)
			missed++
		}
	}
	if missed == 0 {
		t.Logf("add() carries all %d counters", v.NumField())
	}
}
