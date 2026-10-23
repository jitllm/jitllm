package jit

import (
	"cmp"
	"fmt"
	"hash/fnv"
	"math"
	"reflect"
	"slices"
	"sync"
)

// Key names a request by everything it is formed from: its graph (every
// value, op, weight and attribute), formats, rotaries, step and device, and
// the profile and lowering the registry forms it under. Two requests of one
// Key form the same kernels, so a call site's kernels are cached under it,
// once, here and on its device.
type Key uint64

// formCache is a registry's formed requests by Key, each kept with the words
// it was keyed from: a hit whose words differ is a collision, an error, never
// another request's kernels. A request that did not form is kept too.
type formCache struct {
	mu sync.Mutex
	m  map[Key]formed
}

type formed struct {
	words []uint64
	codes []Code
	err   error
}

func newFormCache() *formCache { return &formCache{m: map[Key]formed{}} }

// keyOf is req's Key and the words it hashes: req, then the profile and the
// lowering of its device and the registry's decompositions, each walked
// field by field, so a field added to any of them is keyed without being
// named here.
func (r *Registry) keyOf(req Request) (Key, []uint64, error) {
	var w keyWords
	if err := w.walk(reflect.ValueOf(req)); err != nil {
		return 0, nil, err
	}
	if err := w.walk(reflect.ValueOf(r.Profiles[req.Device.Kind])); err != nil {
		return 0, nil, err
	}
	if err := w.walk(reflect.ValueOf(&r.Lowerings).Elem().MapIndex(reflect.ValueOf(req.Device.Kind))); err != nil {
		return 0, nil, err
	}
	// The decompositions, each by its code: a registry whose entry was
	// replaced forms another Key, never the old entry's kernels.
	if err := w.walk(reflect.ValueOf(r.Decompositions)); err != nil {
		return 0, nil, err
	}
	h := fnv.New64a()
	var b [8]byte
	for _, x := range w {
		for i := range b {
			b[i] = byte(x >> (8 * i))
		}
		h.Write(b[:])
	}
	return Key(h.Sum64()), w, nil
}

type keyWords []uint64

// walk appends v's words: a scalar its bits, a string its hash, a sequence
// its length and elements, a map its length and entries in key order, a
// pointer or interface whether it is nil and what it holds (an interface its
// dynamic type's name too), a struct its fields.
func (w *keyWords) walk(v reflect.Value) error {
	if !v.IsValid() {
		*w = append(*w, 0)
		return nil
	}
	switch v.Kind() {
	case reflect.Bool:
		*w = append(*w, b2u(v.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		*w = append(*w, uint64(v.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		*w = append(*w, v.Uint())
	case reflect.Float32, reflect.Float64:
		*w = append(*w, math.Float64bits(v.Float()))
	case reflect.String:
		w.str(v.String())
	case reflect.Array, reflect.Slice:
		*w = append(*w, uint64(v.Len()))
		for i := range v.Len() {
			if err := w.walk(v.Index(i)); err != nil {
				return err
			}
		}
	case reflect.Map:
		keys := v.MapKeys()
		var err error
		slices.SortFunc(keys, func(a, b reflect.Value) int {
			c, e := compareKeys(a, b)
			if e != nil {
				err = e
			}
			return c
		})
		if err != nil {
			return err
		}
		*w = append(*w, uint64(len(keys)))
		for _, k := range keys {
			if err := w.walk(k); err != nil {
				return err
			}
			if err := w.walk(v.MapIndex(k)); err != nil {
				return err
			}
		}
	case reflect.Pointer:
		*w = append(*w, b2u(!v.IsNil()))
		if !v.IsNil() {
			return w.walk(v.Elem())
		}
	case reflect.Interface:
		*w = append(*w, b2u(!v.IsNil()))
		if !v.IsNil() {
			w.str(v.Elem().Type().String())
			return w.walk(v.Elem())
		}
	case reflect.Func:
		// A function is its code: within the process that holds the
		// cache, one address is one function.
		*w = append(*w, uint64(v.Pointer()))
	case reflect.Struct:
		for i := range v.NumField() {
			if err := w.walk(v.Field(i)); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("jit: a request's %s (%s) has no key", v.Type(), v.Kind())
	}
	return nil
}

func (w *keyWords) str(s string) {
	h := fnv.New64a()
	h.Write([]byte(s))
	*w = append(*w, uint64(len(s)), h.Sum64())
}

func b2u(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

func compareKeys(a, b reflect.Value) (int, error) {
	switch a.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return cmp.Compare(a.Int(), b.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return cmp.Compare(a.Uint(), b.Uint()), nil
	case reflect.String:
		return cmp.Compare(a.String(), b.String()), nil
	}
	return 0, fmt.Errorf("jit: a map keyed by %s has no key order", a.Type())
}
