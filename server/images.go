package server

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // a GIF's first frame, which image.Decode returns
	_ "image/jpeg"
	_ "image/png"
	"net/netip"
	"strings"
	"time"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// Pictures in a chat request.
//
// Both HTTP shims accept pictures the way their APIs send them -- OpenAI's
// image_url parts, Anthropic's image blocks -- and turn each into a decoded
// image.Image, in the order the client sent them, carried on ChatInput.Images.
// The engine then lays them out where the model's own chat template puts its
// image markers (Engine.pictureSpans), so a picture is a span of the prompt and
// the prefix cache names it by its ImageKey as it does on every other path.
//
// Every number that sizes a decode comes from a header checked against
// ImagePolicy first: the encoded bytes before base64 is decoded or a body is
// read, and the pixel count from image.DecodeConfig before image.Decode
// allocates the raster.

// ImagePolicy bounds the pictures a request may carry. The zero value takes
// every default and fetches nothing remote.
type ImagePolicy struct {
	// FetchAllow is the hosts a picture named by an http(s) URL may be
	// fetched from (imagefetch.go): exact names, or "*.example.com" for one
	// label under example.com. Empty, the default, fetches nothing: a server
	// that fetches what a request names is a proxy into whatever network it
	// sits on.
	FetchAllow []string
	// FetchHTTP lets a fetch use plain http (default https only).
	FetchHTTP bool
	// FetchPorts is the ports a fetch may connect to (default 443 and 80).
	FetchPorts []int
	// FetchConcurrency bounds the fetches running at once across the server
	// (default 4).
	FetchConcurrency int
	// FetchTimeout bounds one fetch, the connection and the body together
	// (default 10 s).
	FetchTimeout time.Duration
	// DecodeTimeout bounds decoding one picture (default 5 s).
	DecodeTimeout time.Duration
	// MaxBytes bounds one picture's encoded bytes (default 20 MiB).
	MaxBytes int64
	// MaxRequestBytes bounds the encoded bytes of all of a request's pictures
	// (default 64 MiB).
	MaxRequestBytes int64
	// MaxPixels bounds one picture's width times height (default 4096x4096).
	MaxPixels int64
	// MaxImages bounds the pictures of one request (default 8).
	MaxImages int
	// MaxVideoPixels bounds the pixels of a request's video frames together
	// (default MaxImages pictures of MaxPixels).
	MaxVideoPixels int64

	// fetchSlots is the server-wide fetch bound, made once by withDefaults.
	fetchSlots chan struct{}
	// exempt is the addresses a fetch may connect to whatever their range; only
	// tests set it, since an httptest server is on the loopback. Nothing a
	// request carries reaches it.
	exempt []netip.AddrPort
	// roots is the TLS roots a fetch trusts, nil for the system's; only tests
	// set it.
	roots *x509.CertPool
	// spent is the encoded bytes a request's pictures have used, per request
	// (newPictures).
	spent *int64
}

// The ImagePolicy defaults.
const (
	DefaultImageMaxBytes         = 20 << 20
	DefaultImageMaxRequestBytes  = 64 << 20
	DefaultImageMaxPixels        = 4096 * 4096
	DefaultImageMaxImages        = 8
	DefaultImageFetchTimeout     = 10 * time.Second
	DefaultImageDecodeTimeout    = 5 * time.Second
	DefaultImageFetchConcurrency = 4
)

func (p ImagePolicy) withDefaults() ImagePolicy {
	if p.MaxBytes <= 0 {
		p.MaxBytes = DefaultImageMaxBytes
	}
	if p.MaxPixels <= 0 {
		p.MaxPixels = DefaultImageMaxPixels
	}
	if p.MaxImages <= 0 {
		p.MaxImages = DefaultImageMaxImages
	}
	if p.MaxVideoPixels <= 0 {
		p.MaxVideoPixels = int64(p.MaxImages) * p.MaxPixels
	}
	if p.FetchTimeout <= 0 {
		p.FetchTimeout = DefaultImageFetchTimeout
	}
	if p.DecodeTimeout <= 0 {
		p.DecodeTimeout = DefaultImageDecodeTimeout
	}
	if p.MaxRequestBytes <= 0 {
		p.MaxRequestBytes = DefaultImageMaxRequestBytes
	}
	if len(p.FetchPorts) == 0 {
		p.FetchPorts = []int{443, 80}
	}
	if p.FetchConcurrency <= 0 {
		p.FetchConcurrency = DefaultImageFetchConcurrency
	}
	if p.fetchSlots == nil {
		p.fetchSlots = make(chan struct{}, p.FetchConcurrency)
	}
	return p
}

// ready is p with its defaults and its allow-list normalised. An entry that
// does not normalise turns fetching off rather than fetch from a list nobody
// meant; jitllmd refuses such a list at start (Validate).
func (p ImagePolicy) ready() ImagePolicy {
	p = p.withDefaults()
	v, err := p.Validate()
	if err != nil {
		p.FetchAllow = nil
		return p
	}
	return v
}

// Validate normalises the allow-list, refusing an entry that is not a host
// name or a one-label wildcard, and a port that is not one.
func (p ImagePolicy) Validate() (ImagePolicy, error) {
	allow := make([]string, 0, len(p.FetchAllow))
	for _, e := range p.FetchAllow {
		if strings.TrimSpace(e) == "" {
			continue
		}
		h, err := normAllow(e)
		if err != nil {
			return p, err
		}
		allow = append(allow, h)
	}
	p.FetchAllow = allow
	for _, port := range p.FetchPorts {
		if port <= 0 || port > 65535 {
			return p, fmt.Errorf("image fetch port %d is not a port", port)
		}
	}
	return p, nil
}

// imageFormats maps a decoder's format name to the media type it is sent as.
func imageFormats() map[string]string {
	return map[string]string{
		"png":  "image/png",
		"jpeg": "image/jpeg",
		"gif":  "image/gif",
		"webp": "image/webp",
		"bmp":  "image/bmp",
		"tiff": "image/tiff",
	}
}

// imageMediaType is the media type for a declared one, "image/jpg" taken as
// the alias clients send; ok is false for a type that is not decoded here.
func imageMediaType(mt string) (string, bool) {
	mt = strings.ToLower(strings.TrimSpace(mt))
	if mt == "image/jpg" {
		mt = "image/jpeg"
	}
	for _, v := range imageFormats() {
		if v == mt {
			return mt, true
		}
	}
	return "", false
}

// decodeImage decodes b, a picture whose declared media type is mt ("" for
// none), within p's bounds.
func (p ImagePolicy) decodeImage(b []byte, mt string) (image.Image, error) {
	if int64(len(b)) > p.MaxBytes {
		return nil, fmt.Errorf("is %d bytes, over the limit of %d", len(b), p.MaxBytes)
	}
	if p.spent != nil {
		if *p.spent+int64(len(b)) > p.MaxRequestBytes {
			return nil, fmt.Errorf("brings the request's pictures to %d bytes, over the limit of %d",
				*p.spent+int64(len(b)), p.MaxRequestBytes)
		}
		*p.spent += int64(len(b))
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("is not a PNG, JPEG, WebP, GIF, BMP or TIFF picture: %v", err)
	}
	got, ok := imageFormats()[format]
	if !ok {
		return nil, fmt.Errorf("is %s; the pictures accepted are PNG, JPEG, WebP, GIF, BMP and TIFF", format)
	}
	if mt != "" && mt != got {
		return nil, fmt.Errorf("is declared %s and its bytes are %s", mt, got)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("is %dx%d, an empty picture", cfg.Width, cfg.Height)
	}
	if px := int64(cfg.Width) * int64(cfg.Height); px > p.MaxPixels {
		return nil, fmt.Errorf("is %dx%d, %d pixels, over the limit of %d", cfg.Width, cfg.Height, px, p.MaxPixels)
	}
	// The decode is bounded by the pixel limit above; the deadline bounds its
	// time apart from any fetch's. A decode past it finishes in the
	// background, within the same pixel bound, and its picture is dropped.
	type result struct {
		img image.Image
		err error
	}
	done := make(chan result, 1)
	start := time.Now()
	go func() {
		img, _, err := image.Decode(bytes.NewReader(b))
		done <- result{img, err}
	}()
	t := time.NewTimer(p.DecodeTimeout)
	defer t.Stop()
	select {
	case r := <-done:
		// A coarse timer (Windows ticks in milliseconds) can lose the race to
		// a decode that already ran past the deadline; the clock decides.
		if time.Since(start) > p.DecodeTimeout {
			return nil, fmt.Errorf("did not decode within %s", p.DecodeTimeout)
		}
		if r.err != nil {
			return nil, fmt.Errorf("does not decode: %v", r.err)
		}
		return r.img, nil
	case <-t.C:
		return nil, fmt.Errorf("did not decode within %s", p.DecodeTimeout)
	}
}

// decodeBase64Image decodes a base64 payload declared as mt, sizing nothing
// from the payload before its length is checked.
func (p ImagePolicy) decodeBase64Image(data, mt string) (image.Image, error) {
	data = strings.TrimSpace(data)
	if int64(base64.StdEncoding.DecodedLen(len(data))) > p.MaxBytes+2 {
		return nil, fmt.Errorf("is about %d bytes, over the limit of %d",
			base64.StdEncoding.DecodedLen(len(data)), p.MaxBytes)
	}
	b, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		// Some clients send the unpadded or URL-safe alphabet.
		if b, err = base64.RawStdEncoding.DecodeString(strings.TrimRight(data, "=")); err != nil {
			if b, err = base64.RawURLEncoding.DecodeString(strings.TrimRight(data, "=")); err != nil {
				return nil, errors.New("is not valid base64")
			}
		}
	}
	return p.decodeImage(b, mt)
}

// decodeDataURL decodes a data: URL, data:<media type>[;params];base64,<data>.
func (p ImagePolicy) decodeDataURL(u string) (image.Image, error) {
	rest, ok := cutPrefixFold(u, "data:")
	if !ok {
		return nil, errors.New("is not a data: URL")
	}
	head, data, ok := strings.Cut(rest, ",")
	if !ok {
		return nil, errors.New("is a data: URL with no comma before its data")
	}
	params := strings.Split(head, ";")
	mt, ok := imageMediaType(params[0])
	if !ok {
		return nil, fmt.Errorf("is a data: URL of type %q; the pictures accepted are image/png, "+
			"image/jpeg, image/webp, image/gif, image/bmp and image/tiff", params[0])
	}
	b64 := false
	for _, x := range params[1:] {
		if strings.EqualFold(strings.TrimSpace(x), "base64") {
			b64 = true
		}
	}
	if !b64 {
		return nil, errors.New("is a data: URL that is not base64; send ;base64,")
	}
	return p.decodeBase64Image(data, mt)
}

func cutPrefixFold(s, prefix string) (string, bool) {
	if len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix) {
		return s[len(prefix):], true
	}
	return s, false
}

// imageURL decodes the picture a URL names: a data: URL inline, an http(s)
// one fetched when p allows it.
func (p ImagePolicy) imageURL(ctx context.Context, u string) (image.Image, error) {
	u = strings.TrimSpace(u)
	if _, ok := cutPrefixFold(u, "data:"); ok {
		return p.decodeDataURL(u)
	}
	_, http1 := cutPrefixFold(u, "http://")
	_, https := cutPrefixFold(u, "https://")
	if !http1 && !https {
		return nil, errors.New("is neither a data: URL nor an http(s) URL")
	}
	b, err := p.fetch(ctx, u)
	if err != nil {
		return nil, err
	}
	return p.decodeImage(b, "")
}

// imageError names the part of a request a picture came from.
type imageError struct {
	part string
	err  error
}

func (e *imageError) Error() string { return e.part + " " + e.err.Error() }
func (e *imageError) Unwrap() error { return e.err }

// pictures collects a request's pictures in order, against p's bounds.
type pictures struct {
	p    ImagePolicy
	imgs []image.Image
}

// newPictures is a request's collector: p with a byte count of its own.
func newPictures(p ImagePolicy) *pictures {
	p.spent = new(int64)
	return &pictures{p: p}
}

// add decodes one picture with load, naming part on failure.
func (ps *pictures) add(part string, load func() (image.Image, error)) error {
	if len(ps.imgs) >= ps.p.MaxImages {
		return &imageError{part, fmt.Errorf("is picture %d; a request carries at most %d",
			len(ps.imgs)+1, ps.p.MaxImages)}
	}
	img, err := load()
	if err != nil {
		return &imageError{part, err}
	}
	ps.imgs = append(ps.imgs, img)
	return nil
}
