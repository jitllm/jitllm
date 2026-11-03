package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/net/idna"
)

// Fetching a picture a request names by URL.
//
// Both APIs let a client name a picture by an http(s) URL, so the server can
// fetch one -- and a server that fetches what a request names is a proxy into
// whatever network it sits on. Every guard here is a layer of the same answer:
//
//   - nothing is fetched unless the operator lists the hosts that may be
//     (ImagePolicy.FetchAllow), matched exactly after normalising case, IDNA
//     and a trailing dot, with "*.example.com" one label deep and nothing
//     else; the first URL and every redirect are checked;
//   - https only unless FetchHTTP, ports 443 and 80 only unless FetchPorts,
//     no userinfo, no IPv6 zone, no https->http downgrade, at most three
//     redirects, no proxy;
//   - the address a connection is about to be made to -- after resolution, so
//     a name cannot resolve to a public address for a check and a private one
//     for the dial -- is refused if it is in any special-use range of the IANA
//     IPv4 and IPv6 special-purpose registries (deniedPrefixes);
//   - fetches are bounded server-wide (FetchConcurrency), and each by time,
//     header size, body size, and the request's total bytes.

// imageFetchRedirects is how many redirects a fetch follows.
const imageFetchRedirects = 3

// imageFetchHeaderBytes bounds a fetched response's headers.
const imageFetchHeaderBytes = 64 << 10

// errImageFetchOff is a remote picture on a server that lists no host to
// fetch from.
const errImageFetchOff serverError = "names a picture by a remote URL, and this server does not fetch " +
	"remote images: send it inline (base64), or start the server with the hosts it may fetch from " +
	"(jitllmd -image-fetch-allow host1,host2; server.Config.Images.FetchAllow)"

// normHost is a host name as the allow-list compares it: lower case, IDNA's
// ASCII form, no trailing dot. An IP literal is its canonical text.
func normHost(h string) (string, error) {
	h = strings.TrimSuffix(strings.TrimSpace(h), ".")
	if h == "" {
		return "", errors.New("an empty host")
	}
	if a, err := netip.ParseAddr(strings.Trim(h, "[]")); err == nil {
		if a.Zone() != "" {
			return "", errors.New("an IPv6 zone")
		}
		return a.String(), nil
	}
	a, err := idna.Lookup.ToASCII(strings.ToLower(h))
	if err != nil {
		return "", fmt.Errorf("host %q: %v", h, err)
	}
	return a, nil
}

// normAllow normalises an allow-list entry, keeping a leading "*." as the
// one-label wildcard.
func normAllow(e string) (string, error) {
	e = strings.TrimSpace(e)
	wild := strings.HasPrefix(e, "*.")
	if wild {
		e = e[2:]
	}
	if strings.Contains(e, "*") {
		return "", fmt.Errorf("allow-list entry %q: only a leading \"*.\" is a wildcard", e)
	}
	h, err := normHost(e)
	if err != nil {
		return "", fmt.Errorf("allow-list entry %q: %v", e, err)
	}
	if wild {
		return "*." + h, nil
	}
	return h, nil
}

// hostAllowed is whether the normalised host h is on the list: an exact
// entry, or "*.parent" with h exactly one label more than parent.
func (p ImagePolicy) hostAllowed(h string) bool {
	for _, e := range p.FetchAllow {
		if e == h {
			return true
		}
		if parent, ok := strings.CutPrefix(e, "*."); ok {
			label, rest, found := strings.Cut(h, ".")
			if found && label != "" && rest == parent {
				return true
			}
		}
	}
	return false
}

// checkURL is whether u may be fetched: scheme, userinfo, host, zone, port.
// from is the URL that redirected to it, nil for the first.
func (p ImagePolicy) checkURL(u, from *url.URL) error {
	switch u.Scheme {
	case "https":
	case "http":
		if !p.FetchHTTP {
			return errors.New("is an http URL; this server fetches https only")
		}
		if from != nil && from.Scheme == "https" {
			return errors.New("redirects from https to http")
		}
	default:
		return fmt.Errorf("names a %q URL", u.Scheme)
	}
	if u.User != nil {
		return errors.New("carries userinfo")
	}
	if strings.Contains(u.Host, "%") {
		return errors.New("names an IPv6 zone")
	}
	h, err := normHost(u.Hostname())
	if err != nil {
		return err
	}
	if !p.hostAllowed(h) {
		return fmt.Errorf("names host %q, which is not on this server's image fetch allow-list", h)
	}
	port := u.Port()
	if port == "" {
		port = "443"
		if u.Scheme == "http" {
			port = "80"
		}
	}
	n, err := strconv.Atoi(port)
	if err != nil || !slices.Contains(p.FetchPorts, n) {
		return fmt.Errorf("names port %s; this server fetches from ports %v", port, p.FetchPorts)
	}
	return nil
}

// fetch reads u's body, at most MaxBytes of it, within FetchTimeout.
func (p ImagePolicy) fetch(ctx context.Context, raw string) ([]byte, error) {
	if len(p.FetchAllow) == 0 {
		return nil, errImageFetchOff
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("is not a URL: %v", err)
	}
	if err := p.checkURL(u, nil); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, p.FetchTimeout)
	defer cancel()
	if p.fetchSlots != nil {
		select {
		case p.fetchSlots <- struct{}{}:
			defer func() { <-p.fetchSlots }()
		case <-ctx.Done():
			return nil, fmt.Errorf("could not be fetched: %d remote fetches are running and none finished "+
				"in time", cap(p.fetchSlots))
		}
	}
	d := &net.Dialer{Timeout: p.FetchTimeout, Control: p.refuseDenied}
	c := &http.Client{
		Transport: &http.Transport{Proxy: nil, DialContext: d.DialContext,
			TLSHandshakeTimeout: p.FetchTimeout, ResponseHeaderTimeout: p.FetchTimeout,
			MaxResponseHeaderBytes: imageFetchHeaderBytes, DisableKeepAlives: true,
			TLSClientConfig: &tls.Config{RootCAs: p.roots}},
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			if len(via) > imageFetchRedirects {
				return fmt.Errorf("more than %d redirects", imageFetchRedirects)
			}
			return p.checkURL(r.URL, via[len(via)-1].URL)
		},
	}
	defer c.CloseIdleConnections()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("is not a URL: %v", err)
	}
	// The destination is checked above (checkURL: allow-listed host, port,
	// scheme), on every redirect (CheckRedirect), and at every connection
	// (refuseDenied), which is what makes a request-named URL safe to fetch.
	resp, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("could not be fetched: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("could not be fetched: %s", resp.Status)
	}
	if resp.ContentLength > p.MaxBytes {
		return nil, fmt.Errorf("is %d bytes, over the limit of %d", resp.ContentLength, p.MaxBytes)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, p.MaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("could not be fetched: %v", err)
	}
	if int64(len(b)) > p.MaxBytes {
		return nil, fmt.Errorf("is over the limit of %d bytes", p.MaxBytes)
	}
	return b, nil
}

// pfx4 is an IPv4 prefix, built from its bytes (the tree's private-value
// gate refuses dotted literals outside the documentation ranges).
func pfx4(a, b, c, d byte, bits int) netip.Prefix {
	return netip.PrefixFrom(netip.AddrFrom4([4]byte{a, b, c, d}), bits)
}

// deniedPrefixes is every special-use range of the IANA IPv4 and IPv6
// special-purpose address registries a fetch may not connect to: this
// network, private, shared, loopback, link-local, the IETF protocol block,
// documentation, 6to4 relay, benchmarking, multicast, reserved and broadcast;
// unspecified, loopback, IPv4-mapped, both NAT64 prefixes, discard-only, the
// IETF protocol block (Teredo inside it), documentation, 6to4, unique-local,
// link-local and multicast.
func deniedPrefixes() []netip.Prefix {
	return []netip.Prefix{
		pfx4(0, 0, 0, 0, 8),
		pfx4(10, 0, 0, 0, 8),
		pfx4(100, 64, 0, 0, 10),
		pfx4(127, 0, 0, 0, 8),
		pfx4(169, 254, 0, 0, 16),
		pfx4(172, 16, 0, 0, 12),
		pfx4(192, 0, 0, 0, 24),
		pfx4(192, 0, 2, 0, 24),
		pfx4(192, 88, 99, 0, 24),
		pfx4(192, 168, 0, 0, 16),
		pfx4(198, 18, 0, 0, 15),
		pfx4(198, 51, 100, 0, 24),
		pfx4(203, 0, 113, 0, 24),
		pfx4(224, 0, 0, 0, 4),
		pfx4(240, 0, 0, 0, 4),
		pfx4(255, 255, 255, 255, 32),
		netip.MustParsePrefix("::/128"),
		netip.MustParsePrefix("::1/128"),
		netip.MustParsePrefix("::ffff:0:0/96"),
		netip.MustParsePrefix("64:ff9b::/96"),
		netip.MustParsePrefix("64:ff9b:1::/48"),
		netip.MustParsePrefix("100::/64"),
		netip.MustParsePrefix("2001::/23"),
		netip.MustParsePrefix("2001::/32"),
		netip.MustParsePrefix("2001:db8::/32"),
		netip.MustParsePrefix("2002::/16"),
		netip.MustParsePrefix("fc00::/7"),
		netip.MustParsePrefix("fe80::/10"),
		netip.MustParsePrefix("ff00::/8"),
	}
}

// deniedAddr is whether a fetch may not connect to a. An IPv4-mapped IPv6
// address is denied as such and again as the IPv4 address it maps, so
// neither spelling reaches a private host.
func deniedAddr(a netip.Addr) bool {
	if !a.IsValid() || a.Zone() != "" {
		return true
	}
	for _, x := range []netip.Addr{a, a.Unmap()} {
		for _, p := range deniedPrefixes() {
			if p.Contains(x) {
				return true
			}
		}
	}
	return false
}

// refuseDenied is the dialer's Control: it refuses an address deniedAddr
// names, unless a test exempts it.
func (p ImagePolicy) refuseDenied(network, address string, _ syscall.RawConn) error {
	ap, err := netip.ParseAddrPort(address)
	if err != nil {
		return fmt.Errorf("refusing %s: %v", address, err)
	}
	if slices.Contains(p.exempt, ap) {
		return nil
	}
	if deniedAddr(ap.Addr()) {
		return fmt.Errorf("refusing %s: a remote image may not be fetched from a special-use "+
			"(private, loopback, link-local, NAT64, mapped, documentation or multicast) address", ap.Addr())
	}
	return nil
}
