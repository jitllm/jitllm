#!/usr/bin/env python3
"""Block uncapped builds. See require-cap.sh for why this exists.

Matching happens only at SHELL COMMAND POSITIONS. An earlier version regex'd the
raw command string and refused a `git commit` whose message contained the word
"make" -- a false positive that blocks real work, which is how a safety check
earns itself a bypass. Quoted strings and heredoc bodies are stripped first so
that prose can never look like a command.
"""
import json, re, sys

HEAVY = re.compile(
    r'(?:^|[;&|(]|\bthen\b|\bdo\b|&&|\|\|)\s*'
    r'(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*'          # leading VAR=val assignments
    r'(?:go\s+(?:test|build|run|vet|generate)\b'
    r'|(?:gcc|clang|clang\+\+|g\+\+|cc|nvcc|make)\b'
    r'|\./[A-Za-z0-9_./-]*\.test\b)'
)

def strip_quoted(s: str) -> str:
    """Blank out heredoc bodies and quoted strings, preserving length-ish shape."""
    # Heredocs: <<'TAG' ... TAG  /  <<TAG ... TAG  /  <<-TAG
    for m in list(re.finditer(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", s)):
        tag = m.group(2)
        end = re.search(r'^\s*%s\s*$' % re.escape(tag), s[m.end():], re.M)
        stop = m.end() + (end.start() if end else len(s) - m.end())
        s = s[:m.end()] + ' ' * (stop - m.end()) + s[stop:]
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in ("'", '"'):
            j = i + 1
            while j < n and s[j] != c:
                if c == '"' and s[j] == '\\':
                    j += 1
                j += 1
            out.append(' ' * (min(j, n - 1) - i + 1))
            i = j + 1
            continue
        out.append(c)
        i += 1
    return ''.join(out)

def main() -> int:
    try:
        cmd = json.load(sys.stdin).get('tool_input', {}).get('command', '')
    except Exception:
        return 0                      # fail open
    if not cmd:
        return 0
    if 'scripts/cap' in cmd or 'JITLLM_NO_CAP=1' in cmd:
        return 0
    if not HEAVY.search(strip_quoted(cmd)):
        return 0
    sys.stderr.write(
        "BLOCKED by scripts/require-cap.sh (AGENTS.md RULE 3).\n\n"
        "This command builds or runs compiled code and is not wrapped in\n"
        "scripts/cap, so it can consume unbounded memory and take the desktop\n"
        "down with it. That has happened twice.\n\n"
        "Re-run it as:\n\n    ./scripts/cap 8G -- <your command>\n\n"
        "and for anything that measures, pin the cores too:\n\n"
        "    ./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 <your command>\n\n"
        "If a command genuinely must run uncapped, prefix it with JITLLM_NO_CAP=1\n"
        "and say why in the same message.\n\nRefused: %s\n" % cmd[:400])
    return 2

sys.exit(main())
