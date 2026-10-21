#!/usr/bin/env python3
"""Regenerate docs/kernel-inventory.csv from the source tree.

    scripts/kernel-inventory.py > docs/kernel-inventory.csv
    scripts/kernel-inventory.py --summary          # the counts the .md quotes
    scripts/kernel-inventory.py --skip jit_old/gpu/kernels/x.go ...   # leave a file out
    scripts/kernel-inventory.py --root DIR         # read another checkout, e.g.
                                                   # a `git archive <commit>` export

Every row is derived from the source, nothing is carried over:

  kind              cpu  -- a top-level exported `func Emit...` in jit_old/cpu
                          whose result is []byte or ([]byte, error): a
                          machine-code generator
                    gpu  -- a top-level exported func in jit_old/gpu/kernels
                          whose result carries *ir.Kernel: an IR constructor
  symbol, file,     the declaration as written; a name declared once per
  line              architecture file appears once per declaration
  build_constraint  the file's //go:build expression, else its GOOS/GOARCH
                    filename suffix, else "all"
  signature         the declaration up to its body, joined onto one line

Test files are skipped. Methods are not constructors and are skipped.
--summary also prints who calls each GPU constructor outside its package
(non-test .go files under the module), which the .md's family table uses.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CPU = "jit_old/cpu"
GPU = "jit_old/gpu/kernels"
ARCHES = {"amd64", "arm64", "386", "arm", "riscv64", "ppc64le", "s390x", "wasm", "loong64", "mips64"}
OSES = {"linux", "darwin", "windows", "freebsd", "openbsd", "netbsd", "android", "ios", "js", "wasip1"}


def go_files(pkg):
    d = os.path.join(ROOT, pkg)
    return sorted(f for f in os.listdir(d) if f.endswith(".go") and not f.endswith("_test.go"))


def constraint(path, name):
    with open(path) as fh:
        for line in fh:
            if line.startswith("//go:build "):
                return line[len("//go:build "):].strip()
            if line.startswith("package "):
                break
    parts = name[:-3].split("_")[1:]
    if parts and parts[-1] in ARCHES:
        if len(parts) > 1 and parts[-2] in OSES:
            return parts[-2] + " && " + parts[-1]
        return parts[-1]
    if parts and parts[-1] in OSES:
        return parts[-1]
    return "all"


def declarations(path):
    """Yield (line, name, signature) for every top-level exported func."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"func ([A-Z]\w*)\(", line)
        if not m:
            continue
        text, j = line, i
        while body_start(text) < 0 and j + 1 < len(lines):
            j += 1
            text += " " + lines[j].strip()
        cut = body_start(text)
        sig = text[:cut].strip() if cut >= 0 else text.strip()
        sig = re.sub(r"\s+", " ", sig).replace("( ", "(").replace(", )", ")")
        yield i + 1, m.group(1), sig


def body_start(text):
    """Index of the `{` that opens the body, or -1: the first brace at
    parenthesis depth zero that follows a space (so `interface{}` in a
    parameter type is not mistaken for it)."""
    depth = 0
    for k, c in enumerate(text):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "{" and depth == 0 and k > 0 and text[k - 1] == " ":
            return k
    return -1


def results(sig):
    """The result list: whatever follows the parameter list's close paren."""
    depth = 0
    for k, c in enumerate(sig):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return sig[k + 1:].strip()
    return ""


def rows(skip=()):
    out = []
    for kind, pkg in (("cpu", CPU), ("gpu", GPU)):
        for name in go_files(pkg):
            if pkg + "/" + name in skip:
                continue
            path = os.path.join(ROOT, pkg, name)
            bc = constraint(path, name)
            for line, sym, sig in declarations(path):
                res = results(sig)
                if kind == "cpu":
                    keep = sym.startswith("Emit") and re.match(r"\(?\[\]byte\b", res)
                else:
                    keep = "*ir.Kernel" in res
                if keep:
                    out.append((kind, sym, pkg + "/" + name, line, bc, sig))
    return out


def csv_field(s):
    return '"' + s.replace('"', '""') + '"' if re.search(r'[,"\n]', s) else s


def emitter_slots():
    """Count cpu.Emitters' fields: generator slots, queries, flags."""
    src = open(os.path.join(ROOT, CPU, "emitters.go")).read()
    body = re.search(r"^type Emitters struct \{\n(.*?)^\}", src, re.S | re.M).group(1)
    gen, query, flag = [], [], []
    for line in body.split("\n"):
        line = line.split("//")[0].strip()
        m = re.match(r"(\w+)\s+(.*)", line)
        if not m:
            continue
        name, typ = m.groups()
        if typ.startswith("func"):
            (gen if "[]byte" in results(typ[4:]) else query).append(name)
        elif typ == "bool":
            flag.append(name)
    return gen, query, flag


def callers(symbols):
    """Non-test packages outside jit_old/gpu/kernels that name kernels.<sym> --
    a call or a function value; comments are stripped first."""
    found = defaultdict(set)
    pat = re.compile(r"\bkernels\.(" + "|".join(sorted(symbols, key=len, reverse=True)) + r")\b")
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("models", "node_modules")]
        rel = os.path.relpath(dirpath, ROOT)
        if rel == GPU:
            continue
        for f in files:
            if not f.endswith(".go") or f.endswith("_test.go"):
                continue
            with open(os.path.join(dirpath, f), errors="replace") as fh:
                code = re.sub(r"//.*", "", fh.read())
                for m in pat.finditer(code):
                    found[m.group(1)].add(rel)
    return found


def main():
    args, skip = sys.argv[1:], set()
    global ROOT
    if "--root" in args:
        i = args.index("--root")
        ROOT = os.path.abspath(args[i + 1])
        del args[i:i + 2]
    while "--skip" in args:
        i = args.index("--skip")
        skip.add(args[i + 1])
        del args[i:i + 2]
    rs = rows(skip)
    if "--summary" in args:
        cpu = [r for r in rs if r[0] == "cpu"]
        gpu = [r for r in rs if r[0] == "gpu"]
        gen, query, flag = emitter_slots()
        print(f"cpu.Emitters: {len(gen)} generator slots, {len(query)} queries {query}, flags {flag}")
        print(f"jit_old/cpu: {len(cpu)} declarations, {len(set(r[1] for r in cpu))} unique names")
        print(f"jit_old/gpu/kernels: {len(gpu)} constructors")
        calls = callers({r[1] for r in gpu})
        for r in gpu:
            print(f"  {r[1]:28} {r[2]}:{r[3]}  <- {', '.join(sorted(calls.get(r[1], ()))) or '(no non-test caller outside the package)'}")
        return
    # CRLF, as RFC 4180 has it and as the checked-in file always was.
    out = ["kind,symbol,file,line,build_constraint,signature"]
    out += [",".join(csv_field(str(x)) for x in r) for r in rs]
    sys.stdout.buffer.write(("\r\n".join(out) + "\r\n").encode())


if __name__ == "__main__":
    main()
