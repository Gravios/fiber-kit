"""Knobs, command-line option names, entry points, comments and small debts of fiber-kit."""
import ast, glob, sys, re, collections, os
R = sys.argv[1]; S = R + '/src/fiber_kit/'
src = sorted(glob.glob(S + '*.py')); T = {f: open(f).read() for f in src}
allsrc = '\n'.join(T.values()); runner = open(R + '/scripts/fiber-pipeline').read()
# --- knobs
def keys(p): return re.findall(r'^\s{0,4}#?\s?(FK_[A-Z0-9_]+|FIBER_[A-Z0-9_]+)\s*:', open(p).read(), re.M)
reg = {}
for y in ('fiber-kit-exp.yaml', 'fiber-kit.yaml', 'fiber-kit-best.yaml', 'fiber-kit-klusters.yaml'):
    reg[y] = set(keys(S + y)); print(f'{y}: {len(reg[y])} knob keys, {len(open(S + y).read().splitlines())} lines')
exp = reg['fiber-kit-exp.yaml']
insrc = set(re.findall(r'\b(FK_[A-Z0-9_]+|FIBER_[A-Z0-9_]+)\b', allsrc)); inrun = set(re.findall(r'\b(FK_[A-Z0-9_]+|FIBER_[A-Z0-9_]+)\b', runner))
print('knob names in src:', len(insrc), '| in the runner script:', len(inrun), '| in either:', len(insrc | inrun))
print('registered in exp, named nowhere in src or runner:', len(exp - insrc - inrun), sorted(exp - insrc - inrun)[:25])
unr = sorted(k for k in (insrc | inrun) - exp if not k.endswith('_'))
print('named in src or runner, not registered in exp:', len(unr), unr[:40])
pref = collections.Counter(re.match(r'(FK|FIBER)_([A-Z0-9]+)', k).group(0) for k in exp)
print('prefixes:', len(pref), pref.most_common(45))
print('ways a knob is read: os.environ.get', len(re.findall(r'os\.environ\.get\(', allsrc)), '| os.environ[', len(re.findall(r'os\.environ\[', allsrc)), '| os.getenv', len(re.findall(r'os\.getenv\(', allsrc)),
      '| cfg.knob / config helpers', {h: len(re.findall(r'\b' + h + r'\(', allsrc)) for h in ('env_int', 'env_float', 'env_str', 'env_bool', 'knob', 'cfg.get', 'config.get', '_env')})
# --- CLI option vocabulary
opts = collections.defaultdict(set)
for f, s in T.items():
    for o in re.findall(r'add_argument\(\s*["\'](--[a-z0-9-]+)["\']', s): opts[o].add(os.path.basename(f))
print('\ndistinct long options:', len(opts), '| add_argument calls:', len(re.findall(r'add_argument\(', allsrc)))
for group in (('--clu-stage', '--in-clu', '--in-stage', '--stage', '--in-tag', '--in'), ('--out-tag', '--out-stage', '--out-clu', '--out', '--tag'), ('--clu-method', '--method', '--spk-method', '--fet-method', '--variant'), ('--elec', '--group'), ('--dry-run', '--dry'), ('--jobs', '--threads', '--workers', '-j'), ('--seed',), ('--force', '--overwrite'), ('--verbose', '--quiet', '-v')):
    print('  ', {o: len(opts[o]) for o in group if o in opts})
# --- entry points
pp = open(R + '/pyproject.toml').read(); eps = re.findall(r'^([a-z_][a-z0-9_-]*)\s*=\s*"fiber_kit\.([a-z_0-9]+):', pp, re.M)
print('\nconsole entry points:', len(eps), '| modules with main():', sum('def main(' in s for s in T.values()), '| modules:', len(src))
epm = {m for _, m in eps}
print('modules with main() and no entry point:', sorted(os.path.basename(f)[:-3] for f, s in T.items() if 'def main(' in s and os.path.basename(f)[:-3] not in epm))
known = re.findall(r'fiber-[a-z-]+', re.search(r'unknown stage.*', runner).group(0)) if re.search(r'unknown stage', runner) else []
stages_md = open(R + '/docs/stages.md').read(); print('stage sections in docs/stages.md:', len(re.findall(r'^## ', stages_md, re.M)))
prefixes = collections.Counter(os.path.basename(f).split('_')[0] for f in src); print('module name prefixes:', prefixes.most_common())
# --- comments
tot = com = hist = long120 = long160 = 0; histf = collections.Counter(); doc = 0
pat = re.compile(r'used to|no longer|previously|formerly|was once|the old |before 0\.\d|since 0\.\d|\(0\.\d+\.\d+\)|\b0\.\d\d\.\d+\b|\b0[5-9]\d\d\b|20\d\d-\d\d-\d\d|this bug|regression', re.I)
for f, s in T.items():
    for l in s.split('\n'):
        tot += 1; st = l.strip(); long120 += len(l) > 120; long160 += len(l) > 160
        if st.startswith('#'):
            com += 1
            if pat.search(st): hist += 1; histf[os.path.basename(f)] += 1
    for n in ast.walk(ast.parse(s)):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and ast.get_docstring(n): doc += len(ast.get_docstring(n).split('\n'))
print(f'\nlines {tot} | # comment lines {com} ({com * 100 // tot}%) | docstring lines {doc} ({doc * 100 // tot}%) | comment lines that narrate history or cite a version / date / patch: {hist} in {len(histf)} files', histf.most_common(8))
print('lines over 120 columns:', long120, '| over 160:', long160)
for y in ('fiber-kit-exp.yaml', 'fiber-kit.yaml'):
    L = open(S + y).read().split('\n'); c = [l for l in L if l.strip().startswith('#')]; print(f'{y}: {len(c)} comment lines of {len(L)}; narrating history/version: {sum(bool(pat.search(l)) for l in c)}; longest line {max(map(len, L))}')
cl = open(R + '/CHANGELOG.md').read(); print('CHANGELOG entries:', len(re.findall(r'^## \[', cl, re.M)))
# --- debts
print('\nTODO/FIXME/XXX:', len(re.findall(r'\b(TODO|FIXME|XXX)\b', allsrc)), '| bare except:', len(re.findall(r'except\s*:', allsrc)), '| except Exception:', len(re.findall(r'except Exception', allsrc)), '| except Exception followed by pass:', len(re.findall(r'except Exception[^\n]*:\s*(#[^\n]*)?\n\s*pass\b', allsrc)))
print('print(', len(re.findall(r'\bprint\(', allsrc)), '| logging.', len(re.findall(r'\blogging\.|\blog\.(info|warning|debug|error)\(', allsrc)), '| sys.exit / SystemExit:', len(re.findall(r'sys\.exit\(|SystemExit\(', allsrc)), '| global statements:', len(re.findall(r'^\s+global \w', allsrc, re.M)))
print('functions with a docstring:', sum(bool(ast.get_docstring(n)) for s in T.values() for n in ast.walk(ast.parse(s)) if isinstance(n, ast.FunctionDef)), '| classes:', len(re.findall(r'^class \w', allsrc, re.M)))
print('lazy imports inside functions:', len(re.findall(r'^\s{4,}(import|from) \w', allsrc, re.M)))
