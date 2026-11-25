"""Textual candidates: functions whose name appears nowhere else, function bodies that occur more than once, modules nothing imports."""
import ast, glob, sys, re, collections, hashlib, os
root = sys.argv[1]
src = sorted(glob.glob(root + '/src/fiber_kit/*.py'))
everything = src + glob.glob(root + '/test/*.py') + glob.glob(root + '/tools/*.py') + glob.glob(root + '/scripts/*') + [root + '/pyproject.toml'] + glob.glob(root + '/plans/**/*', recursive=True)
text = {f: open(f, errors='replace').read() for f in everything if os.path.isfile(f)}
words = collections.Counter()
for f, s in text.items(): words.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', s))
defs = []; bodies = collections.defaultdict(list)
class Norm(ast.NodeTransformer):
    def visit_Name(self, n): return ast.copy_location(ast.Name(id='_', ctx=n.ctx), n)
    def visit_arg(self, n): n.arg = '_'; n.annotation = None; return n
for f in src:
    t = ast.parse(text[f])
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ln = n.end_lineno - n.lineno + 1
            defs.append((f.split('/')[-1], n.name, n.lineno, ln))
            body = [b for b in n.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, 'value', None), ast.Constant))]
            if ln >= 6 and body:
                key = hashlib.md5(''.join(ast.dump(b) for b in body).encode()).hexdigest()
                bodies[key].append((f.split('/')[-1], n.name, n.lineno, ln))
dead = [d for d in defs if words[d[1]] == 1 and not d[1].startswith('__') and not d[1].startswith('visit_')]
print('functions defined:', len(defs))
print('name appears once in src, test, tools, scripts, plans, pyproject:', len(dead), '| lines', sum(d[3] for d in dead), '| of 4 lines or more:', sum(d[3] >= 4 for d in dead))
for d in sorted(dead, key=lambda d: -d[3])[:40]: print(f'  {d[3]:4d}  {d[0]}:{d[2]}  {d[1]}')
onlytest = []
srcwords = collections.Counter()
for f in src: srcwords.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text[f]))
scr = collections.Counter()
for f in text:
    if '/scripts/' in f or f.endswith('pyproject.toml'): scr.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text[f]))
onlytest = [d for d in defs if srcwords[d[1]] == 1 and words[d[1]] > 1 and scr[d[1]] == 0 and d[1] != 'main']
print('\nused only by tests or tools (one mention in src):', len(onlytest), '| lines', sum(d[3] for d in onlytest))
for d in sorted(onlytest, key=lambda d: -d[3])[:15]: print(f'  {d[3]:4d}  {d[0]}:{d[2]}  {d[1]}')
dup = [v for v in bodies.values() if len(v) > 1]
print('\nidentical bodies (6 lines or more, literal-identical):', len(dup), 'groups |', sum(len(v) - 1 for v in dup), 'extra copies |', sum(sum(x[3] for x in v[1:]) for v in dup), 'lines')
for v in sorted(dup, key=lambda v: -v[0][3] * (len(v) - 1))[:30]: print('  ', v[0][3], 'lines:', ', '.join(f'{x[0]}:{x[2]} {x[1]}' for x in v))
names = collections.defaultdict(list)
for d in defs:
    if d[3] >= 6: names[d[1]].append(d)
same = {k: v for k, v in names.items() if len({x[0] for x in v}) > 1 and k not in ('main', '__init__')}
print('\nsame function name defined in more than one module (6+ lines):', len(same))
for k, v in sorted(same.items(), key=lambda kv: -len(kv[1]))[:30]: print(f'   {k}: ' + ', '.join(f'{x[0]}({x[3]})' for x in v))
mods = [os.path.basename(f)[:-3] for f in src]
imp = collections.Counter()
for f, s in text.items():
    for m in mods:
        if os.path.basename(f)[:-3] != m and re.search(r'\b' + m + r'\b', s): imp[m] += 1
print('\nmodules no other file mentions:', [m for m in mods if imp[m] == 0])
tests = ''.join(s for f, s in text.items() if '/test/' in f)
print('modules no test mentions:', len([m for m in mods if not re.search(r'\b' + m + r'\b', tests)]), [m for m in mods if not re.search(r'\b' + m + r'\b', tests)])
