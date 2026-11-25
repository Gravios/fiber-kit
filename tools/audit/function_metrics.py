"""Function metrics of fiber-kit's Python with lizard: length, cyclomatic complexity, parameters."""
import glob, sys, json, lizard
root = sys.argv[1]; files = sorted(glob.glob(root + '/src/fiber_kit/*.py'))
fns = []
for f in files:
    r = lizard.analyze_file(f)
    for fn in r.function_list:
        fns.append(dict(file=f.split('/')[-1], name=fn.name, nloc=fn.nloc, ccn=fn.cyclomatic_complexity, params=len(fn.parameters), line=fn.start_line, length=fn.end_line - fn.start_line + 1))
json.dump(fns, open(sys.argv[2], 'w'))
n = len(fns); L = sorted(f['length'] for f in fns)
print('files', len(files), '| functions', n, '| median length', L[n // 2], '| 90th', L[int(n * .9)], '| 99th', L[int(n * .99)])
for t in (50, 100, 200, 400): print(f'  longer than {t} lines: {sum(l > t for l in L)}  ({sum(l for l in L if l > t) * 100 // sum(L)}% of function lines)')
print('  cyclomatic complexity > 15:', sum(f['ccn'] > 15 for f in fns), '| > 30:', sum(f['ccn'] > 30 for f in fns), '| > 60:', sum(f['ccn'] > 60 for f in fns))
print('  more than 6 parameters:', sum(f['params'] > 6 for f in fns), '| more than 12:', sum(f['params'] > 12 for f in fns))
print('\nlongest:')
for f in sorted(fns, key=lambda f: -f['length'])[:25]: print(f"  {f['length']:5d} lines  ccn {f['ccn']:4d}  {f['file']}:{f['line']}  {f['name']}")
print('\nmost parameters:')
for f in sorted(fns, key=lambda f: -f['params'])[:10]: print(f"  {f['params']:3d}  {f['file']}:{f['line']}  {f['name']}")
mains = [f for f in fns if f['name'] == 'main']
print('\nmain() functions:', len(mains), '| total lines', sum(f['length'] for f in mains), f"({sum(f['length'] for f in mains) * 100 // sum(L)}% of function lines)", '| over 200 lines:', sum(f['length'] > 200 for f in mains))
