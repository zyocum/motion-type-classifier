import codecs, json, os, re
from collections import defaultdict

def nest():
    return defaultdict(nest)

entry = re.compile(
    r"""
    (<a href=")?            # anchor markup start
    (?P<url>(.+))?          # URL for term
    >?<b>                   # bold markup start
    (?P<abbreviation>.+)    # abbreviated term
    </b>                    # bold markup end
    (</a>)?                 # anchor markup end
    [,\s]*                  # comma or whitespace
    (?P<description>.+)     # description of the term
    """,
    re.VERBOSE
 )

term = re.compile(
    r"""
    <i>           # italic markup start
    (?P<term>.+?) # unabbreviated full term
    </i>          # italic markup end
    """,
    re.VERBOSE
)

text = os.path.join('resources', 'abbreviations.txt')

with codecs.open(text, mode='r', encoding='utf-8') as file:
    lines = unicode(file.read()).splitlines()

entries = (re.match(entry, line).groupdict() for line in lines)

abbreviations = dict((e.get('abbreviation'), e) for e in entries)

terms = dict()

for a, d in abbreviations.iteritems():
    d['terms'] = map(unicode, [a] + re.findall(term, d['description']))
    d['type'] = None
    for t in d['terms']:
        terms[t] = d

print '# of terms : {}'.format(len(terms))
#for abbreviation, d in terms.iteritems():
#    print abbreviation
#    print '\t' + '\n\t'.join(u'{} : {}'.format(*x) for x in d.iteritems())

jsonfile = os.path.join('resources', 'abbreviations.json')

def dump(data, filename):
    with codecs.open(filename, mode='w', encoding='utf-8') as file:
        json.dump(
            data,
            file,
            encoding='utf-8',
            ensure_ascii=False,
            sort_keys=True,
            indent=4
        )

#dump(terms, jsonfile)

languages = dict(
    (k, v) for k, v in terms.iteritems() if v['type'] == 'language'
)
for language in sorted(languages.keys()):
    print language