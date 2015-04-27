import codecs, json, os, re
from collections import defaultdict
from string import *

entry = re.compile(
    r"""
    ^                    # start of line
    \s*                  # optional whitespace
    (<a href=")?         # anchor start
    (?P<url>(.+))?       # URL for term
    >?                   # href end
    <b>                  # bold start
    (?P<abbreviation>.+) # abbreviated term
    </b>                 # bold end
    (</a>)?              # anchor end
    [,\s]*               # optional comma or whitespace
    (?P<description>.+)  # description of the term
    \s*                  # optional trailing whitespace
    $                    # end of line
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

text = os.path.join('resources', 'abbreviation.txt')

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

#for abbreviation, d in terms.iteritems():
#    print abbreviation
#    print '\t' + '\n\t'.join(u'{} : {}'.format(*x) for x in d.iteritems())

jsonfile = os.path.join('resources', 'abbreviation.json')

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

def load(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as file:
        return json.load(file)

#dump(terms, jsonfile)

data = load(jsonfile)

additional_languages = [
    u'Old English',
    
]
LANGUAGES = set(
    k.strip(punctuation) for k, v in data.iteritems() if v['type'] == 'language'
)

def main():
    for language in sorted(LANGUAGES.keys()):
        print language
    print '# of languages : {}'.format(len(LANGUAGES))

if __name__ == '__main__':
    main()