from __future__ import print_function

import codecs, os, json, re, sys, time
from collections import defaultdict
from operator import itemgetter
from string import *

from bs4 import BeautifulSoup as BS, CData
from bs4.element import Tag
from progressbar import ProgressBar, Percentage, Bar
from language import LANGUAGES
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wnl = WordNetLemmatizer()

term = re.compile(
    r"""
    ^                         # start of line
    \s*                       # optional whitespace
    (?P<term>.+?)             # the term
    \s*                       # optional whitespace
    (\((?P<pos>\w+\.)\)?)?    # optional part of speech
    (\(?(?P<index>[0-9]+)\))? # optional term (if term has multiple entries)
    \s*                       # optional whitespace
    $                         # end of line
    """,
    re.VERBOSE
)

prefix = re.compile(
    r"""
    ^                   # start of line
    (?P<prefix>[^-'].+) # prefix text
    [-']                # prefix end
    $                   # end of line
    """,
    re.VERBOSE
)


infix = re.compile(
    r"""
    ^             # start of line
    -             # infix start
    (?P<infix>.+) # infix text
    -             # infix end
    $             # end of line
    """,
    re.VERBOSE
)

suffix = re.compile(
    r"""
    ^                  # start of line
    [-']               # suffix start
    (?P<suffix>.+[^-]) # suffix text
    $                  # end of line
    """,
    re.VERBOSE
)

word = re.compile(
    r"""
    ^                      # start of line
    (?P<word>[^-'].+[^-']) # word text
    $                      # end of line
    """,
    re.VERBOSE
)

AFFIXES = prefix, infix, suffix, word

FIRST, SECOND, PENULTIMATE, ULTIMATE = map(itemgetter, range(2) + range(-2, 0))

def index(items):
    """Create an index/codebook that maps items to integers."""
    return dict((v, k) for k, v in enumerate(sorted(items)))

def nest():
    """A recursive dictionary."""
    return defaultdict(nest)

def find_files(directory='.', pattern='.*', recursive=True):
    if recursive:
        return (os.path.join(directory, filename)
            for directory, subdirectories, filenames in os.walk(directory)
            for filename in filenames if re.match(pattern, filename))
    else:
        return (os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if re.match(pattern, filename))

class Entry(dict):
    def __init__(self, term, definition):
        super(Entry, self).__init__()
        self.parse(term)
        self.definition = definition.text
        self['definition'] = self.definition
        self.text = self['term']
        self.pos = unicode(self['pos'] or '')
        self.prefix = self.get('prefix')
        self.infix = self.get('infix')
        self.suffix = self.get('suffix')
        self.word = self.get('word')
        self.key = u'/'.join([self.text, self.pos])
        self.languages = search(
            self.definition,
            sorted(LANGUAGES.keys(), key=len),
            set()
        )
        self['languages'] = sorted(self.languages)
    
    def __repr__(self):
        return json.dumps(self, indent=4, ensure_ascii=False)
    
    def parse(self, dt):
        entry = re.match(term, dt.text).groupdict()
        index = entry.pop('index')
        if not index:
            self.index = 0
        else:
            self.index = int(index) - 1
        for affix in AFFIXES:
            match = re.match(affix, entry['term'])
            if match:
                gd = match.groupdict()
                for k, v in gd.iteritems():
                    entry.update({k : v.strip()})
        self.update(entry)
    
def search(text, languages, results):
    if not languages:
        return results
    else:
        language = languages.pop()
        match = re.search(language, text)
        if match:
            results.add(language)
            text = u''.join(text.split(language))
        return search(text, languages, results)

def load(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as file:
        return json.load(file)

def load_page(filename):
    with open(filename, 'r') as file:
        soup = BS(file)
        dl = soup.find('dl')   # list of terms and their definitions
        dts = dl.findAll('dt') # terms
        dds = dl.findAll('dd') # definitions
    entries = zip(dts, dds)
    return entries

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

def etymologies(pages):
    etymologies = defaultdict(list)
    print('building etymology dictionary...')
    with ProgressBar(maxval=len(pages)) as progress:
        for i, page in enumerate(pages):
            entries = (Entry(*e) for e in load_page(page))
            for entry in entries:
                etymologies[entry.text].insert(entry.index, entry)
            progress.update(i)
    return etymologies

def affixes(etymologies):
    for affix in AFFIXES:
        name = FIRST(affix.groupindex.keys())
        print('building {} dictionary...'.format(name))
        dictionary = defaultdict(list)
        with ProgressBar(maxval=len(etymologies)) as progress:
            for i, key in enumerate(etymologies):
                for definition in etymologies[key]:
                    if definition.has_key(name):
                        dictionary[key].append(definition)
                progress.update(i)
        yield name, dictionary

def etym(query, pos, dictionary):
    pos = wordnet_pos(pos)
    lemma = wnl.lemmatize(query, pos)
    results = dictionary.get(lemma, [])
    for result in results:
        match = re.match(*map(wordnet_pos, (result['pos'], pos)))
        if match:
            return lemma, [result]
    return lemma, results

def wordnet_pos(pos):
    pos = (pos or '').strip(punctuation + whitespace).lower()
    if pos.startswith('j') or 'adjective'.startswith(pos):
        return wordnet.ADJ
    elif 'verb'.startswith(pos):
        return wordnet.VERB
    elif 'noun'.startswith(pos):
        return wordnet.NOUN
    elif pos.startswith('r') or 'adverb'.startswith(pos):
        return wordnet.ADV
    else:
        return None

etymology_file = os.path.join('resources', 'etymology.json')
site = os.path.expanduser(
    os.path.join('~', 'Downloads', 'www.etymonline.com')
)

def setup():
    if not os.path.isfile(etymology_file):
        page = re.compile(r'index.php\?l=\w+&p=\d+&allowed_in_frame=0.html')
        pages = list(find_files(directory=site, pattern=page, recursive=False))
        etymology = etymologies(pages)
        dump(etymology, etymology_file)
        for affix, dictionary in affixes(etymology):
            affix_file = os.path.join('resources', '{}.json'.format(affix))
            if not os.path.isfile(affix_file):
                dump(dictionary, affix_file)

def main(args):
    print(args)
    setup()
    etymology = load(etymology_file)
    query = args.pop(0)
    if args:
        pos = args.pop(0)
    else:
        pos = 'n'
    lemma, results = etym(query, pos, etymology)
    for result in results:
        print(lemma, ':', json.dumps(result, indent=True, ensure_ascii=False))

if __name__ == '__main__':
    main(sys.argv[1:])
