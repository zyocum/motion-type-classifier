"""Module for constructing etymological dictionary from www.etymonline.com."""

__author__ = "Zachary Yocum"
__email__  = "zyocum@brandeis.edu"

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

def nest():
    """A recursive dictionary."""
    return defaultdict(nest)

def find_files(directory='.', pattern='.*', recursive=True):
    """Search recursively for files matching a pattern"""
    if recursive:
        return (os.path.join(directory, filename)
            for directory, subdirectories, filenames in os.walk(directory)
            for filename in filenames if re.match(pattern, filename))
    else:
        return (os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if re.match(pattern, filename))

class Entry(dict):
    """dict wrapper to represent an instance of an etymology entry."""
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
            sorted(LANGUAGES, key=len),
            set()
        )
        self['languages'] = sorted(self.languages)
    
    def __repr__(self):
        return json.dumps(self, indent=4, ensure_ascii=False)
    
    def parse(self, dt):
        """Parse the entry and update it."""
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
    """Find the set of all languages mentioned in text."""
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

LANGUAGES = load(os.path.join('resources', 'languages.json'))

def load_page(filename):
    """Parse an HTML page and get a list of entries from it."""
    with open(filename, 'r') as file:
        soup = BS(file)
        dl = soup.find('dl')   # list of terms and their definitions
        dts = dl.findAll('dt') # terms
        dds = dl.findAll('dd') # definitions
    entries = zip(dts, dds)
    return entries

def dump(data, filename):
    """Dump an object to file as JSON."""
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
    """Build the dictionary from the scraped list of pages."""
    etymologies = defaultdict(list)
    print('Building etymology dictionary...')
    with ProgressBar(maxval=len(pages)) as progress:
        for i, page in enumerate(pages):
            entries = (Entry(*e) for e in load_page(page))
            for entry in entries:
                etymologies[entry.text].insert(entry.index, entry)
            progress.update(i)
    return etymologies

def affixes(etymologies):
    """Generate entries for affixes only."""
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
    """Look up an etymology in the dictionary."""
    wn_pos = wordnet_pos(pos)
    try:
        lemma = wnl.lemmatize(query, wn_pos)
    except KeyError:
        lemma = query
    results = dictionary.get(lemma, [])
    for result in results:
        try:
            wn_pos_tags = map(wordnet_pos, (result['pos'] or '', wn_pos))
            match = re.match(*wn_pos_tags)
        except TypeError:
            #print 'query:', query
            #print 'pos:', pos
            #print 'wn_pos:', wn_pos
            #print 'lemma:', lemma
            #print 'wn_pos_tags:', wn_pos_tags
            match = False
        if match:
            return lemma, [result]
    return lemma, results

def lookup(*args):
    """Returns a structured result from a query term + pos + dictionary."""
    lemma, results = etym(*args)
    languages = nest()
    if not results:
        query, _, dictionary = args
        lemma, results = etym(query, None, dictionary)
    for result in results:
        languages[lemma][(unicode(result['pos']))] = result['languages']
    return languages

def wordnet_pos(pos):
    """Munge part-of-speech tags to be compatible with WordNet."""
    pos = (pos or '').strip(punctuation + whitespace).lower()
    if pos.startswith('j') or 'adj'.startswith(pos):
        return wordnet.ADJ
    elif pos.startswith('v'):
        return wordnet.VERB
    elif pos.startswith('n'):
        return wordnet.NOUN
    elif pos.startswith('r') or 'adv'.startswith(pos):
        return wordnet.ADV
    elif pos.startswith('in') or 'prep'.startswith(pos):
        return u'p'
    elif pos.startswith('fw'):
        return u'v'
    else:
        return None

site = os.path.expanduser(os.path.join('resources', 'www.etymonline.com'))

etymology_file = os.path.join('resources', 'etymology.json')

def setup():
    """Setup the etymological dictionary and store it as JSON."""
    if not os.path.isfile(etymology_file):
        page = re.compile(r'index.php\?l=\w+&p=\d+&allowed_in_frame=0.html')
        pages = list(find_files(directory=site, pattern=page, recursive=False))
        etymology = etymologies(pages)
        dump(etymology, etymology_file)
        for affix, dictionary in affixes(etymology):
            affix_file = os.path.join('resources', '{}.json'.format(affix))
            if not os.path.isfile(affix_file):
                dump(dictionary, affix_file)
setup()

ETYMOLOGY = load(etymology_file)

def languages(query, pos):
    """Look up a set of source languages for the query term."""
    lemma, results = etym(query, pos, ETYMOLOGY)
    languages = set()
    if not results:
        lemma, results = etym(query, None, ETYMOLOGY)
    for result in results:
        languages.update(set(result['languages']))
    return languages

def main(args):
    print(args)
    query = args.pop(0)
    if args:
        pos = args.pop(0)
    else:
        pos = ''
    print(
        json.dumps(
            languages(query, pos, ETYMOLOGY),
            indent=True,
            ensure_ascii=False
        )
    )

if __name__ == '__main__':
    main(sys.argv[1:])
