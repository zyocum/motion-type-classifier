"""Module for working with MAE/MAI annotation documents."""

__author__ = "Zachary Yocum"
__email__  = "zyocum@brandeis.edu"

import codecs, os, json, re
from collections import OrderedDict
from itertools import chain
from string import *
from warnings import warn
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from nltk.tag.stanford import POSTagger
from progressbar import ProgressBar
from stanford import JAR, MODEL
from etymology import *
import word2vec

stanford_tagger = POSTagger(MODEL, JAR)

class Sentence(object):
    def __init__(self, document, tokens):
        self.document = document
        self.tokens = tokens
        self.words = [t.text for t in self.tokens]
        self.pos_tagged_tokens = self._pos_tag()
        self.pos_tags = [pos for _, pos in self.pos_tagged_tokens]
        try:
            self.start = int(self.tokens[0].attrib['begin'])
            self.end = int(self.tokens[-1].attrib['end'])
        except IndexError:
            print '!!! Error getting sentences from file {}'.format(
                self.document.file
            )
        self.motions = list(self._motions())
    
    def _motions(self):
        for motion in self.document.motion_tags:
            start, end = motion.attrib['start'], motion.attrib['end']
            if all([int(start) >= self.start, int(end) <= self.end]):
                yield Motion(self.document, self, **motion.attrib)
    
    def _pos_tag(self):
        pos_tagged = stanford_tagger.tag(self.words)[0]
        return pos_tagged

class Motion(dict):
    """docstring for Motion"""
    def __init__(self, document, sentence, **kwargs):
        super(Motion, self).__init__(**kwargs)
        self.document = document
        self.sentence = sentence
        self.start = int(self['start'])
        self.update({'start' : self.start})
        self.end = int(self['end'])
        self.update({'end' : self.end})
        self.partition = self.partition_sentence()
        self.index = self.partition[1][-1]
        self.update({'sentence_index' : self.index})
        self.update({'document' : self.document.file})
        for i in range(-5,6):
            self.update(self.pos(offset=i))
            self.update(self.word(offset=i))
            self.update(self.cluster(offset=i))
        self.update(self.languages())
    
    def _pos(self):
        _, center, _ = self.partition
        if center:
            _, pos = self.sentence.pos_tagged_tokens[center[-1]]
            return pos
        else:
            return ''
    
    def partition_sentence(self):
        left, center, right = [], [], []
        for i, token in enumerate(self.sentence.tokens):
            start, end = map(int, (token.attrib['begin'], token.attrib['end']))
            if end <= self.start:
                left.append(i)
            elif start >= self.end:
                right.append(i)
            else:
                center.append(i)
        return left, center, right
    
    def pos(self, offset=0):
        _, center, _ = self.partition
        if not center:
            pos = ''
        else:
            pos_tags = self.sentence.pos_tags
            bounds = range(len(pos_tags))
            target = center[-1] + offset
            if target in bounds:
                pos = pos_tags[target]
            else:
                pos = ''
        return {'pos[{}]'.format(offset) : pos }
    
    def word(self, offset=0):
        _, center, _ = self.partition
        if not center:
            word = ''
        else:
            words = self.sentence.words
            bounds = range(len(words))
            target = center[-1] + offset
            if target in bounds:
                word = words[target]
            else:
                word = ''
        return {'word[{}]'.format(offset) : word }
    
    def cluster(self, offset=0, filename='clusters.json'):
        with codecs.open(filename, mode='r', encoding='utf-8') as file:
            clusters = json.load(file)
        _, center, _ = self.partition
        if not center:
            cluster = 0
        else:
            words = self.sentence.words
            bounds = range(len(words))
            target = center[-1] + offset
            if target in bounds:
                key = words[target].strip(punctuation + whitespace).lower()
                cluster = clusters.get(key, 0)
            else:
                cluster = 0
        return {'cluster[{}]'.format(offset) : cluster }
    
    def languages(self):
        word = self['word[0]'].strip(punctuation + whitespace).lower()
        pos = self['pos[0]']
        results = languages(word, pos)
        return dict((l, 1) for l in results)

class Corpus(object):
    """A class for working with collections of Documents."""
    def __init__(
        self,
        directory,
        pattern='.*\.xml',
        recursive=True,
        n=None,
        loadfile=None,
        dumpfile=None
    ):
        self.directory = directory
        self.pattern = pattern
        self.recursive = recursive
        self.n = n
        self.loadfile = loadfile
        self.dumpfile = dumpfile
        self.files = list(
            find_files(self.directory, self.pattern, self.recursive)
        )[:self.n]
        self.documents = list(self._documents())[:n]
        self.tokens = OrderedDict(self._tokens())
        if loadfile and os.path.exists(loadfile):
            self.motions = load(self.loadfile)
        else:
            self.motions = list(self._motions())
        if self.dumpfile:
           self.dump(self.dumpfile)
    
    def __repr__(self):
        return '{name}({path})'.format(
            name=self.__class__.__name__,
            path=repr(self.directory)
        )
    
    def __iter__(self):
        return iter(self.documents)
    
    def _documents(self):
        print 'Loading documents for {}...'.format(self)
        with ProgressBar(maxval=len(self.files)) as progress:
            for i, file in enumerate(self.files):
                yield Document(file)
                progress.update(i+1)
    
    def _tokens(self):
        print 'Loading tokens...'
        for document in self.documents:
            for (start, end), token in document._tokens():
                yield (document, start, end), token
    
    def _motions(self):
        print 'Loading motions...'
        for document in self.documents:
            for sentence in document.sentences:
                for motion in sentence.motions:
                    yield motion
    
    def query(self, *tagtypes, **attributes):
        for document in self.documents:
            for tag in document.query(*tagtypes, **attributes):
                yield document, tag
    
    def dump(self, filename):
        with codecs.open(filename, mode='w', encoding='utf-8') as file:
            json.dump(
                self.motions,
                file,
                indent=True,
                ensure_ascii=False,
                sort_keys=True
            )
    
    def validate(self):
        return all(map(Document.validate, self.documents))

class Document(object):
    """A MAE/MAI annotation document."""
    def __init__(self, file):
        self.file = file
        self.tree = ElementTree.parse(file)
        self.root = self.tree.getroot()
        self.task = self.root.tag
        self.text = self.root.find('TEXT').text
        self.tags = self.root.find('TAGS').getchildren()
        self.motion_tags = [tag for tag in self.query('MOTION')]
        self.sentences = list(self._sentences())
        self.tag_types = set(tag.tag for tag in self.tags)
        self.extent_types = set(tag.tag for tag in self.extent_tags())
        self.link_types = set(tag.tag for tag in self.link_tags())
    
    def __repr__(self):
        return "Document({})".format(repr(self.file))
    
    def __str__(self):
        return self.text.encode('utf-8')
    
    def __hash__(self):
        return hash(self.text)
    
    def extent_tags(self):
        return (tag for tag in self.tags if 'start' in tag.attrib)
    
    def link_tags(self):
        return (tag for tag in self.tags if 'from' in tag.attrib)
    
    def consuming_tags(self):
        tags = self.extent_tags()
        return (tag for tag in tags if int(tag.attrib['start']) > -1)
    
    def non_consuming_tags(self):
        tags = self.extent_tags()
        return (tag for tag in tags if int(tag.attrib['start']) <= -1)
    
    def _sentences(self):
        sentences = self.root.find('TOKENS').getchildren()
        for sentence in sentences:
            yield Sentence(self, sentence.getchildren())
    
    def _tokens(self):
        for sentence in self.sentences:
            for token in sentence.tokens:
                start = int(token.attrib['begin'])
                end = int(token.attrib['end'])
                yield (start, end), unicode(token.text)
    
    def query(self, *tagtypes, **attributes):
        for tag in self.tags:
            if tag.tag in tagtypes:
                if all(
                    (tag.attrib[k] == v) for k, v in attributes.iteritems()
                ):
                    yield tag

def word_clusters(
    corpora,
    size=100,
    verbose=True,
    text='text.txt',
    phrases='phrases.txt',
    binary='text.bin',
    clusters='clusters.txt'
):
    words = []
    for corpus in corpora:
        for document in corpus.documents:
            for sentence in document.sentences:
                for word in sentence.words:
                    words.append(word.lower().strip(punctuation + whitespace))
    with codecs.open(text, mode='w', encoding='utf-8') as file:
        file.write(u' '.join(words))
    word2vec.word2phrase(text, phrases, verbose=verbose)
    word2vec.word2vec(phrases, binary, size=size, verbose=verbose)
    word2vec.word2clusters(text, clusters, size, verbose=verbose)
    json_clusters = clusters.rstrip('.txt') + '.json'
    with codecs.open(clusters, mode='r', encoding='utf-8') as file:
        d = dict(
            (w, int(c)) for w, c in map(split, file.read().splitlines())
        )
    with codecs.open(json_clusters, mode='w', encoding='utf-8') as file:
        json.dump(d, file, indent=True, ensure_ascii=False)
    return d

def load(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as file:
        return json.load(file)

def find_files(directory='.', pattern='.*', recursive=True):
    if recursive:
        return (os.path.join(directory, filename)
            for directory, subdirectories, filenames in os.walk(directory)
            for filename in filenames if re.match(pattern, filename))
    else:
        return (os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if re.match(pattern, filename))

def main():
    train = Corpus('data/train', dumpfile='train.motions.json')
    test = Corpus('data/test', dumpfile='test.motions.json')
    corpora = train, test
    clusters = word_clusters(corpora)

if __name__ == '__main__':
    main()
