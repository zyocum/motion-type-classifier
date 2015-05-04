"""Module for evaluating (PATH|MANNER|COMPOUND) classifier."""

__author__ = "Zachary Yocum"
__email__  = "zyocum@brandeis.edu"

import os, warnings

from corpus import *

import numpy as np
from collections import Counter
from scipy.stats.stats import pearsonr
from sklearn import svm, metrics, cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

FEATURE_SETS = load(os.path.join('resources', 'features.json'))
SCORES = OrderedDict.fromkeys([
    'precision_weighted',
    'recall_weighted',
    'f1_weighted'
])

WIDTH = max(map(len, SCORES.keys()))
HEADER = '{{:>{}}}'.format(WIDTH)
SCORE = '{{:>{}.3f}}'.format(WIDTH)

class Table(object):
    """A class for working with tabular data."""
    def __init__(
        self,
        data,
        align='>',
        rowdelim='\n',
        coldelim='\t'
    ):
        self.data = map(list, data)
        self.shape = self.dimensions()
        self.height, self.width = self.shape
        self.align = align
        self.rowdelim = rowdelim
        self.coldelim = coldelim
        self.header = self.data[0]
        self.rows = self.data[1:]
        self.columns = zip(*self.data)
    
    def __repr__(self):
        return '<Table rows={} x cols={}>'.format(*self.shape)
    
    def __str__(self):
        table = []
        widths = self.widths()
        for row in self.data:
            entry = []
            for col, datum in enumerate(row):
                cell = u'{{:{}{}}}'.format(self.align, widths[col])
                entry.append(cell.format(datum))
            table.append(entry)
        return self.rowdelim.join((self.coldelim.join(r) for r in table))
                
    
    def dimensions(self):
        width = max(map(len, self.data))
        for i, row in enumerate(self.data):
            row_width = len(row)
            message = 'Row {} has length {} (should be {})'.format(
                i,
                row_width,
                width
            )
            assert(row_width == width), message
        height = len(self.data)
        return height, width
    
    def widths(self):
        widths = [max(map(len, column)) for column in self.columns]
        return widths

def tabulate(args, cell=SCORE, delimiter='\t'):
    return delimiter.join(cell.format(arg).replace('_', ' ') for arg in args)

def scorer(*args):
    for key in SCORES:
        SCORES[key] = metrics.SCORERS[key](*args)
    print tabulate(SCORES.values())
    return SCORES[SCORES.keys()[-1]]

langs = {
    u'American English',
    u'Anglo-French',
    u'Celtic',
    u'Danish',
    u'Dutch',
    u'English',
    u'Frankish',
    u'French',
    u'Frisian',
    u'Gaelic',
    u'Gaulish',
    u'German',
    u'Germanic',
    u'Gothic',
    u'Greek',
    u'I.E.',
    u'Irish',
    u'Italian',
    u'Late Latin',
    u'Latin',
    u'Lithuanian',
    u'Low German',
    u'Middle Dutch',
    u'Middle English',
    u'Middle French',
    u'Middle Low German',
    u'Modern English',
    u'Old Church Slavonic',
    u'Old English',
    u'Old French',
    u'Old Frisian',
    u'Old High German',
    u'Old Irish',
    u'Old Norse',
    u'Old Saxon',
    u'PIE',
    u'Proto-Germanic',
    u'Russian',
    u'Sanskrit',
    u'Scandinavian',
    u'Scot.',
    u'Scottish',
    u'Spanish',
    u'Swedish',
    u'Vulgar Latin',
    u'West Frisian',
    u'West Germanic'
}

lemmas = {
    u'abandon',
    u'approach',
    u'arrive',
    u'avoid',
    u'bicycle',
    u'bike',
    u'biking',
    u'bring',
    u'clear',
    u'climb',
    u'come',
    u'connect',
    u'continue',
    u'cross',
    u'cycle',
    u'dance',
    u'descend',
    u'detour',
    u'drive',
    u'encounter',
    u'enter',
    u'entering',
    u'entry',
    u'find',
    u'follow',
    u'gather',
    u'go',
    u'head',
    u'hike',
    u'hiking',
    u'join',
    u'lead',
    u'leave',
    u'locate',
    u'loop',
    u'meet',
    u'move',
    u'park',
    u'pass',
    u'passing',
    u'pitch',
    u'reach',
    u'reduce',
    u'remove',
    u'return',
    u'rid',
    u'ride',
    u'rise',
    u'rout',
    u'run',
    u'rush',
    u'search',
    u'slow',
    u'split',
    u'stop',
    u'swimming',
    u'take',
    u'throw',
    u'travel',
    u'traverse',
    u'trip',
    u'turn',
    u'use',
    u'visit',
    u'walk',
    u'way'
}

def main(features):
    train_data = load('train.motions.json')
    test_data = load('test.motions.json')
    all_data = train_data + test_data
    for datum in all_data:
        for feature in datum.keys():
            selected = False
            for selected_feature in features:
                if feature.startswith(selected_feature):
                   selected = True 
            if not selected:
                datum.pop(feature)
    all_samples = [td.pop('motion_type') for td in all_data]
    train, test = all_data[:len(train_data)], all_data[len(train_data):]
    train_samples = all_samples[:len(train_data)]
    test_samples = all_samples[len(train_data):]
    vectorizer = DictVectorizer()
    features = vectorizer.fit_transform(all_data).toarray()
    print 'Features vector shape (rows x columns):', '{} x {}'.format(*features.shape)
    train_features = features[:len(train_data)]
    test_features = features[len(train_data):]
    
    c = 1.0
    iterations = 10 ** 6
    
    weights = {
        'MANNER' : c / 3,
        'PATH' : c / 3,
        'COMPOUND' : c / 3
    }
    
    lin_svc = svm.LinearSVC(
        tol=10 ** -6,
        dual=False,
        multi_class='crammer_singer',
        max_iter=iterations,
        class_weight=weights,
        C=c
    )
    
    degree = 2
    poly_svc = svm.SVC(
        kernel='poly',
        degree=degree,
        C=c
    )
    
    logistic_regression = LogisticRegression(
        multi_class='multinomial',
        max_iter=iterations,
        solver='lbfgs',
        class_weight=weights,
        dual=False,
        C=c
    )
    
    models = OrderedDict({
        #'Degree {} Polynomial SVM'.format(degree) : poly_svc,
        #'Logistic Regression' : logistic_regression,
        'Linear SVM' : lin_svc,
    })

    fold = 10
    for label, model in models.iteritems():
        print 'Model={}'.format(label)
        print 'Train:Test={}:{}'.format(*map(len, (train, test)))
        model.fit(train_features, train_samples)
        
        reference = test_samples
        predicted = model.predict(test_features)
        
        labels = list(set(train_samples + test_samples))
        
        print metrics.classification_report(
            reference,
            predicted,
            target_names=labels
        )
        
################################################################################
# Cross Validation
################################################################################
        #print 'Train:Test={}-fold cross-validation'.format(label, fold)
        #model.fit(features, all_samples)
        #print tabulate(SCORES.keys(), cell=HEADER)
        #scores = cross_validation.cross_val_score(
        #    model,
        #    features,
        #    all_samples,
        #    cv=fold,
        #    scoring=scorer
        #)
        
################################################################################
# Error Analysis
################################################################################

        #motions = []
        #for datum in test_data:
        #    text = []
        #    text.append(datum.get('word[-5]', u''))
        #    text.append(datum.get('word[-4]', u''))
        #    text.append(datum.get('word[-3]', u''))
        #    text.append(datum.get('word[-2]', u''))
        #    text.append(datum.get('word[-1]', u''))
        #    text.append('[{}]'.format(datum.get('word[0]', u'')))
        #    text.append(datum.get('word[1]', u''))
        #    text.append(datum.get('word[2]', u''))
        #    text.append(datum.get('word[3]', u''))
        #    text.append(datum.get('word[4]', u''))
        #    text.append(datum.get('word[5]', u''))
        #    motions.append(u' '.join(filter(None, text)))
        #
        #header = ('text', 'reference', 'predicted')
        #comparisons = zip(motions, test_samples, predicted)
        #label_width = max(map(len, set(list(test_samples) + list(predicted))))
        #motion_width = max(map(len, motions))
        #for comparison in comparisons:
        #    motion, ref, pred = comparison
        #    motion = u'{{:<{}}}'.format(motion_width).format(motion)
        #    ref = u'{{:<{}}}'.format(label_width).format(ref)
        #    pred = u'{{:<{}}}'.format(label_width).format(pred)
        #    print u'\t'.join((motion, ref, pred))

################################################################################
# Sparse Feature Table
################################################################################
        
        #label_dict = OrderedDict(
        #    (v,k) for k,v in enumerate(sorted(set(test_samples)))
        #)
        #lang_dict = OrderedDict(
        #    (v,k) for k,v in enumerate(sorted(langs))
        #)
        #
        #a = np.zeros((len(lang_dict), len(label_dict)))
        #lines = []
        #for datum, label in zip(test_data, test_samples):
        #    line = []
        #    line.append(datum.get('word[0]') or u'!!!')
        #    line.append(datum.get('pos[0]') or u'!!!')
        #    line.append(datum.get('lemma') or u'!!!')
        #    for lang in sorted(langs):
        #        if datum.get(lang):
        #            line.append(u'+')
        #            row = lang_dict[lang]
        #            col = label_dict[label]
        #            a[row,col] += 1
        #        else:
        #            line.append(u'')
        #    line.append(label)
        #    lines.append(line)
        ##header = ['Word', 'POS', 'Lemma'] + sorted(langs) + ['Reference Label']
        ##print u' & ' .join(header)
        ##t = Table(lines, rowdelim='\\\\\n', coldelim=u' & ', align='')
        ##print repr(t)
        ##print t
        #print u',' + u','.join(sorted(set(test_samples)))
        #for i, row in enumerate(a):
        #    print u','.join(map(str, [lang_dict.keys()[i]] + np.divide(a[i], a[i].sum()).tolist()))

################################################################################
# Correlation Coefficients
################################################################################
        #label_dict = OrderedDict.fromkeys(sorted(set(test_samples)))
        #for key in label_dict:
        #    label_dict[key] = []
        #
        #lang_dict = OrderedDict.fromkeys(sorted(langs))
        #for key in lang_dict:
        #    lang_dict[key] = []
        #
        #for datum, sample in zip(test_data, test_samples):
        #    for lang in lang_dict.keys():
        #        lang_dict[lang].append(datum.get(lang, 0))
        #    for label in label_dict.keys():
        #        label_dict[label].append(sample==label)
        #
        #a = np.zeros((len(lang_dict), len(label_dict)))
        #for i, lang in enumerate(lang_dict.keys()):
        #    for j, label in enumerate(label_dict.keys()):
        #        x, y = (lang_dict[lang],label_dict[label])
        #        correlation =  np.corrcoef(x, y)[0, 1]
        #        a[i,j] = correlation
        #print u',' + u','.join(sorted(set(test_samples)))
        #for i, row in enumerate(a):
        #    coefs = [u'{:1.2f}'.format(f) for f in a[i].tolist()]
        #    print u','.join(map(str, [lang_dict.keys()[i]] + coefs))


################################################################################
# Correlation Coefficients
################################################################################

    #all_counts = Counter(all_samples)
    #train_counts = Counter(train_samples)
    #test_counts = Counter(test_samples)
    #for counts in (train_counts, test_counts, all_counts):
    #    print counts, sum(counts.values())

if __name__ == '__main__':
    for label in sorted(FEATURE_SETS.keys(), key=len):
        features = FEATURE_SETS[label]
        print '-' * 80
        print 'Features={}'.format(label)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(features)
