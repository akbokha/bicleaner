#!/usr/bin/env python

import argparse
import logging
import os
import sys
import subprocess
import math
import nltk
from heapq import heappush, heappop
from multiprocessing import Queue, Process, cpu_count
from tempfile import TemporaryFile, NamedTemporaryFile
from timeit import default_timer

import numpy as np
from mosestokenizer import MosesTokenizer
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from toolwrapper import ToolWrapper

nltk.download('punkt')

# Allows to load modules while inside or outside the package

# Allows to load modules while inside or outside the package
try:
    from .features import feature_extract, FEATURES_VERSION
    from .prob_dict import ProbabilisticDictionary
    from .util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from .training import shuffle, precision_recall, repr_right, write_metadata, train_fluency_filter
except (SystemError, ImportError):
    from features import feature_extract, FEATURES_VERSION
    from prob_dict import ProbabilisticDictionary
    from util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from training import shuffle, precision_recall, repr_right, write_metadata, train_fluency_filter

__author__ = "Sergio Ortiz-Rojas"
# Please, don't delete the previous descriptions. Just add new version description at the end.

__version__ = "Version 0.1 # December 2017 # Initial version # Sergio Ortiz-Rojas"
__version__ = "Version 0.2 # 09/01/2018 # Adding argument for injecting wrong examples from a file # Jorge Ferrández-Tordera"
__version__ = "Version 0.3 # 18/01/2019 # Integrated training of LM and refactor to avoid code duplicity # Víctor M. Sánchez-Cartagena"


# Argument parsing
def initialization():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="Tab-separated bilingual input file")

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-m', '--metadata', type=argparse.FileType('w'), required=True,
                        help="Training metadata (YAML file)")
    groupM.add_argument('-c', '--classifier', type=argparse.FileType('wb'), required=True, help="Classifier data file")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('-d', '--source_dictionary', type=argparse.FileType('r'), required=True,
                        help="LR gzipped probabilistic dictionary")
    groupM.add_argument('-D', '--target_dictionary', type=argparse.FileType('r'), required=True,
                        help="RL gzipped probabilistic dictionary")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-S', '--source_tokeniser_path', help="Source language tokeniser absolute path")
    groupO.add_argument('-T', '--target_tokeniser_path', help="Target language tokeniser absolute path")
    groupO.add_argument('--normalize_by_length', action='store_true', help="Normalize by length in qmax dict feature")
    groupO.add_argument('--treat_oovs', action='store_true', help="Special treatment for OOVs in qmax dict feature")
    groupO.add_argument('--qmax_limit', type=check_positive_or_zero, default=20,
                        help="Number of max target words to be taken into account, sorted by length")
    groupO.add_argument('--disable_features_quest', action='store_false', help="Disable less important features")
    groupO.add_argument('-g', '--good_examples', type=check_positive_or_zero, default=50000,
                        help="Number of good examples")
    groupO.add_argument('-w', '--wrong_examples', type=check_positive_or_zero, default=50000,
                        help="Number of wrong examples")
    groupO.add_argument('--good_test_examples', type=check_positive_or_zero, default=10000,
                        help="Number of good test examples")
    groupO.add_argument('--wrong_test_examples', type=check_positive_or_zero, default=10000,
                        help="Number of wrong test examples")
    groupO.add_argument('--classifier_type', choices=['svm', 'nn', 'nn1', 'adaboost', 'random_forest'],
                        default="random_forest", help="Classifier type")
    groupO.add_argument('--dump_features', type=argparse.FileType('w'), default=None,
                        help="Dump training features to file")
    groupO.add_argument('-b', '--block_size', type=check_positive, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=check_positive, default=max(1, cpu_count() - 1),
                        help="Number of process to use")
    groupO.add_argument('--wrong_examples_file', type=argparse.FileType('r'), default=None,
                        help="File with wrong examples extracted to replace the synthetic examples from method used by default")
    groupO.add_argument('--features_version', type=check_positive, default=FEATURES_VERSION,
                        help="Version of the features")
    groupO.add_argument('--gpu', default='0',
                        help="The GPUs (device-ids) that should be used for dcce-scoring and ced-scoring")

    # for dcce scoring
    groupO.add_argument('--dcce_model_src_trg', default=None,
                        help="Translation model (src-trg) used for dual-conditional cross-entropy scoring")
    groupO.add_argument('--dcce_model_trg_src', default=None,
                        help="Translation model (trg-src) used for dual-conditional cross-entropy scoring")
    groupO.add_argument('--dcce_src_vocab_src_trg', default=None,
                        help="Vocab (src-side) of MT model (src-trg) used for dual-conditional cross-entropy scoring")
    groupO.add_argument('--dcce_trg_vocab_src_trg', default=None,
                        help="Vocab (trg-side)of MT model (src-trg) used for dual-conditional cross-entropy scoring")
    groupO.add_argument('--dcce_src_vocab_trg_src', default=None,
                        help="Vocab (src-side) of MT model (trg-src) used for dual-conditional cross-entropy scoring")
    groupO.add_argument('--dcce_trg_vocab_trg_src', default=None,
                        help="Vocab (trg-side) of MT model (trg-src) used for dual-conditional cross-entropy scoring")

    # for ced scoring
    groupO.add_argument('--ced_src_model_id', default=None,
                        help="In-domain language model (src) used for cross-entropy difference filtering")
    groupO.add_argument('--ced_vocab_src_model_id', default=None,
                        help="Vocab of in-domain language model (src) used for cross-entropy difference filtering")
    groupO.add_argument('--ced_src_model_nd', default=None,
                        help="Non-domain specific language model (src) used for cross-entropy difference filtering")
    groupO.add_argument('--ced_vocab_src_model_nd', default=None,
                        help="Vocab of non-domain specific language model (src) used for cross-entropy diff filtering")
    groupO.add_argument('--ced_trg_model_id', default=None,
                        help="In-domain language model (trg) used for cross-entropy difference filtering")
    groupO.add_argument('--ced_vocab_trg_model_id', default=None,
                        help="Vocab of in-domain language model (trg) used for cross-entropy difference filtering")
    groupO.add_argument('--ced_trg_model_nd', default=None,
                        help="Non-domain specific language model (trg) used for cross-entropy difference filtering")
    groupO.add_argument('--ced_vocab_trg_model_nd', default=None,
                        help="Vocab of Non-domain specific language model (trg) used for cross-entropy diff filtering")
    groupO.add_argument('--ced_cut_off_value', type=float, default=0.0,
                        help="Non-domain specific language model (trg) used for cross-entropy difference filtering")

    # For LM filtering
    groupO.add_argument('--noisy_examples_file_sl', type=str,
                        help="File with noisy text in the SL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--noisy_examples_file_tl', type=str,
                        help="File with noisy text in the TL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--lm_dev_size', type=check_positive_or_zero, default=2000,
                        help="Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text.")
    groupO.add_argument('--lm_file_sl', type=str, help="SL language model output file.")
    groupO.add_argument('--lm_file_tl', type=str, help="TL language model output file.")
    groupO.add_argument('--lm_training_file_sl', type=str,
                        help="SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_training_file_tl', type=str,
                        help="TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_clean_examples_file_sl', type=str,
                        help="File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size.")
    groupO.add_argument('--lm_clean_examples_file_tl', type=str,
                        help="File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size.")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    args = parser.parse_args()
    # Logging
    logging_setup(args)
    return args


# Training function: receives two file descriptors, input and test, and a
# type classifiers and trains a classifier storing it in classifier_output
# and returns some quality estimates.
def train_classifier(input_features, test_features, classifier_type, classifier_output):
    feats = []
    labels = []

    # Load features and labels and format them as numpy array
    for line in input_features:
        parts = line.rstrip("\n").split("\t")
        feats.append([float(v) for v in parts[:-1]])
        labels.append(int(parts[-1]))

    dataset = dict()
    dataset['data'] = np.array(feats)
    dataset['target'] = np.array(labels)

    # Train classifier
    if classifier_type == "svm":
        clf = svm.SVC(gamma=0.001, C=100., probability=True)
    elif classifier_type == "nn":
        clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif classifier_type == "nn1":
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    elif classifier_type == "adaboost":
        clf = AdaBoostClassifier(n_estimators=100)
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                                     criterion='gini',
                                     max_depth=2,
                                     max_features='auto',
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     min_impurity_split=None,
                                     min_samples_leaf=1,
                                     min_samples_split=2,
                                     min_weight_fraction_leaf=0.0,
                                     n_estimators=200, n_jobs=-1,
                                     oob_score=False,
                                     random_state=0,
                                     verbose=0,
                                     warm_start=False)
    else:
        logging.error("Unknown classifier: " + classifier_type)
        sys.exit(1)

    clf.fit(dataset['data'], dataset['target'])

    joblib.dump(clf, classifier_output)

    feats = []
    labels = []

    for line in test_features:
        parts = line.rstrip("\n").split("\t")
        feats.append([float(v) for v in parts[:-1]])
        labels.append(int(parts[-1]))

    dataset = np.array(feats)
    prediction = clf.predict_proba(dataset)

    pos = 0
    good = []
    wrong = []
    for pred in prediction:
        if labels[pos] == 1:
            good.append(pred[1])
        else:
            wrong.append(pred[1])
        pos += 1

    hgood = np.histogram(good, bins=np.arange(0, 1.1, 0.1))
    hwrong = np.histogram(wrong, bins=np.arange(0, 1.1, 0.1))

    return hgood[0].tolist(), hwrong[0].tolist()


# Writes all features of the input TUs into a temporary file
def reduce_process(output_queue, output_file):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    output_file.write(i)
                filein.close()

            os.unlink(filein_name)

        job = output_queue.get()
        if job:
            nblock, filein_name = job
            heappush(h, (nblock, filein_name))
        else:
            logging.debug("Exiting reduce loop")
            break

    if len(h) > 0:
        logging.debug("Still elements in heap")

    while len(h) > 0 and h[0][0] == last_block:
        nblock, filein_name = heapq.heappop(h)
        last_block += 1

        with open(filein_name, 'r') as filein:
            for i in filein:
                output_file.write(i)
            filein.close()

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")
        sys.exit(0)

    output_file.close()


# Calculates all the features needed for the training
def worker_process(i, jobs_queue, output_queue, args, dcce_scores=None, ced_src_scores=None, ced_trg_scores=None):
    if args.source_tokeniser_path:
        source_tokeniser = ToolWrapper(args.source_tokeniser_path.split(' '))
    else:
        source_tokeniser = MosesTokenizer(args.source_lang)
    if args.target_tokeniser_path:
        target_tokeniser = ToolWrapper(args.target_tokeniser_path.split(' '))
    else:
        target_tokeniser = MosesTokenizer(args.target_lang)
    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {}".format(job.__repr__()))
            nblock, filein_name, label = job

            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False) as fileout:
                logging.debug("Filtering: creating temporary file {}".format(fileout.name))
                for i in filein:
                    srcsen, trgsen = i.split("\t")[:2]
                    trgsen = trgsen.strip()
                    #                    print(str(srcsen) + " --- " + str(trgsen))
                    features = feature_extract(srcsen, trgsen, source_tokeniser, target_tokeniser, args,
                                               dcce_scores, ced_src_scores, ced_trg_scores)

                    for j in features:
                        fileout.write("{}".format(j))
                        fileout.write("\t")
                    fileout.write("{}".format(label))
                    fileout.write("\n")
                ojob = (nblock, fileout.name)
                fileout.close()
                filein.close()
                output_queue.put(ojob)
            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker")
            source_tokeniser.close()
            target_tokeniser.close()
            break


# Divides the input among processors to speed up the throughput
def map_process(input, block_size, jobs_queue, label, first_block=0):
    logging.info("Start mapping")
    nblock = first_block
    nline = 0
    mytemp = None
    for line in input:
        if (nline % block_size) == 0:
            if mytemp:
                job = (nblock, mytemp.name, label)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False)
            logging.debug("Mapping: creating temporary file {}".format(mytemp.name))
        parts = line.split("\t")[0:2]
        if len(parts) >= 2:
            mytemp.write(line)
            nline += 1

    input.close()

    if nline > 0:
        job = (nblock, mytemp.name, label)
        mytemp.close()
        jobs_queue.put(job)

    return nblock


def calculate_dcce_scores(input_file, model_src_trg, model_trg_src, sv_src_trg, tv_src_trg, sv_trg_src, tv_trg_src, gpus):
    src_sentences = NamedTemporaryFile(mode="w", delete=False, encoding='utf-8')
    trg_sentences = NamedTemporaryFile(mode="w", delete=False, encoding='utf-8')

    sentences = list()
    dcce_scores = dict()

    with open(input_file.name) as input_f:
        input_f.seek(0)
        for line in input_f:
            # print('reading input')
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                # print('writing to tempfiles')
                src_sentences.write(parts[0] + '\n')
                trg_sentences.write(parts[1] + '\n')
                sentences.append((parts[0], parts[1]))

    src_sentences.seek(0)
    trg_sentences.seek(0)

    logging.info("DCCE scoring src-trg")

    src_trg_result = subprocess.run(
        ['./scripts/dcce_scoring.sh', model_src_trg, src_sentences.name, trg_sentences.name, sv_src_trg, tv_src_trg, gpus],
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    
    src_sentences.seek(0)
    trg_sentences.seek(0)

    logging.info("DCCE scoring trg-src")
    
    trg_src_result = subprocess.run(
        ['./scripts/dcce_scoring.sh', model_trg_src, trg_sentences.name, src_sentences.name, sv_trg_src, tv_trg_src, gpus],
        stdout=subprocess.PIPE).stdout.decode('utf-8')

    src_trg_scores = src_trg_result.splitlines()
    trg_src_scores = trg_src_result.splitlines()

    assert len(src_trg_scores) == len(trg_src_scores) == len(sentences)

    for sentence_pair, src_trg_score, trg_src_score in zip(sentences, src_trg_scores, trg_src_scores):
        hA, hB = abs(float(src_trg_score)), abs(float(trg_src_score))
        dcce_score = math.exp(-1.0 * (abs(hA - hB) + 0.5 * (hA + hB)))
        dcce_scores[sentence_pair] = dcce_score
    
    os.remove(src_sentences.name)
    os.remove(trg_sentences.name)
    
    return dcce_scores


def calculate_ced_scores(input_file, is_source, cut_off_value, model_id, model_nd, vocab_id, vocab_nd, gpus):
    sentences_file = NamedTemporaryFile(mode="w", delete=False, encoding='utf-8')
    sentences = list()
    ced_scores = dict()

    sentence_index = 0 if is_source else 1

    with open(input_file.name) as input_f:
        input_f.seek(0)
        for line in input_f:
            # print('reading input')
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                # print('writing to tempfiles')
                sentences_file.write(parts[sentence_index] + '\n')
                sentences.append(parts[sentence_index])

    sentences_file.seek(0)

    logging.info("CED scoring id-model")

    model_id_result = subprocess.run(
        ['./scripts/ced_scoring.sh', model_id, sentences_file.name, vocab_id, gpus],
        stdout=subprocess.PIPE).stdout.decode('utf-8')

    sentences_file.seek(0)

    logging.info("CED scoring nd-model")

    model_nd_result = subprocess.run(
        ['./scripts/ced_scoring.sh', model_nd, sentences_file.name, vocab_nd, gpus],
        stdout=subprocess.PIPE).stdout.decode('utf-8')

    id_scores = model_id_result.splitlines()
    nd_scores = model_nd_result.splitlines()

    assert len(id_scores) == len(nd_scores) == len(sentences)

    for sentence, id_score, nd_score in zip(sentences, id_scores, nd_scores):
        h_diff = (abs(float(id_score)) - abs(float(nd_score))) / len(nltk.word_tokenize(sentence))
        dom_exp = math.exp(-1.0 * h_diff)
        dom = min(dom_exp, 1.0)
        if dom < cut_off_value:
            dom = 0.0
        ced_scores[sentence] = dom

    return ced_scores


# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.info("Starting process")
    logging.info("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize=maxsize)
    worker_count = process_count

    # Read input to a named temporary file
    # We may need to read it multiple times and that would be problematic if it is sys.stdin
    input = NamedTemporaryFile(mode="w", delete=False)
    for line in args.input:
        input.write(line)
    input.close()

    stats = None
    with open(input.name) as input_f:
        args.input = input_f
        stats = train_fluency_filter(args)
        input_f.seek(0)

        # Shuffle and get length ratio
        total_size, length_ratio, good_sentences, wrong_sentences = shuffle(args.input,
                                                                            args.good_examples + args.good_test_examples,
                                                                            args.wrong_examples + args.wrong_test_examples,
                                                                           args.wrong_examples_file)

    dcce_scores = None
    ced_src_scores = None
    ced_trg_scores = None

    if args.dcce_model_src_trg and args.dcce_model_trg_src and\
            args.dcce_src_vocab_src_trg and args.dcce_trg_vocab_src_trg and\
            args.dcce_src_vocab_trg_src and args.dcce_trg_vocab_trg_src:

        dcce_scores = calculate_dcce_scores(args.input, args.dcce_model_src_trg, args.dcce_model_trg_src,
                                            args.dcce_src_vocab_src_trg, args.dcce_trg_vocab_src_trg,
                                            args.dcce_src_vocab_trg_src, args.dcce_trg_vocab_trg_src, args.gpu)

    if args.ced_src_model_id and args.ced_vocab_src_model_id and args.ced_src_model_nd and args.ced_vocab_src_model_nd:
        ced_src_scores = calculate_ced_scores(args.input, True, args.ced_cut_off_value,
                                              args.ced_src_model_id, args.ced_src_model_nd,
                                              args.ced_vocab_src_model_id, args.ced_vocab_src_model_nd, args.gpu)

    if args.ced_trg_model_id and args.ced_vocab_trg_model_id and args.ced_trg_model_nd and args.ced_vocab_trg_model_nd:
        ced_trg_scores = calculate_ced_scores(args.input, False, args.ced_cut_off_value,
                                              args.ced_trg_model_id, args.ced_trg_model_nd,
                                              args.ced_vocab_trg_model_id, args.ced_vocab_trg_model_nd, args.gpu)

    os.remove(input.name)

    args.length_ratio = length_ratio

    # Load dictionaries
    args.dict_sl_tl = ProbabilisticDictionary(args.source_dictionary)
    args.dict_tl_sl = ProbabilisticDictionary(args.target_dictionary)

    features_file = TemporaryFile('w+')
    # Start reducer
    reduce = Process(target=reduce_process,
                     args=(output_queue, features_file))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize=maxsize)
    workers = []
    for i in range(worker_count):
        worker = Process(target=worker_process,
                         args=(i, jobs_queue, output_queue, args, dcce_scores, ced_src_scores, ced_trg_scores))
        worker.daemon = True  # dies with the parent process
        worker.start()
        workers.append(worker)

    # Mapper process (foreground - parent)
    last_block = map_process(good_sentences, args.block_size, jobs_queue, 1, 0)
    good_sentences.close()

    map_process(wrong_sentences, args.block_size, jobs_queue, 0, last_block + 1)
    wrong_sentences.close()

    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    logging.info("End mapping")

    for w in workers:
        w.join()

    # Reducer termination
    output_queue.put(None)
    reduce.join()

    features_file.seek(0)

    if args.dump_features:
        logging.info("Dumping features to " + os.path.abspath(args.dump_features.name))
        for i in features_file:
            args.dump_features.write(i)
        args.dump_features.close()
        features_file.seek(0)

    logging.info("Start training")

    hgood = []
    hwrong = []
    with TemporaryFile("w+") as features_train, TemporaryFile("w+") as features_test:
        nline = 0
        for line in features_file:
            if nline < args.good_examples:
                features_train.write(line)
            elif nline < args.good_examples + args.good_test_examples:
                features_test.write(line)
            elif nline < args.good_examples + args.good_test_examples + args.wrong_examples:
                features_train.write(line)
            else:
                features_test.write(line)
            nline += 1

        features_train.flush()
        features_test.flush()

        features_train.seek(0)
        features_test.seek(0)
        hgood, hwrong = train_classifier(features_train, features_test, args.classifier_type, args.classifier)
        features_train.close()
        features_test.close()

    logging.info("End training")

    write_metadata(args, length_ratio, hgood, hwrong, stats)
    args.metadata.close()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {:.2f} s".format(elapsed_time))


# Main function: setup logging and calling the main loop
def main(args):
    # Parameter parsing
    #    args = initialization()

    # Filtering
    perform_training(args)


if __name__ == '__main__':
    args = initialization()
    main(args)
