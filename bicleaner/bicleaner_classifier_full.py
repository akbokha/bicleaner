#!/usr/bin/env python

import logging
import os
import sys
import math
import traceback
import subprocess

import numpy as np
import yaml
from mosestokenizer import MosesTokenizer
from sklearn.externals import joblib
from toolwrapper import ToolWrapper

# Allows to load modules while inside or outside the package
try:
    from .features import feature_extract, Features
    from .prob_dict import ProbabilisticDictionary
    from .lm import DualLMFluencyFilter, LMType, DualLMStats
    from .util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, \
        logging_setup
    from .bicleaner_hardrules import *

except (ImportError, SystemError):
    from features import feature_extract, Features
    from prob_dict import ProbabilisticDictionary
    from lm import DualLMFluencyFilter, LMType, DualLMStats
    from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, \
        logging_setup
    from bicleaner_hardrules import *

# import cProfile  # search for "profile" throughout the file

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 28/12/2017 # Initial release # Sergio Ortiz"
__version__ = "Version 0.8 # 25/05/2018 # Bicleaner + Hardrules integrated # Marta Bañón"
__version__ = "Version 0.9 # 27/09/2018 # Changed input parameters for feature_extract # Marta Bañón"
__version__ = "Version 0.9.1 # 03/10/2018 # YAML is mandatory # Marta Bañón"
__version__ = "Version 0.10.4 # 17/10/2018 # Default block size is now 200 # Marta Bañón"
__version__ = "Version 0.10.8 # 18/12/2018 # Generalized tokenizer # Leopoldo Pla"
__version__ = "Version 0.11.0 # 17/01/2019 # Added fluency filter # Víctor M. Sánchez-Cartagena"


# All the scripts should have an initialization according with the usage. Template:
def initialization():
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rt'), default=None,
                        help="Tab-separated files to be classified")
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="Output of the classification")
    parser.add_argument('metadata', type=argparse.FileType('r'), default=None, help="Training metadata (YAML file)")

    ## Parameters required
    # groupM = parser.add_argument_group('Mandatory')

    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument("-S", "--source_tokeniser_path", type=str,
                        help="Source language (SL) tokeniser executable absolute path")
    groupO.add_argument("-T", "--target_tokeniser_path", type=str,
                        help="Target language (TL) tokeniser executable absolute path")

    groupO.add_argument('--tmp_dir', default=gettempdir(),
                        help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-b', '--block_size', type=int, default=200, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count() - 1),
                        help="Number of processes to use")

    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None,
                        help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--threshold', type=check_positive_between_zero_and_one, default=0.5,
                        help="Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default.")
    groupO.add_argument('--lm_threshold', type=check_positive_between_zero_and_one, default=0.5,
                        help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set.")
    groupO.add_argument('--keep_lm_result', action='store_true',
                        help="Add an additional column to the results with the language model fluency score and do not discard any TU based on that score.")
    groupO.add_argument('--dom_threshold', default=None,
                        help="if specified, filter sentence pairs using the src and trg dom-score and this threshold")
    groupO.add_argument('--keep_dom_result', action='store_true',
                        help="Add an additional column to the results with the dom_score (src and trg) and do not discard any TU based on that score.")
    groupO.add_argument('--gpu', default='0',
                        help="The GPUs (device-ids) that should be used for dcce-scoring and ced-scoring")

    # for dcce scoring
    groupO.add_argument('--dcce_scores', type=argparse.FileType('rt'), default=None,
                        help="dcce scores")

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
    groupO.add_argument('--ced_src_scores', type=argparse.FileType('rt'), default=None,
                        help="ced source scores")
    groupO.add_argument('--ced_trg_scores', type=argparse.FileType('rt'), default=None,
                        help="ced target scores")

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

    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__,
                        help="show version of this script and exit")

    # Validating & parsing
    # Checking if metadata is specified
    args = parser.parse_args()
    logging_setup(args)

    try:
        yamlpath = os.path.dirname(os.path.abspath(args.metadata.name))

        metadata_yaml = yaml.load(args.metadata)

        args.source_lang = metadata_yaml["source_lang"]
        args.target_lang = metadata_yaml["target_lang"]
        if "source_tokeniser_path" in metadata_yaml:
            args.source_tokeniser_path = metadata_yaml["source_tokeniser_path"]
        if "target_tokeniser_path" in metadata_yaml:
            args.target_tokeniser_path = metadata_yaml["target_tokeniser_path"]

        try:
            args.clf = joblib.load(os.path.join(yamlpath, metadata_yaml["classifier"]))
        except:
            args.clf = joblib.load(metadata_yaml["classifier"])

        #        args.clf.n_jobs = None
        args.classifier_type = metadata_yaml["classifier_type"]

        try:
            args.dict_sl_tl = ProbabilisticDictionary(os.path.join(yamlpath, metadata_yaml["source_dictionary"]))
        except:
            args.dict_sl_tl = ProbabilisticDictionary(metadata_yaml["source_dictionary"])
        try:
            args.dict_tl_sl = ProbabilisticDictionary(os.path.join(yamlpath, metadata_yaml["target_dictionary"]))
        except:
            args.dict_tl_sl = ProbabilisticDictionary(metadata_yaml["target_dictionary"])

        args.normalize_by_length = metadata_yaml["normalize_by_length"]
        args.treat_oovs = metadata_yaml["treat_oovs"]
        args.qmax_limit = metadata_yaml["qmax_limit"]
        args.disable_features_quest = metadata_yaml["disable_features_quest"]
        args.good_examples = metadata_yaml["good_examples"]
        args.wrong_examples = metadata_yaml["wrong_examples"]
        args.good_test_examples = metadata_yaml["good_test_examples"]
        args.wrong_test_examples = metadata_yaml["wrong_test_examples"]
        args.length_ratio = metadata_yaml["length_ratio"]
        args.features_version = 1 if "features_version" not in metadata_yaml else int(metadata_yaml["features_version"])

        threshold = np.argmax(metadata_yaml["accuracy_histogram"]) * 0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold

        # Load LM stuff if model was trained with it
        if "source_lm" in metadata_yaml and "target_lm" in metadata_yaml:
            fullpath_source_lm = os.path.join(yamlpath, metadata_yaml['source_lm'])
            if os.path.isfile(fullpath_source_lm):
                args.source_lm = fullpath_source_lm
            else:
                args.source_lm = metadata_yaml['source_lm']

            fullpath_target_lm = os.path.join(yamlpath, metadata_yaml['target_lm'])
            if os.path.isfile(fullpath_target_lm):
                args.target_lm = fullpath_target_lm
            else:
                args.target_lm = metadata_yaml['target_lm']

            args.lm_type = LMType[metadata_yaml['lm_type']]
            stats = DualLMStats(metadata_yaml['clean_mean_perp'], metadata_yaml['clean_stddev_perp'],
                                metadata_yaml['noisy_mean_perp'], metadata_yaml['noisy_stddev_perp'])
            args.lm_filter_stats = stats
        else:
            args.source_lm = None
            args.target_lm = None
            args.lm_type = None
            args.lm_filter_stats = None

        logging.debug("YAML")
        logging.debug(metadata_yaml)
        parser.set_defaults(**metadata_yaml)

    except:
        print("Error loading metadata")
        traceback.print_exc()
        sys.exit(1)

    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args


# def profile_classifier_process(i, jobs_queue, output_queue,args):
#    cProfile.runctx('classifier_process(i, jobs_queue, output_queue, args)', globals(), locals(), 'profiling-{}.out'.format(i))
def classifier_process(i, jobs_queue, output_queue, args, dcce_scores=None, ced_src_scores=None, ced_trg_scores=None,
                       dom_src_scores=None, dom_trg_scores=None):
    if args.source_tokeniser_path:
        source_tokeniser = ToolWrapper(args.source_tokeniser_path.split(' '))
    else:
        source_tokeniser = MosesTokenizer(args.source_lang)
    if args.target_tokeniser_path:
        target_tokeniser = ToolWrapper(args.target_tokeniser_path.split(' '))
    else:
        target_tokeniser = MosesTokenizer(args.target_lang)

    # Load LM for fluency scoring
    lm_filter = None
    if args.source_lm and args.target_lm:
        lm_filter = DualLMFluencyFilter(args.lm_type, args.source_lang, args.target_lang)
        lm_filter.load(args.source_lm, args.target_lm, args.lm_filter_stats)

    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {0}".format(job.__repr__()))
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False,
                                                                      dir=args.tmp_dir) as fileout:
                logging.debug("Classification: creating temporary filename {0}".format(fileout.name))
                feats = []
                lm_scores = []
                dom_src_scrs = []
                dom_trg_scrs = []

                # Create the following arrays:
                # valid_sentences: boolean, length of input. States whether each sentence passed
                #  hard rules and lm fluency filtering
                # feats: vector of tuples, input features to the classifier, length equals number
                #  of sentences in the input that passed hard rules + lm fluency filtering

                valid_sentences = []
                for i in filein:
                    parts = i.split("\t")
                    sl_sentence = None
                    tl_sentence = None
                    if len(parts) >= 4:
                        sl_sentence = parts[2]
                        tl_sentence = parts[3]
                    if sl_sentence and tl_sentence and len(sl_sentence.strip()) != 0 and len(
                            tl_sentence.strip()) != 0 and wrong_tu(sl_sentence.strip(), tl_sentence.strip(),
                                                                   args) == False:
                        lm_score = None
                        if lm_filter:
                            lm_score = lm_filter.score(sl_sentence, tl_sentence)
                        if args.dom_threshold:
                            dom_src_score = float(dom_src_scores[sl_sentence.rstrip('\n')])
                            dom_trg_score = float(dom_trg_scores[tl_sentence.rstrip('\n')])

                        if args.keep_dom_result:
                            dom_src_scrs.append(dom_src_score)
                            dom_trg_scrs.append(dom_trg_score)

                        if lm_filter and lm_score < args.lm_threshold and not args.keep_lm_result:
                            valid_sentences.append(False)
                            lm_scores.append(lm_score)
                        elif args.dom_threshold and \
                                (dom_src_score < args.dom_threshold or dom_trg_score < args.dom_threshold):
                            valid_sentences.append(False)
                        else:
                            features = feature_extract(sl_sentence, tl_sentence, source_tokeniser, target_tokeniser,
                                                       args, dcce_scores, ced_src_scores, ced_trg_scores)
                            feats.append([float(v) for v in features])
                            lm_scores.append(lm_score)
                            valid_sentences.append(True)
                    else:
                        valid_sentences.append(False)

                predictions = args.clf.predict_proba(np.array(feats)) if len(feats) > 0 else []
                filein.seek(0)

                piter = iter(predictions)
                if lm_filter:
                    lmiter = iter(lm_scores)
                if args.keep_dom_result:
                    dom_src_iter = iter(dom_src_scrs)
                    dom_trg_iter = iter(dom_trg_scrs)
                for i, valid_sentence in zip(filein, valid_sentences):
                    if valid_sentence:
                        p = next(piter)

                        fileout.write(i.strip())
                        fileout.write("\t")
                        fileout.write(str(p[1]))
                        if lm_filter and args.keep_lm_result:
                            lm_score = next(lmiter)
                            fileout.write("\t")
                            fileout.write(str(lm_score))
                        if args.keep_dom_result:
                            dom_src = next(dom_src_iter)
                            dom_trg = next(dom_trg_iter)
                            fileout.write("\t")
                            fileout.write(str(dom_src))
                            fileout.write("\t")
                            fileout.write(str(dom_trg))
                        fileout.write("\n")
                    else:
                        fileout.write(i.strip("\n"))
                        fileout.write("\t0")
                        if lm_filter and args.keep_lm_result:
                            lm_score = next(lmiter)
                            fileout.write("\t")
                            fileout.write(str(lm_score))
                        if args.keep_dom_result:
                            dom_src = next(dom_src_iter)
                            dom_trg = next(dom_trg_iter)
                            fileout.write("\t")
                            fileout.write(str(dom_src))
                            fileout.write("\t")
                            fileout.write(str(dom_trg))
                        fileout.write("\n")

                ojob = (nblock, fileout.name)
                filein.close()
                fileout.close()

            if ojob:
                output_queue.put(ojob)

            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker")
            break


def mapping_process(args, jobs_queue):
    logging.info("Start mapping")
    nblock = 0
    nline = 0
    mytemp = None
    for line in args.input:
        if (nline % args.block_size) == 0:
            logging.debug("Creating block {}".format(nblock))
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        mytemp.write(line)

        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()
        jobs_queue.put(job)

    return nline


def reduce_process(output_queue, args):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {0}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    args.output.write(i)

                    if args.discarded_tus:
                        args.discarded_tus.write(i)
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

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Classification finished. Output available in {}".format(args.output.name))
    args.output.close()
    if args.discarded_tus:
        logging.info("Discarded TUs are available in {}".format(args.discarded_tus.name))
        args.discarded_tus.close()


def calculate_dcce_score(input_file, model_src_trg, model_trg_src, sv_src_trg, tv_src_trg, sv_trg_src, tv_trg_src,
                         gpus):
    src_sentences = NamedTemporaryFile(mode="w+t", delete=False, encoding='utf-8')
    trg_sentences = NamedTemporaryFile(mode="w+t", delete=False, encoding='utf-8')
    
    sentences = list()
    dcce_scores = dict()

    input_file.seek(0)

    for line in input_file:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            src_sentences.write(parts[2] + '\n')
            trg_sentences.write(parts[3] + '\n')
            sentences.append((parts[2], parts[3]))
    
    input_file.seek(0)
    src_sentences.seek(0)
    trg_sentences.seek(0)
   
    src_trg_result = subprocess.run(
        ['./scripts/dcce_scoring.sh', model_src_trg, src_sentences.name, trg_sentences.name, sv_src_trg, tv_src_trg,
         gpus], 
        stdout=subprocess.PIPE).stdout.decode('utf-8')
    src_sentences.seek(0)
    trg_sentences.seek(0)

    trg_src_result = subprocess.run(
        ['./scripts/dcce_scoring.sh', model_trg_src, trg_sentences.name, src_sentences.name, sv_trg_src, tv_trg_src,
         gpus],
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
    sentences_file = NamedTemporaryFile(mode="w+t", delete=False, encoding='utf-8')
    sentences = list()
    ced_scores = dict()
    dom_scores = dict()

    sentence_index = 2 if is_source else 3

    input_file.seek(0)

    for line in input_file:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            sentences_file.write(parts[sentence_index] + '\n')
            sentences.append(parts[sentence_index])

    input_file.seek(0)
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
        ced_scores[sentence] = h_diff
        dom_scores[sentence] = dom

    os.remove(sentences_file.name)

    return ced_scores, dom_scores


def extract_dcce_scores(input_file, scores_file):
    sentences = list()
    scores = list()

    input_file.seek(0)

    for line in input_file:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            sentences.append((parts[2], parts[3]))

    input_file.seek(0)

    scores_file.seek(0)

    for line in scores_file:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 1:
            scores.append(float(parts[0]))

    scores_file.seek(0)

    assert len(sentences) == len(scores)

    return dict(zip(sentences, scores))


def extract_ced_scores(input_file, scores_file, is_source):
    sentences = list()
    scores = list()

    sentence_index = 2 if is_source else 3

    input_file.seek(0)

    for line in input_file:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            sentences.append(parts[sentence_index])

    input_file.seek(0)

    scores_file.seek(0)

    for line in scores_file:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 1:
            scores.append(float(parts[0]))

    scores_file.seek(0)

    assert len(sentences) == len(scores)

    return dict(zip(sentences, scores))


# Filtering input texts
def perform_classification(args):
    time_start = default_timer()
    logging.info("Starting process")
    logging.info("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize=maxsize)
    worker_count = process_count

    dcce_scores = None
    ced_src_scores = None
    ced_trg_scores = None

    if args.dcce_scores:
        dcce_scores = extract_dcce_scores(args.input, args.dcce_scores)
    elif args.dcce_model_src_trg and args.dcce_model_trg_src and\
            args.dcce_src_vocab_src_trg and args.dcce_trg_vocab_src_trg and\
            args.dcce_src_vocab_trg_src and args.dcce_trg_vocab_trg_src:

        dcce_scores = calculate_dcce_score(args.input, args.dcce_model_src_trg, args.dcce_model_trg_src,
                                           args.dcce_src_vocab_src_trg, args.dcce_trg_vocab_src_trg,
                                           args.dcce_src_vocab_trg_src, args.dcce_trg_vocab_trg_src, args.gpu)

    if args.ced_src_scores:
        ced_src_scores = extract_ced_scores(args.input, args.ced_src_scores, True)
    elif args.ced_src_model_id and args.ced_vocab_src_model_id and args.ced_src_model_nd and args.ced_vocab_src_model_nd:
        ced_src_scores, dom_src_scores = calculate_ced_scores(args.input, True, args.ced_cut_off_value,
                                              args.ced_src_model_id, args.ced_src_model_nd,
                                              args.ced_vocab_src_model_id, args.ced_vocab_src_model_nd, args.gpu)

    if args.ced_trg_scores:
        ced_trg_scores = extract_ced_scores(args.input, args.ced_trg_scores, False)
    elif args.ced_trg_model_id and args.ced_vocab_trg_model_id and args.ced_trg_model_nd and args.ced_vocab_trg_model_nd:
        ced_trg_scores, dom_trg_scores = calculate_ced_scores(args.input, False, args.ced_cut_off_value,
                                              args.ced_trg_model_id, args.ced_trg_model_nd,
                                              args.ced_vocab_trg_model_id, args.ced_vocab_trg_model_nd, args.gpu)

    # Start reducer
    reduce = Process(target=reduce_process,
                     args=(output_queue, args))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize=maxsize)
    workers = []
    for i in range(worker_count):
        filter = Process(target=classifier_process,  # profile_classifier_process
                         args=(i, jobs_queue, output_queue, args, dcce_scores, ced_src_scores, ced_trg_scores,
                               dom_src_scores, dom_trg_scores))
        filter.daemon = True  # dies with the parent process

        filter.start()
        workers.append(filter)

    # Mapper process (foreground - parent)
    nline = mapping_process(args, jobs_queue)
    args.input.close()

    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    logging.info("End mapping")

    for w in workers:
        w.join()

    # Reducer termination
    output_queue.put(None)
    reduce.join()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline * 1.0) / elapsed_time)))


### END PARALLELIZATION METHODS ###

def main(args):
    logging.info("Executing main program...")
    perform_classification(args)
    logging.info("Program finished")


if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization()  # Parsing parameters
        main(args)  # Running main program
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
