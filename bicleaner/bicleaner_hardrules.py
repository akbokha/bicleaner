#!/usr/bin/env python

import argparse
import io
import logging
import os
import sys
import traceback
from heapq import heappush, heappop
from multiprocessing import Queue, Process, cpu_count
from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer

import Levenshtein
import nltk
import pycld2
import regex

# Allows to load modules while inside or outside the package
try:
    from .util import logging_setup
except (SystemError, ImportError):
    from util import logging_setup

regex_blank = regex.compile("[ \u00A0]")
regex_digit = regex.compile("[[:digit:]]")
regex_alpha = regex.compile("[[:alpha:]]")
regex_url = regex.compile(
    '((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\((:?[^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
regex_email_address = regex.compile('[^@]+@[^@]+\.[^@]+')
regex_breadcrumbs = regex.compile("([ ][-/»][ ]|[|<>→←]|[ ][:][:][ ])")
regex_unicode_noise = regex.compile("[\x80-\xFF]{3,}")
regex_spaces_noise = regex.compile("([ ].){4,}[ ]")
regex_paren = regex.compile("[][(){}]")
regex_unwanted = regex.compile("[+*]")
regex_inconditional = regex.compile("=\"")
regex_escaped_unicode = regex.compile("[\\\\]u[0-9a-fA-F]{3,}")
safe_noise_detection_langs = {"en", "es", "fr", "pl", "de", "it", "pt", "nl", "cs", "ro", "fi", "lv", "et", "bg", "hr",
                              "da", "hu", "ga", "eu", "gl", "sl", "sv", "mt", "sk"}


def initialization():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('input', nargs='?', type=argparse.FileType('rt', errors="replace"),
                        default=io.TextIOWrapper(sys.stdin.buffer, errors="replace"),
                        help="Tab-separated bilingual tagged file")
    parser.add_argument('output', nargs='?', type=argparse.FileType('wt'), default=sys.stdout,
                        help="Output of the classification")
    parser.add_argument('--annotated_output', type=argparse.FileType('wt'),
                        help="Annotated output of the classification")

    groupM = parser.add_argument_group('Mandatory')
    groupM.add_argument("-s", "--source_lang", type=str, required=True, help="Source language (SL) of the input")
    groupM.add_argument("-t", "--target_lang", type=str, required=True, help="Target language (TL) of the input")

    groupO = parser.add_argument_group('Optional')
    groupO.add_argument('--tmp_dir', default=gettempdir(),
                        help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count() - 1),
                        help="Number of processes to use")

    groupO.add_argument('--new_hardrules', help='use the new hardrules on top of the existing hardrules in v1.1',
                        action='store_true', dest='new_hardrules')

    args = parser.parse_args()

    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    return args


def c_identical(left, right):
    return left != right


def c_identical_wo_digits(left, right):
    left = regex_digit.sub("", left)
    right = regex_digit.sub("", right)
    return left != right


def c_minimal_length(sentence):
    """ Counts number of whitespace, requires > 2 """
    return len(regex_blank.findall(sentence)) > 2


def c_maximal_number_of_tokens(sentence):
    """ Counts number of tokens, requires <= 80 """
    return len(nltk.word_tokenize(sentence)) <= 80


def c_minimal_edit_distance(left, right):
    return Levenshtein.distance(left, right) >= 2


def c_minimal_edit_distance_ratio(left, right):
    avg_sentence_length = (len(left) + len(right)) / 2
    return (Levenshtein.distance(left, right) / avg_sentence_length) >= 0.1


def c_length(left, right, args):
    length_range = [0.4, 2.5] if args.new_hardrules else [0.5, 2.0]
    return length_range[0] <= float(len(left)) / float(len(right)) <= length_range[1]


def c_different_language(left, right):
    l_reliable = False
    l_bytes = 0
    l_details = ()

    try:
        l_reliable, l_bytes, l_details = pycld2.detect(left)
    except:
        return False  # encoding error -> noise

    r_reliable = False
    r_bytes = 0
    r_details = ()

    try:
        r_reliable, r_bytes, r_details = pycld2.detect(right)
    except:
        return False  # encoding error -> noise

    if l_reliable and r_reliable and l_details[0][1] != r_details[0][1]:
        return True
    elif not l_reliable or not r_reliable:
        return True
    else:
        return False


def c_reliable_long_language(sentence, language):
    reliable = False
    bytes = 0
    details = ()

    try:
        reliable, bytes, details = pycld2.detect(sentence)
    except:
        return True  # encoding error -> noise

    if len(sentence) > 30 and reliable and details[0][1] != language:
        return False
    else:
        return True


def c_alpha(sentence):
    return len(regex_alpha.findall(sentence)) > 0


def c_majority_alpha(sentence, args):
    cut_off_value = 0.2 if args.new_hardrules else 0.5
    return float(len(regex_alpha.findall(sentence))) / float(len(sentence)) >= cut_off_value


def c_no_urls(sentence):
    return sum([len("".join(i)) for i in regex_url.findall(sentence)]) < 15


def c_consistency_of_special_tokens(left, right):
    same_urls = sorted(regex_url.findall(left)) == sorted(regex_url.findall(right))
    same_email_addresses = sorted(regex_email_address.findall(left)) == sorted(regex_email_address.findall(right))
    return same_urls and same_email_addresses


def c_no_breadcrumbs(sentence):
    return len(regex_breadcrumbs.findall(sentence)) < 3


def c_no_noise(sentence):
    return len(regex_unicode_noise.findall(sentence)) == 0


def c_no_space_noise(sentence):
    return len(regex_spaces_noise.findall(sentence)) == 0


def c_no_paren(sentence):
    return len(regex_paren.findall(sentence)) < 10


def c_unwanted(sentence):
    return len(regex_unwanted.findall(sentence)) < 5


def c_inconditional(sentence):
    return len(regex_inconditional.findall(sentence)) < 1


def c_no_literals(literals, sentence):
    return not any(l in sentence for l in literals)


def c_no_escaped_unicode(sentence):
    return len(regex_escaped_unicode.findall(sentence)) == 0


def wrong_tu(left, right, args):
    try:
        args.new_hardrules = args.new_hardrules
    except AttributeError:
        args.new_hardrules = False

    if len(left) >= 1024:
        return "len(left) >= 1024"
    if len(right) >= 1024:
        return "len(right) >= 1024"
    elif not c_no_literals(["Porn"], left):
        return "c_no_literals(['Porn'], left)"
    elif not c_no_literals(["Porn"], right):
        return "c_no_literals(['Porn'], right)"
    elif not c_minimal_length(left):
        return "c_minimal_length(left)"
    elif not c_minimal_length(right):
        return "c_minimal_length(right)"
    elif args.new_hardrules and not c_maximal_number_of_tokens(left):
        return "c_maximal_number_of_tokens(left)"
    elif args.new_hardrules and not c_maximal_number_of_tokens(right):
        return "c_maximal_number_of_tokens(right)"
    elif not c_length(left, right, args):
        return "c_length"
    elif not c_identical(left, right):
        return "c_identical"
    elif not c_identical_wo_digits(left, right):
        return "c_identical_wo_digits"
    elif not c_different_language(left, right):
        return "c_different_language"
    elif not c_majority_alpha(left, args):
        return "c_majority_alpha(left)"
    elif not c_majority_alpha(right, args):
        return "c_majority_alpha(right)"
    elif args.new_hardrules and not c_minimal_edit_distance(left, right):
        return "c_minimal_edit_distance"
    elif args.new_hardrules and not c_minimal_edit_distance_ratio(left, right):
        return "c_minimal_edit_distance_ratio"
    elif not c_no_urls(left):
        return "c_no_urls(left)"
    elif not c_no_urls(right):
        return "c_no_urls(right)"
    elif args.new_hardrules and not c_consistency_of_special_tokens(left, right):
        return "c_consistency_of_special_tokens"
    elif not c_no_breadcrumbs(left):
        return "c_no_bradcrumbs(left)"
    elif not c_no_breadcrumbs(right):
        return "c_no_breadcrumbs(right)"
    elif args.source_lang in safe_noise_detection_langs and not c_no_noise(left):
        return "args.source_lang in safe_noise_detection_langs and not c_no_noise(left)"
    elif args.target_lang in safe_noise_detection_langs and not c_no_noise(right):
        return "args.target_lang in safe_noise_detection_langs and not c_no_noise(right)"
    elif not c_no_space_noise(left):
        return "c_no_space_noise(left)"
    elif not c_no_space_noise(right):
        return "c_no_space_noise(right)"
    elif not c_no_paren(left):
        return "c_no_paren(left)"
    elif not c_no_paren(right):
        return "c_no_paren(right)"
    elif not c_unwanted(left):
        return "c_unwanted(left)"
    elif not c_unwanted(right):
        return "c_unwanted(right)"
    elif not c_inconditional(left):
        return "c_inconditional(left)"
    elif not c_inconditional(right):
        return "c_inconditional(right)"
    elif not c_no_escaped_unicode(left):
        return "c_no_escaped_unicode(left)"
    elif not c_no_escaped_unicode(right):
        return "c_no_escaped_unicode(right)"
    elif not c_no_literals(["{{", "%s", "}}"], left):
        return 'c_no_literals(["{{", "%s", "}}"], left)'
    elif not c_no_literals(["{{", "%s", "}}"], right):
        return 'c_no_literals(["{{", "%s", "}}"], right)'
    elif left.istitle() and right.istitle():
        return 'left.istitle() and right.istitle()'
    elif not c_reliable_long_language(left, args.source_lang):
        return "c_reliable_long_language(left, sourcelang)"
    elif not c_reliable_long_language(right, args.target_lang):
        return "c_reliable_long_language(right, targetlang)"

    return False


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
                args.output.write(i)
            filein.close()

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Hard rules applied. Output available in {}".format(args.output.name))
    args.output.close()


def worker_process(i, jobs_queue, output_queue, args):
    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {0}".format(job.__repr__()))
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False,
                                                                      dir=args.tmp_dir) as fileout:
                logging.debug("Classification: creating temporary filename {0}".format(fileout.name))

                for i in filein:
                    parts = i.strip().split("\t")
                    left = parts[0]
                    right = parts[1] if len(parts) >= 2 else ""
                    wrong_tu_results = wrong_tu(left, right, args)
                    if wrong_tu_results != False:
                        fileout.write("{}\t{}\t0.0000000000000000\tdiscard\n".format(left, right))
                        if args.annotated_output:
                            args.annotated_output.write("{}\t{}\t{}\n".format(left, right, wrong_tu_results))
                    else:
                        fileout.write(i)

                ojob = (nblock, fileout.name)
                filein.close()
                fileout.close()

            if ojob:
                output_queue.put(ojob)
            if args.annotated_output:
                args.annotated_output.flush()
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


def perform_hardrules_filtering(args):
    time_start = default_timer()
    logging.info("Starting process")
    logging.info("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize=maxsize)
    worker_count = process_count

    # Start reducer
    reduce = Process(target=reduce_process,
                     args=(output_queue, args))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize=maxsize)
    workers = []
    for i in range(worker_count):
        filter = Process(target=worker_process,
                         args=(i, jobs_queue, output_queue, args))
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

    if args.annotated_output:
        args.annotated_output.close()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline * 1.0) / elapsed_time)))


def main(args):
    logging.info("Executing main program...")
    perform_hardrules_filtering(args)
    logging.info("Program finished")


if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization()
        main(args)
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
