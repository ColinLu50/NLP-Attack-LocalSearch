import os
import sys
sys.path.append(sys.path[0] + '/../')


import logging
import re

def _clean(s):
    """
    Removes trailing and leading spaces and collapses multiple consecutive internal spaces to a single one.
    :param s: The string.
    :return: A cleaned-up string.
    """
    return re.sub(r'\s+', ' ', s.strip())

def process_to_text(rawfile, txtfile):
    """Processes raw files to plain text files.
    :param rawfile: the input file (possibly SGML)
    :param txtfile: the plaintext file
    """

    if not os.path.exists(txtfile) or os.path.getsize(txtfile) == 0:
        logging.info("Processing %s to %s", rawfile, txtfile)
        if rawfile.endswith('.sgm') or rawfile.endswith('.sgml'):
            with open(rawfile) as fin, open(txtfile, 'wt') as fout:
                for line in fin:
                    if line.startswith('<seg '):
                        print(_clean(re.sub(r'<seg.*?>(.*)</seg>.*?', '\\1', line)), file=fout)

datset_dir = 'data/morpheus/dataset/newstest/'

# ================= French ==================
rawfile_path = os.path.join(datset_dir, 'final/newstest2014-fren-src.en.sgm')
outfile_path = os.path.join(datset_dir, 'raw_text/newstest2014-en.txt')

process_to_text(rawfile_path, outfile_path)

# ================ English ===================
rawfile_path = os.path.join(datset_dir, 'final/newstest2014-fren-src.fr.sgm')
outfile_path = os.path.join(datset_dir, 'raw_text/newstest2014-fr.txt')

process_to_text(rawfile_path, outfile_path)