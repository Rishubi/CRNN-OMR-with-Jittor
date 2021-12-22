#!/usr/bin/python
# encoding: utf-8

import collections
import numpy as np
import os

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        domain (str): type of representation, semantic or agnostic.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, domain, root='data'):
        self.domain = domain
        with open(os.path.join(root, 'vocab_{}.txt'.format(domain))) as f:
            alphabet = f.readlines()
            alphabet = [note.strip() for note in alphabet]
        alphabet.append('-')  # for `-1` index
        self.alphabet = alphabet

        self.dict = {}
        for i, note in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[note] = i + 1

    def encode(self, text):
        """encode batch str.

        Args:
            text (list of str): texts to convert.

        Returns:
            np.array [n, l]: encoded texts. (l = Maximum of all text lengths)
            np.array [n]:    length of each text.

        Example:
            >>> encode(["clef-G2 keySignature-EbM timeSignature-3/4", "note-Ab5_sixteenth note-G5_sixteenth"])
            np.array([[11, 230, 1758], [519, 215, 0]]), np.array([3, 2])
        """

        noteStrings = [t.split() for t in text]
        length = [len(t) for t in noteStrings]
        max_length = max(length)
        text = [ [self.dict[char] for char in t] + [0] * (max_length - len(t)) for t in noteStrings ]
        return np.array(text), np.array(length)

    def decode(self, encoded_t, raw=False):
        """Decode encoded texts back into lists of strs.

        Args:
            encoded_t (np.array [n, length]): encoded text to decode
            raw       (bool): do not ignore '-' and repeated characters

        Returns:
            list of decoded texts
        
        Example:
            >>> decode(np.array([[11, 230, 1758], [519, 519, 0]]))
            raw=True:  ["clef-G2 keySignature-EbM timeSignature-3/4", "note-Ab5_sixteenth note-Ab5_sixteenth -"]]
            raw=False: ["clef-G2 keySignature-EbM timeSignature-3/4", "note-Ab5_sixteenth"]
        """   

        texts = []
        length = encoded_t.shape[1]
        for t in encoded_t:
            if raw:
                texts.append([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                texts.append(' '.join(char_list))
        return texts



class averager(object):
    """Compute average for numpy.array."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.size
        v = v.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def array2str(arr, separator=", "):
    """
    transform Iterable to string. 
    e.g. [[1, 2], [3]]  ->  "[[1, 2], [3]]"
    """
    if isinstance(arr, collections.Iterable):
        return "[" + separator.join([array2str(x) for x in arr]) + "]"
    else:
        return str(arr)

def jtArray2str(arr):
    """
    transform jt.array to string. 
    """
    return array2str(np.array(arr).tolist())
