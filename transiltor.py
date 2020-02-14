import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pytils import translit
import re
import io

class ScopusExtractor:
    __scopus_authors = None
    __author_ids = None

    def __init__(self, article_file_path, d1=0.5, f1=0.5):
        """
        :param article_file_path: path to json-file with article information
        :param d1: Hamming distance threshold.
        :param f1: threshold for similarity measures.
        """
        self.__scopus_authors, self.__author_ids = self.__read_articles_from_json(article_file_path)
        self.d1 = d1
        self.f1 = f1

    def __extract_scopus_id(self, link):
        """
        Extract SCOPUS ID from URL.
        :param link: URL of author's page.
        :return: SCOPUS ID
        :rtype: str
        """
        start = 50
        end = start + link[start:].find('&')

        return link[start:end]

    def __read_articles_from_json(self, filename):
        """
        :param filename: path to json-file with article information.
        :returns:
        __scopus_authors: dictionary of <scopus_id;set(names)> pairs.
        __author_ids: dictionary of <name;set(scopus_ids)> pairs.
        """
        scopus_authors: defaultdict[str, set] = defaultdict(set)
        author_ids: defaultdict[str, set] = defaultdict(set)
        file = io.open('texts/users_articles_data_2.json',encoding='utf-8')
        for line in file:
            try:
                article = json.loads(file.readline())
                author_list = article['authorlist']
                authorlist_links = article['authorlistLinks']

                for idx, link in enumerate(authorlist_links):
                    scopus_id = self.__extract_scopus_id(link)
                    author = author_list[idx].lower()
                    scopus_authors[scopus_id].add(author)
                    author_ids[author].add(scopus_id)
            except:
                continue

        return scopus_authors, author_ids

    def __get_cooccurrence_with_order(self, w1, w2):
        """Calculate fraction of co-occurrences in two words,
        taking into account an order of symbols.

        :param w1: some word.
        :param w2: some word.
        :return: fraction of co-occurrences in two words
        """
        if len(w1) < len(w2):
            min_w = w1
            max_w = w2
        else:
            min_w = w2
            max_w = w1

        curr_idx = 0

        for i in range(len(max_w)):
            if min_w[curr_idx] == max_w[i]:
                curr_idx += 1

                if curr_idx == len(min_w):
                    break

        return curr_idx / len(max_w)

    def __get_symbol_intersection(self, w1, w2):
        """Calculate fraction of symbol intersection in two words.
        :param w1: some word.
        :param w2: some word.
        :return: fraction of symbol intersection in two words.
        """
        set1 = set(w1)
        set2 = set(w2)
        max_length = max(len(set1), len(set2))
        intr = set.intersection(set1, set2)

        return len(intr) / max_length

    def __hamming_distance(self, w1, w2):
        """
        Calculate Hamming distance between two words.
        :param w1: some word.
        :param w2: some word.
        :return: Hamming distance.
        """
        if len(w1) < len(w2):
            w = w1
        else:
            w = w2

        distance = 0

        for i in range(len(w)):
            if w1[i] != w2[i]:
                distance += 1

        distance += np.abs(len(w2) - len(w1))

        return distance / len(w)

    def __get_letters(self, name):
        return "".join(re.findall("[а-яёА-ЯЁ]+|[a-zA-Z]+", name))

    def __get_name_with_initials(self, fullname, initial_count=2):
        """
        Convert fullname to lastname+initials.
        :param fullname: fullname.
        :param initial_count: number of name parts, which is should be included as initials.
        :return: lastname+initials.
        """
        tokens = fullname.split(' ')
        name = tokens[0] + ', '  # lastname

        for i in range(initial_count):
            if i + 1 < len(tokens):
                name += tokens[1 + i][0] + '.'

        return name

    def __get_lastname(self, fullname, article=False):
        """
        Extract lastname from fullname.
        :param fullname: fullname.
        :param article: whether fullname is from article info.
        :return: lastname.
        """
        tokens = fullname.split(' ')
        if article:
            return tokens[0][:-1]
        return tokens[0]

    def __get_initials_coincidence_count(self, article_name, fullname_translit):
        """
        Calculate number of coincidence in initials.
        :param article_name: name from article info.
        :param fullname_translit: origin name in translit.
        :return: number of coincidence in initials.
        """
        parts = fullname_translit.split(' ')
        tokens = article_name.split(',')
        initials = tokens[-1].strip()

        if initials.count('.') > 1:
            initials = initials[:-1].split('.')
        else:
            initials = [initials[:-1]]

        coincidence_count = 0
        try:
            for idx, letter in enumerate(initials):
                if letter != parts[idx + 1][0]:
                    return False
                coincidence_count += 1
        except:
            return coincidence_count

        return coincidence_count

    def __get_last_coincidence(self, fullname):
        """
        Get coincidence of fullname (which is transformed into lastname+initials)
        with author name from article info file.
        :param fullname: fullname in translit.
        :return: author name from article info file.
        """
        tokens = fullname.split(' ')
        name = tokens[0] + ', '  # lastname

        coincidence = None
        for i in range(len(tokens) - 1):
            name += tokens[i + 1][0] + '.'
            if name in self.__author_ids:
                coincidence = name

        return coincidence

    def get_len_diff(self, w1, w2):
        min_len = min(len(w1), len(w2))
        max_len = max(len(w1), len(w2))

        return max_len / min_len

    def try_get_scopus_ids(self, fullname):
        """
        Get list of possible SCOPUS ids.
        :param fullname: origin full name in Russian.
        :return: matched article authors and their SCOPUS ids.
        """
        fullname = fullname.lower()
        fullname_translit = translit.translify(fullname)

        authors = set()
        scopus_ids = set()

        coincidence = self.__get_last_coincidence(fullname_translit)

        if coincidence is not None:
            authors.add(coincidence)
            ids = self.__author_ids[coincidence]
            scopus_ids.update(ids)
            return authors, scopus_ids

        for author, ids in self.__author_ids.items():
            coincidence_count = self.__get_initials_coincidence_count(author, fullname_translit)
            if coincidence_count == 0:
                continue

            initial_count = author.count('.')
            translit_article_format = self.__get_name_with_initials(fullname_translit, initial_count)

            name1 = self.__get_letters(translit_article_format)
            name2 = self.__get_letters(author)

            d1 = self.__hamming_distance(name1, name2)
            sim1 = self.__get_cooccurrence_with_order(name1, name2)
            sim2 = self.__get_symbol_intersection(name1, name2)
            f1 = 2 * (sim1 * sim2) / (sim1 + sim2) if (sim1 + sim2 > 0) else 0
            len_diff = self.get_len_diff(name1, name2)

            # if (d1 < 0.5) or (sim1 >= 0.6 and sim2 > 0.8):
            if ((d1 < self.d1) or (f1 >= self.f1)) and (len_diff <= 1.5):
                authors.add(author)
                scopus_ids.update(ids)

        return authors, scopus_ids



