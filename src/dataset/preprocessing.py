""" Functions for processing DRID dataset and token manipulation. """

import joblib, glob
import time, datetime
import re, ast, copy, os
from itertools import product
from typing import Optional, Tuple, List, Dict
from functools import lru_cache

import numpy as np
import pandas as pd
import pubchempy

from pandas._libs.missing import NAType
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from xml.etree.ElementTree import Element
from tqdm.auto import tqdm
from IPython import display


try:
    import Levenshtein
    from rapidfuzz import fuzz
    from lxml import html
    HAS_LEVENSHTEIN = True
    HAS_RAPIDFUZZ = True
    HAS_LXML = True
except ImportError:
    HAS_LEVENSHTEIN = False
    HAS_RAPIDFUZZ = False
    HAS_LXML = False


class Drug:
    """
    A class to extract and store relevant data from a DrugBank XML child object.

    Parameters
    ----------
    xml : xml.etree.ElementTree.Element
        An XML element representing a drug from the DrugBank database.

    Attributes
    ----------
    name : str
        The name of the drug.
    cas_number : str
        The CAS number of the drug.
    mass : str
        The average mass of the drug.
    drugbank_id : str
        The primary DrugBank ID of the drug.
    products : list of str
        A sorted list of product names containing given drug.
    mixtures : list of tuple
        A list of tuples, each containing the name of the mixture and its ingredients.
    synonyms : list of str
        A sorted list of synonyms for the drug.
    international : list of str
        A sorted list of international brand names for the drug that are not mixtures.
    kingdom_ : str
        The kingdom classification of the drug.
    superclass_ : str
        The superclass classification of the drug.
    class_ : str
        The class classification of the drug.
    subclass_ : str
        The subclass classification of the drug.
    experimental : dict
        A dictionary of experimental properties of the drug.
    calculated : dict
        A dictionary of calculated properties of the drug.

    Methods
    -------
    __str__():
        Returns a string representation of the Drug object.
    __repr__():
        Returns a string representation of the Drug object.
    change_tags(xml_level):
        Changes the tags of the XML element and its sub-elements to remove namespace.
    has_text(object_):
        Returns the text content of an XML element, cleaned and lowercase.
    has_item(object_, item):
        Checks if an XML element has a specified sub-element.
    is_iterable(object_):
        Checks if an object is iterable.
    """

    def __init__(self, xml: Optional[Element], keep_xml: bool = False):

        if not HAS_LXML:
            raise ImportError('lxml library not found, but can be installed using < pip instal lxml >')

        if xml is not None:
            self.xml = xml
            self.change_tags(self.xml)

            self.name = self.has_text(self.xml.find('./name'))
            self.cas_number = self.has_text(self.xml.find('./cas-number'))
            self.mass = self.has_text(self.xml.find('./average-mass'))
            self.drugbank_id = self.has_text(self.xml.find('./drugbank-id[@primary="true"]'))

            self.synonyms = sorted(set([self.has_text(item) for item in self.xml.find('./synonyms')]))
            self.products = sorted(set([self.has_text(item.find('./name')) for item in self.xml.find('./products')]))
            self.international = sorted(set([self.has_text(item.find('./name')) for item in self.xml.find('./international-brands')]))
            self.mixtures = {self.has_text(item.find('./name')): self.has_text(item.find('./ingredients')).split(' + ') for item in self.xml.find('./mixtures')}

            for name in self.mixtures.keys():  # Remove mixture names from products and international, as they will be later used for substitutions
                try:
                    self.products.remove(name)
                    self.international.remove(name)
                except ValueError:  # pass if element not found
                    pass

            self.kingdom_ = self.has_text(self.xml.find('./classification/kingdom'))
            self.superclass_ = self.has_text(self.xml.find('./classification/superclass'))
            self.class_ = self.has_text(self.xml.find('./classification/class'))
            self.subclass_ = self.has_text(self.xml.find('./classification/subclass'))

            if self.has_item(self.xml, './experimental-properties'):
                self.experimental = {self.has_text(item.find('./kind')): self.has_text(item.find('./value'), False)
                                     for item in self.xml.find('./experimental-properties')}

            if self.has_item(self.xml, './calculated-properties'):
                self.calculated = {self.has_text(item.find('./kind')): self.has_text(item.find('./value'), False)
                                   for item in self.xml.find('./calculated-properties')}
            else:
                self.calculated = {}

            if keep_xml is False:
                self.xml = Element('')

        else:
            print('Provide general information about drug\n')
            self.name = input('Generic name: ')
            self.cas_number = input('CAS number: ')
            self.mass = input('Average mass: ')
            self.drugbank_id = input('Drugbank ID: ')
            print('Provide known synonyms and brand. Individual entries should be split by " | "')
            self.synonyms = input('Synonyms: ').split(' | ')
            self.products = input('Products: ').split(' | ')
            self.international = input('International brands: ').split(' | ')
            print('Provide mixtures in form of key : values: ')
            self.mixtures = {}
            key = input('Provide key: ')
            while key != '':
                self.mixtures[key] = input('Provide values: ').split(' | ')
                key = input('Provide key: ')
            print('Provide classification: ')
            self.kingdom_ = input('Kingdom: ')
            self.superclass_ = input('Superclass: ')
            self.class_ = input('Class: ')
            self.subclass_ = input('Subclass: ')
            print('Provide experimental properties in form of key : value')
            self.experimental = {}
            key = input('Provide key: ')
            while key != '':
                self.experimental[key] = input('Provide value: ')
                key = input('Provide key: ')
            print('Provide calculated properties in form of key : value')
            self.calculated = {}
            key = input('Provide key: ')
            while key != '':
                self.calculated[key] = input('Provide value: ')
                key = input('Provide key: ')
            display.clear_output(wait=False)
            print('Finished preparing a Drug entry')
            self.xml = Element('')

    def __str__(self):
        return f'Drug-class object of < {self.name} >'

    def __repr__(self):
        return f'Drug-class object of < {self.name} >'

    def prepare_changes(self):
        """
        Prepare a list of required changes to the XML element
        """

        changes = [('capture_by_regex', [self.plain_string(self.name)])]  # First make sure any already clean instance is removed

        for synonym in self.synonyms:
            changes.append(('substitute_by_regex', [self.plain_string(synonym), self.name]))  # Change all synonyms to main name

        changes.append(('capture_by_regex', [self.plain_string(self.name)]))  # Capture all synonyms that were found

        for name, ingredients in self.mixtures.items():
            if self.name not in name:  # hopefully this will allow to NOT destroy the data
                changes.append(('extend_by_regex', [self.plain_string(name), ' | '.join(ingredients)]))

        changes.append(('capture_by_regex', [self.plain_string(self.name)]))

    def change_tags(self, element):
        """
        Recursively changes the tags of an XML element and its sub-elements to remove namespace.

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            The XML element to process.
        """
        if '}' in element.tag:
            element.tag = element.tag.split('}')[1]

        if self.is_iterable(element):
            for sub_level in element:
                self.change_tags(sub_level)

    @staticmethod
    def has_text(element, lower: bool = True):
        """
        Returns the text content of an XML element, cleaned and lowercase.

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            The XML element to extract text from.
        lower: bool
            Whether to return lowercase string or not

        Returns
        -------
        str
            The cleaned text content, or an empty string if not found.
        """
        try:
            text = element.text
            if isinstance(text, str):
                out = re.sub(r'\s+', ' ', text.strip())
                return out.lower() if lower else out
            else:
                return ''
        except AttributeError:
            return ''

    @staticmethod
    def has_item(element, item):
        """
        Checks if an XML element has a specified sub-element.

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            The XML element to check.
        item : str
            The tag name of the sub-element to find.

        Returns
        -------
        bool
            True if the sub-element is found, False otherwise.
        """
        return element.find(item) is not None

    @staticmethod
    def is_iterable(element):
        """
        Checks if an element is iterable.

        Parameters
        ----------
        element : xml.etree.ElementTree.Element
            The object to check.

        Returns
        -------
        bool
            True if the object is iterable, False otherwise.
        """
        try:
            _ = iter(element)
            return True
        except TypeError:
            return False

    @staticmethod
    def plain_string(string):
        """
        Convert a string to its regex-safe equivalent.

        Parameters
        ----------
        string: str
            Any sting

        Returns
        -------
        string: str
            A string with \\ regex characters

        """
        string = string.strip().lower()
        string = re.sub(r"\s+", " ", string)

        out = re.findall(r"(\[[^\[\]]+])", string)
        for match in out:
            if "-" in match:
                new_match = match.replace("-", "\\-")
                string = string.replace(match, new_match)

        regex_chars = ['(', ')', '[', ']', '{', '}', '.', '^', '$', '+', '?', '|', '*', '\\']

        for regex_char in regex_chars:
            string = string.replace(regex_char, f"\\{regex_char}")

        string = string.replace(" ", r"\s")
        string = string.replace('\\\\', '\\')

        return string


class DridProcessor2:
    """
    Initialize the Processor instance.

    Parameters
    ----------
    df : pd.DataFrame, optional
        A pandas DataFrame containing the initial data. It must contain the specified name, token,
        and target columns. Default is an empty DataFrame with columns ['name', 'tokens', 'active'].

    name_column : str, optional
        The name of the column containing drug names. Default is 'name'.

    token_column : str, optional
        The name of the column containing token information. Default is 'tokens'.

    target_column : str, optional
        The name of the column for the target data. Default is 'active'.
        If this column does not exist in the DataFrame, it will be created with empty lists.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame initialized with the provided or default DataFrame.

    name_column : str
        The column name for drug names.

    token_column : str
        The column name for token information.

    target_column : str
        The column name for target data.

    processed_df : pd.DataFrame
        An empty DataFrame initialized to store processed data.

    drugbank : list[Drug]
        A list of Drug objects representing the drug database.

    pme : dict[str:list]
        The Pharmaceutical Manufacturing Encyclopedia.

    chembl : dict[str:list]
        The ChEMBL 34 trade_name : ingredient mapping.

    drugnames : None
        A placeholder for storing drug names (initialized as None).

    changes_mapping : None
        A placeholder for storing names of changes (initialized as None).

    execution_mode : str
        Mode of execution, either 'immediate' or 'delayed'. Default is 'immediate'.

    automatic_mode : bool
        Indicates whether automatic command saving is enabled. Default is False.

    all_changes : list
        A list to keep track of all changes made during processing.

    changes : list
        A list to keep track of current changes made.

    automatic_changes : dict
        A dictionary to store automatic changes made.

    smiles_mapping : list
        A list to store SMILES mappings for processed tokens.

    tokens : list
        A list to store token information.

    num_updates : int
        A counter for the number of updates made to the DataFrame.

    function_mapping : dict
        A dictionary mapping function names to their corresponding methods for processing changes.

    verbosity : int
        Parameter for controlling the level of errors output.

    Notes
    -----
    Current version: 1.0.0
    """

    mapping_path = '../data/files/'

    drugbank = None  # joblib.load(mapping_path + 'drugbank.joblib')
    pme = None  # joblib.load(mapping_path + 'pme.joblib')
    chembl = None  # joblib.load(mapping_path + 'chembl.joblib')
    smiles_mapping = None  # joblib.load(mapping_path + 'smiles_mapping.joblib')
    drugnames = None  # joblib.load(mapping_path + 'drugnames.joblib')

    verbosity = 0

    def __init__(self, df: pd.DataFrame = pd.DataFrame(columns=['name', 'tokens', 'active']),
                 name_column: str = 'name', token_column: str = 'tokens', target_column: str = 'active',
                 path: str = None):

        if not (HAS_RAPIDFUZZ or HAS_LEVENSHTEIN):
            raise ImportError("rapidfuzz and/or Levenshtein libraries not found, but can be installed using"
                              "< pip install rapidfuzz Levenshtein >")

        self.data_path = path if path is not None else self.__class__.mapping_path
        try:
            self.__class__.drugbank = joblib.load(self.data_path + 'drugbank.joblib')
            self.__class__.pme = joblib.load(self.data_path + 'pme.joblib')
            self.__class__.chembl = joblib.load(self.data_path + 'chembl.joblib')
            self.__class__.smiles_mapping = joblib.load(self.data_path + 'smiles_mapping.joblib')
            self.__class__.drugnames = joblib.load(self.data_path + 'drugnames.joblib')

        except FileNotFoundError as err:
            print(f"Error loading files from {self.data_path}: {err}")

        self.df = df.reset_index(drop=True)
        self.name_column = name_column
        self.token_column = token_column
        self.target_column = target_column

        if self.target_column not in self.df.columns:
            self.df[target_column] = [[] for _ in range(len(self.df))]

        self.processed_df = self.df.head(0)

        self.changes_mapping = None

        self.execution_mode = 'immediate'  # one of ['immediate', 'delayed'] | new in 0.5
        self.automatic_mode = False  # new in 0.5

        self.all_changes = []
        self.changes = []
        self.automatic_changes = {}  # new in 0.5

        self.tokens = []
        self.num_updates = 0

        self.function_mapping = {'substitute_by_string': self.substitute_by_string,
                                 'substitute_by_regex': self.substitute_by_regex,
                                 'remove_by_string': self.remove_by_string,
                                 'remove_by_regex': self.remove_by_regex,
                                 'remove_remaining_by_string': self.remove_remaining_by_string,
                                 'extend_by_string': self.extend_by_string,
                                 'extend_by_regex': self.extend_by_regex,
                                 'move_by_string': self.move_by_string,
                                 'move_by_regex': self.move_by_regex,
                                 'capture_by_string': self.capture_by_string,
                                 'capture_by_regex': self.capture_by_regex,
                                 'update_tokens': self.update_tokens}

        self.collect_similar()

    def __str__(self):
        print(f'Version {self.num_updates} of DridProcessor')

    def __repr__(self):
        return f'Version {self.num_updates} of DridProcessor'

    @classmethod
    def log_error(cls, message: str = '', method: str = '', error: str = ''):
        if cls.verbosity > 0:
            print(f'{message} occurred within {method}: {error}')

    @staticmethod
    def plain_string(string: str):
        """
        Convert provided string to a regex-neutral version by escaping special characters.

        Parameters
        ----------
        string : str
            The input string to be converted into a regex-safe format.

        Returns
        -------
        str
            The regex-safe version of the input string with special characters escaped.
        """
        string = re.sub("\\s+", " ", string.strip().lower())

        for match in re.findall("(\\[[^\\[\\]]+])", string):
            if "-" in match:
                new_match = match.replace("-", "\\-")
                string = string.replace(match, new_match)

        regex_chars = ['(', ')', '[', ']', '{', '}', '.', '^', '$', '+', '?', '|', '*', '\\']

        for regex_char in regex_chars:
            string = string.replace(regex_char, f"\\{regex_char}")

        string = string.replace(" ", "\\s")
        string = string.replace('\\\\', '\\')

        return string

    def load(self, path: str = '../data/files/processor.joblib'):
        """
        Overwrite current attributes of the processor by loading from a saved version.

        Parameters
        ----------
        path : str, optional
            The file path to the saved DridProcessor2.
        """
        processor = joblib.load(path)

        self.df = processor.df
        self.name_column = processor.name_column
        self.token_column = processor.token_column
        self.target_column = processor.target_column
        self.processed_df = processor.processed_df
        self.all_changes = processor.all_changes  # stores changes made to df
        self.changes = processor.changes
        self.automatic_changes = processor.automatic_changes
        self.smiles_mapping = processor.smiles_mapping  # stores smiles mapping from chemical resolver
        self.tokens = processor.tokens
        self.num_updates = processor.num_updates
        self.collect_similar()
        self.collect_drugnames()

    def save(self, path: str = '../data/files/processor.joblib'):
        """
        Save the current version of the processor to a .pkl file.

        Parameters
        ----------
        path : str, optional
            The file path where the current DridProcessor object will be saved
        """
        joblib.dump(self, path)

        joblib.dump(self.__class__.drugbank, self.data_path + 'drugbank.joblib')
        joblib.dump(self.__class__.pme, self.data_path + 'pme.joblib')
        joblib.dump(self.__class__.chembl, self.data_path + 'chembl.joblib')
        joblib.dump(self.__class__.smiles_mapping, self.data_path + 'smiles_mapping.joblib')

    def collect_tokens(self):
        """
        Collect all name:token pairs from the current dataset.
        """
        self.tokens = []

        for i, row in self.df.iterrows():
            name = row[self.name_column]
            tokens = row[self.token_column]

            if len(tokens) > 0:
                self.tokens.extend([pair for pair in product([name], tokens)])

        self.tokens = sorted(set(self.tokens))  # why not

    def clean_tokens(self):
        """
        Clean all tokens in the dataset by converting them to lowercase, stripping trailing whitespaces,
        and replacing multiple whitespaces with a single space.
        """

        pattern = re.compile('\\s+')

        def clean(tokens):
            return [pattern.sub(' ', token.strip().lower()) for token in sorted(set(tokens))]

        self.df.loc[:, self.token_column] = self.df[self.token_column].apply(clean)
        self.df.loc[:, self.target_column] = self.df[self.target_column].apply(clean)

    def substitute_by_string(self, string: str, replace_string: str):
        """
        Replace tokens in the dataset that exactly match the provided string with a specified replacement.

        Parameters
        ----------
        string : str
            The string to search for in tokens.
        replace_string : str
            The string that will replace the matching tokens.
        """

        def substitute(tokens):
            return [replace_string if string == token else token for token in tokens]

        if self.execution_mode == 'immediate':
            self.df.loc[:, self.token_column] = self.df[self.token_column].apply(substitute)
            self.all_changes.append(('substitute_by_string', [string, replace_string]))

            if self.automatic_mode and not self.automatic_changes.get(string):
                self.automatic_changes[string] = ('substitute_by_string', [string, replace_string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('substitute_by_string', [string, replace_string]))

    def substitute_by_regex(self, regex: str, replace_string: str):
        """
        Replace tokens in the dataset that match the provided regular expression with a specified replacement.

        Parameters
        ----------
        regex : str
            The regular expression to match against tokens.
        replace_string : str
            The string that will replace the matching tokens.
        """

        def replace_tokens(tokens, pattern_, replace_string_):
            return [pattern_.sub(f' {replace_string_} ', token) for token in tokens]

        pattern = re.compile(f'\\A{regex}\\Z|\\A{regex}\\s+|\\s+{regex}\\Z|\\s+{regex}\\s+')

        if self.execution_mode == 'immediate':
            self.df.loc[:, self.token_column] = self.df[self.token_column].apply(replace_tokens, pattern_=pattern,
                                                                                 replace_string_=replace_string)
            self.all_changes.append(('substitute_by_regex', [regex, replace_string]))
            if self.automatic_mode and not self.automatic_changes.get(regex):
                self.automatic_changes[regex] = ('substitute_by_regex', [regex, replace_string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('substitute_by_regex', [regex, replace_string]))

    def extend_by_string(self, string: str, extend_string: str):
        """
        Remove the specified token and extend the row with additional tokens split by ' | ' if the string
        exactly matches the token.

        Parameters
        ----------
        string : str
            The string to match against the tokens for removal.
        extend_string : str
            The string containing additional tokens to be added, separated by ' | '.
        """

        def extend(tokens, string_, extend_tokens_):

            found = False

            for token in tokens:
                if string_ == token:
                    tokens.remove(token)
                    found = True
            if found:
                tokens.extend(extend_tokens_)  # if at least one token was matching, add extend_tokens and change

            return tokens

        if self.execution_mode == 'immediate':
            self.df.loc[:, self.token_column] = self.df[self.token_column].apply(extend, string_=string,
                                                                                 extend_tokens_=extend_string.split(' | '))
            self.all_changes.append(('extend_by_string', [string, extend_string]))

            if self.automatic_mode and not self.automatic_changes.get(string):
                self.automatic_changes[string] = ('extend_by_string', [string, extend_string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('extend_by_string', [string, extend_string]))

    def extend_by_regex(self, regex: str, extend_string: str):
        """
        Substitute matching tokens found by the regex with a space and extend the row
        with additional tokens split by ' | '.

        Parameters
        ----------
        regex : str
            The regular expression to search for in the tokens.
        extend_string : str
            The string containing additional tokens to be added, separated by ' | '.
        """

        def extend(tokens, pattern_, extend_tokens_):

            found = False
            new_tokens = []

            for token in tokens:
                if pattern_.search(token) is not None:
                    new_token = pattern_.sub(' ', token)
                    found = True
                else:
                    new_token = token
                new_tokens.append(new_token)
            if found:
                new_tokens.extend(extend_tokens_)

            return new_tokens

        pattern = re.compile(f'\\A{regex}\\Z|\\A{regex}\\s+|\\s+{regex}\\s+|\\s+{regex}\\Z')

        if self.execution_mode == 'immediate':
            self.df.loc[:, self.token_column] = self.df[self.token_column].apply(extend, pattern_=pattern,
                                                                                 extend_tokens_=extend_string.split(' | '))
            self.all_changes.append(('extend_by_regex', [regex, extend_string]))

            if self.automatic_mode and not self.automatic_changes.get(regex):
                self.automatic_changes[regex] = ('extend_by_regex', [regex, extend_string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('extend_by_regex', [regex, extend_string]))

    def remove_by_string(self, string: str):
        """
        Remove tokens from the dataset that exactly match the provided string.

        Parameters
        ----------
        string : str
            The string to match against tokens for removal.
        """

        def remove(tokens, string_):
            return [token for token in tokens if not token == string_]

        if self.execution_mode == 'immediate':
            self.df.loc[:, self.token_column] = self.df[self.token_column].apply(remove, string_=string)
            self.all_changes.append(('remove_by_string', [string]))

            if self.automatic_mode and not self.automatic_changes.get(string):
                self.automatic_changes[string] = ('remove_by_string', [string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('remove_by_string', [string]))

    def remove_remaining_by_string(self, string: str):
        """
        Remove the token from the dataset if it exactly matches the provided string
        and is the only remaining token in the row.

        Parameters
        ----------
        string : str
            The string to match against the remaining token for removal.
        """

        def remove(tokens, string_):

            if (len_ := len(tokens)) > 1:  # if there's more than one token: skip
                return tokens
            elif len_ == 0:  # if there's no tokens: skip
                return []
            else:
                if tokens[0] == string_:
                    return []
                else:
                    return tokens

        if self.execution_mode == 'immediate':
            self.df.loc[:, self.token_column] = self.df[self.token_column].apply(remove, string_=string)
            self.all_changes.append(('remove_remaining_by_string', [string]))

            if self.automatic_mode and not self.automatic_changes.get(string):
                self.automatic_changes[string] = ('remove_remaining_by_string', [string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('remove_remaining_by_string', [string]))

    def remove_by_regex(self, regex: str):
        """
        Remove all tokens from the dataset that match the provided regular expression.

        Parameters
        ----------
        regex : str
            The regular expression to match against tokens for removal.
        """

        def remove(tokens, pattern_):
            return [token for token in tokens if pattern_.search(token) is None]

        pattern = re.compile(f'\\A{regex}\\Z|\\A{regex}\\s+|\\s+{regex}\\s+|\\s+{regex}\\Z')

        if self.execution_mode == 'immediate':
            self.df[self.token_column] = self.df[self.token_column].apply(remove, pattern_=pattern)
            self.all_changes.append(('remove_by_regex', [regex]))

            if self.automatic_mode and not self.automatic_changes.get(regex):
                self.automatic_changes[regex] = ('remove_by_regex', [regex])

        elif self.execution_mode == 'delayed':
            self.changes.append(('remove_by_regex', [regex]))

    def move_by_string(self, string: str):
        """
        Move tokens that exactly match the provided string from the token column to the target column.

        Parameters
        ----------
        string : str
            The string to match against tokens for moving.
        """

        def move(row, string_):

            tokens = row[self.token_column]
            new_tokens = []
            moved_tokens = []

            for token in tokens:
                if string_ == token:
                    moved_tokens.append(token)
                else:
                    new_tokens.append(token)

            row[self.token_column] = new_tokens
            row[self.target_column].extend(moved_tokens)

        if self.execution_mode == 'immediate':
            self.df.apply(move, string_=string, axis=1)
            self.all_changes.append(('move_by_string', [string]))

            if self.automatic_mode and not self.automatic_changes.get(string):
                self.automatic_changes[string] = ('move_by_string', [string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('move_by_string', [string]))

    def move_by_regex(self, regex: str):
        """
        Move all tokens matching the provided regular expression from the token column to the target column.

        Parameters
        ----------
        regex : str
            The regular expression to match against tokens for moving.
        """

        def move(row, pattern_):

            tokens = row[self.token_column]
            new_tokens = []
            moved_tokens = []

            for token in tokens:
                if pattern_.search(token) is not None:
                    moved_tokens.append(token)
                else:
                    new_tokens.append(token)

            row[self.token_column] = new_tokens
            row[self.target_column].extend(moved_tokens)

        pattern = re.compile(f'\\A{regex}\\Z|\\A{regex}\\s+|\\s+{regex}\\s+|\\s+{regex}\\Z')

        if self.execution_mode == 'immediate':
            self.df.apply(move, pattern_=pattern, axis=1)
            self.all_changes.append(('move_by_regex', [regex]))

            if self.automatic_mode and not self.automatic_changes.get(regex):
                self.automatic_changes[regex] = ('move_by_regex', [regex])

        elif self.execution_mode == 'delayed':
            self.changes.append(('move_by_regex', [regex]))

    def capture_by_string(self, string: str):
        """
        Extract the provided string from tokens and move it to the target column.

        Parameters
        ----------
        string : str
            The string to search for in the tokens, which will be moved to the target column if found.
        """

        def capture(row, string_):

            tokens = row[self.token_column]
            new_tokens = []
            captured_tokens = []

            for token in tokens:
                if string_ in token:
                    new_token = token.replace(string_, ' ')
                    captured_tokens.append(string_)
                else:
                    new_token = token
                new_tokens.append(new_token)

            row[self.token_column] = new_tokens
            row[self.target_column].extend(captured_tokens)

        if self.execution_mode == 'immediate':

            self.df.apply(capture, string_=string, axis=1)
            self.all_changes.append(('capture_by_string', [string]))

            if self.automatic_mode and not self.automatic_changes.get(string):
                self.automatic_changes[string] = ('capture_by_string', [string])

        elif self.execution_mode == 'delayed':
            self.changes.append(('capture_by_string', [string]))

    def capture_by_regex(self, regex: str):
        """
        Extract the portion of tokens matching the provided regular expression and move it to the target column.

        Parameters
        ----------
        regex : str
            The regular expression to search for in the tokens, which will be moved to the target column if found.
        """

        def capture(row, pattern_):

            tokens = row[self.token_column]
            new_tokens = []
            captured_tokens = []

            for token in tokens:
                if (result := pattern_.search(token)) is not None:  # if regex pattern is found in token
                    captured = result.group(1)  # extract found pattern
                    captured_tokens.append(captured)  # add token to new list
                    new_tokens.append(token.replace(captured, ' '))
                else:
                    new_tokens.append(token)

            row[self.token_column] = new_tokens
            row[self.target_column].extend(captured_tokens)

            return row

        pattern = re.compile(f'(\\A{regex}\\Z|\\A{regex}\\s+|\\s+{regex}\\s+|\\s+{regex}\\Z)')

        if self.execution_mode == 'immediate':
            self.df = self.df.apply(capture, pattern_=pattern, axis=1)
            self.all_changes.append(('capture_by_regex', [regex]))

            if self.automatic_mode and not self.automatic_changes.get(regex):
                self.automatic_changes[regex] = ('capture_by_regex', [regex])

        elif self.execution_mode == 'delayed':
            self.changes.append(('capture_by_regex', [regex]))

    def update_tokens(self, key: str, updated_tokens: str):
        """
        Replace previous tokens with new ones for a specified row, splitting the provided tokens using ' | '
        as the delimiter.

        Parameters
        ----------
        key : str
            The key corresponding to the row where tokens need to be updated.
        updated_tokens : str
            The new tokens, which will be split into individual tokens using ' | ' as a separator.
        """

        if self.execution_mode == 'immediate':

            inds = self.df[self.df[self.name_column] == key].index.values
            self.df.loc[inds, self.token_column] = updated_tokens
            self.df.loc[inds, self.token_column] = self.df.loc[inds, self.token_column].apply(lambda token: token.split(' | '))
            self.all_changes.append(('update_tokens', [key, updated_tokens]))

        elif self.execution_mode == 'delayed':
            self.changes.append(('update_tokens', [key, updated_tokens]))

    @staticmethod
    def list_similarity(string: str, target_strings: list, method: str = 'fuzzy', threshold: float = 0.75,
                        num_matches: int = 3):
        """
        Find the 3 most similar strings in passed target_strings using either fuzzy matching or Levenshtein distance.

        Parameters
        ----------
        string : str
            The input string to compare against the internal database.
        target_strings : list
            The list of strings to search for similarities
        method : str, optional
            The similarity method to use, either 'fuzzy' or 'levenshtein'. Default is 'fuzzy'.
        threshold : float, optional
            The minimum similarity threshold for considering a match. Default is 0.75
        num_matches: int, optional
            Number of top matches to return

        Returns
        -------
        list[tuple]
            A list of up to 3 tuples, each containing a matching string and its similarity score,
            sorted by similarity in descending order.
        """

        similarities = []

        for item in target_strings:
            if method == 'levenshtein':
                sim = np.round(1 - Levenshtein.distance(string, item) / max(len(string), len(item)), 3)
            elif method == 'fuzzy':
                sim = np.round(fuzz.ratio(string, item) / 100.0, 3)
            else:
                raise ValueError(f"Available options for method are: levenshtein, fuzzy")
            if sim >= threshold:
                similarities.append((item, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:num_matches]

    @staticmethod
    def dict_similarity(string: str, target_mapping: dict, method: str = 'fuzzy', threshold: float = 0.75,
                        num_matches: int = 3):
        """
        Find the most similar strings in keys of passed mapping using either
        fuzzy matching or Levenshtein distance.

        Parameters
        ----------
        string : str
            The input string to compare against the internal database.
        target_mapping : dict
            The mapping from trade_name to ingredients (dict)
        method : str, optional
            The similarity method to use, either 'fuzzy' or 'levenshtein'. Default is 'fuzzy'.
        threshold : float, optional
            The minimum similarity threshold for considering a match. Default is 0.75
        num_matches: int, optional
            Number of top matches to return

        Returns
        -------
        list[tuple]
            A list of up to num_matches tuples, each containing a matching string and its similarity score,
            sorted by similarity in descending order.
        """

        similarities = []

        for key, value in target_mapping.items():  # trade_name : mixture
            if method == 'levenshtein':
                sim = np.round(1 - Levenshtein.distance(string, key) / max(len(string), len(key)), 3)
            elif method == 'fuzzy':
                sim = np.round(fuzz.ratio(string, key) / 100.0, 3)
            else:
                raise ValueError(f"Available options for method are: levenshtein, fuzzy")

            if sim >= threshold:
                similarities.append((key, value, sim))

        similarities.sort(key=lambda x: x[2], reverse=True)

        return similarities[:num_matches]

    @classmethod
    def collect_drugnames(cls):
        """
        Collects and stores all unique drug names from the DrugBank dataset.
        """
        if not isinstance(cls.drugbank, list):
            return
        else:
            cls.drugnames = sorted(set([drug.name for drug in cls.drugbank]))

    def collect_similar(self):
        """
        Collects and stores token mappings based on specific functions in `all_changes`.
        """
        self.changes_mapping = {}

        for function, change in self.all_changes:
            if function in ['extend_by_regex', 'extend_by_string', 'substitute_by_regex', 'substitute_by_string']:
                token, mapping = change
                self.changes_mapping[token.replace('\\s', ' ')] = mapping

    @classmethod
    @lru_cache(maxsize=512)
    def chemical_resolver(cls, token: str):
        """
        Check the validity of a token using the NCI NIH chemical resolver service and
        retrieve its SMILES representation.

        Parameters
        ----------
        token : str
            The chemical name or identifier to be resolved.

        Returns
        -------
        bool
            True if the token was successfully resolved and added to `self.smiles_mapping`,
            False otherwise.
        """

        token_ = token.lower().strip().replace(' ', '%20')
        url = f'https://cactus.nci.nih.gov/chemical/structure/{token_}/smiles'

        try:
            result = urlopen(url, data=None, timeout=5).read().decode()
            cls.smiles_mapping[token] = result
            print(f'Resolved successfully')
            return True
        except (HTTPError, URLError, ConnectionResetError) as err:
            cls.log_error('Connection error', 'chemical_resolver', err)
            return False

    def apply_changes(self, kind: str = 'current'):
        """
        Apply all stored changes to the dataframe, allowing for transferring changes between datasets.

        Parameters
        ----------
        kind : str, optional
            Specifies which changes to apply. 'all' applies all stored changes,
            while 'current' applies only the most recent changes. Default is 'current'.
        """
        original_mode = copy.deepcopy(self.execution_mode)
        self.execution_mode = 'immediate'

        if kind == 'all':
            changes_to_apply = copy.deepcopy(self.all_changes)
        elif kind == 'current':
            changes_to_apply = copy.deepcopy(self.changes)
            self.changes = []
        else:
            raise ValueError(f'Allowed options for kind are: all, current')

        unique_changes = []

        for change in changes_to_apply:
            if change not in unique_changes:
                unique_changes.append(change)

        print(f'> Applying a total of {len(unique_changes)} changes <')
        for name, arguments in tqdm(unique_changes, total=len(unique_changes)):
            self.function_mapping[name](*arguments)

        self.update_datasets()
        self.execution_mode = original_mode

    def update_datasets(self):
        """
        Move all rows with no remaining tokens to process from the main dataframe to the processed dataframe.
        Cleans tokens, removes empty tokens, and resets changes after processing.
        """

        self.df = self.df.reset_index(drop=True)
        self.clean_tokens()
        self.remove_by_string('')  # remove all empty tokens

        cleaned_rows = self.df[self.df[self.token_column].apply(lambda x: len(x) == 0)]

        self.df = self.df[~self.df.index.isin(cleaned_rows.index)]
        self.processed_df = pd.concat([self.processed_df, cleaned_rows], axis=0, ignore_index=True)
        self.df = self.df.reset_index(drop=True)
        self.processed_df = self.processed_df.reset_index(drop=True)

        self.collect_drugnames()
        self.collect_similar()
        self.num_updates += 1

    @classmethod
    def add_drug(cls):
        """
        Create a new Drug object and add it to the internal list of drugs.
        """
        drug = Drug(xml=None)
        cls.drugbank.append(drug)
        cls.drugnames.append(drug.name)

    @classmethod
    @lru_cache(maxsize=512)
    def search_db(cls, token: str, similarity: bool = True):
        """
        Search the internal drugbank database for names, synonyms, products,
        international brands, and mixtures that match the provided token.

        Parameters
        ----------
        token : str
            The token to search for in the drug database.
        similarity : bool
            If no match is found search for the most similar change

        Returns
        -------
        matches: list[str]
            List holding visualise of search
        status: int
            Code number for what kind of match it is:
            - 0 : drug
            - 1 : synonym, product, or mixture
            - 2 : similar name
            - 3 : nothing
        """

        matches = []

        for drug in cls.drugbank:

            if token == drug.name:
                return f'Drug < {token} > found. Capture?', 0  # the name of drug is unique

            # multiple matches for synonyms, brands, and mixtures are allowed
            for synonym in drug.synonyms:
                if token == synonym:
                    matches.append(f'Synonym < {synonym} > for < {drug.name} >')

            for brand in sorted(set(drug.products).union(set(drug.international))):
                if token == brand:
                    matches.append(f'Brand < {brand} > for < {drug.name} >')

            for mixture_name, mixture_ingredients in drug.mixtures.items():
                if token == mixture_name:
                    matches.append(f'Mixture < {mixture_name} > made of < {" | ".join(mixture_ingredients)} >')

        if matches:
            return matches, 1  # something was found in DrugBank

        if similarity:
            if output := cls.list_similarity(token, cls.drugnames, 'fuzzy', 0.80, 3):
                matches.extend([f'The similarity of < {token} > to < {match[0]} > : {match[1]}' for match in output])
                return matches, 2
            else:
                return [], 3

        return [], 3

    @classmethod
    @lru_cache(maxsize=512)
    def search_pme(cls, token: str, similarity: bool = True):
        """
        Search the Pharmaceutical Manufacturing Encyclopedia (PME) for the provided token.

        Parameters
        ----------
        token : str
            The token to search for in the PME database.
        similarity : bool
            If no exact match is found search for the most similar change

        Returns
        -------
        str or None
            A message indicating the matches found in PME for the token, or None
            if no matches are found.
        """
        matches = []

        entry = cls.pme.get(token)
        if entry:
            return [f'The < {token} > matches < {" | ".join(entry)} >']

        if similarity:
            if output := cls.dict_similarity(token, cls.pme, 'fuzzy', 0.85, 3):
                for match in output:
                    matches.append(f'The similarity to < {match[0]} > mapped to < {" | ".join(match[1])} > : {match[2]}')

        return matches

    @classmethod
    @lru_cache(maxsize=512)
    def search_chembl(cls, token: str, similarity: bool = True):
        """
        Search the ChEMBL 34 database for the provided token.

        Parameters
        ----------
        token : str
            The token to search for in the PME database.
        similarity : bool
            If no exact match is found search for the most similar change

        Returns
        -------
        str or None
            A message indicating the matches found in PME for the token, or None
            if no matches are found.
        """
        matches = []

        entry = cls.chembl.get(token)
        if entry:
            return [f'The < {token} > matches < {" | ".join(entry)} >']

        if similarity:
            output = cls.dict_similarity(token, cls.chembl, 'fuzzy', 0.85, 3)
            for match in output:
                matches.append(f'The similarity to < {match[0]} > mapped to < {" | ".join(match[1])} > : {match[2]}')

        return matches

    def search_queries(self, token: str, search: str = 'exact', similarity: bool = True):
        """
        Search previously applied changes for matches with the provided token.

        Parameters
        ----------
        token : str
            The token to search for.
        search : str, optional
            The search method to apply when looking for the token in previous queries.
            Options are:
                - 'exact': Only matches if the token exactly matches a previous query.
                - 'full': Matches if the token is found within any part of a previous query.
        similarity : bool
            If no exact match is found search for the most similar change

        Returns
        -------
        list[str]
            A list of messages reporting found matches. Each message indicates the
            nature of the match in relation to the previous queries.
        """

        matches = []

        plain_token = self.plain_string(token)
        for i, (function, change) in enumerate(self.all_changes):
            if any([plain_token == change[0], token == change[0]]):
                matches.append(f"Token matches previous query < {self.all_changes[i]} >")
            elif any([re.search(plain_token, change[0]) is not None, token in change[0]]) and search == 'full':
                matches.append(f"Token found in previous query: < {self.all_changes[i]} >")
            else:
                continue

        if matches:
            return list(set(matches))

        if similarity:
            output = self.dict_similarity(token, self.changes_mapping, 'fuzzy', 0.85, 3)
            for match in output:
                matches.append(f'The similarity to < {match[0]} > mapped to < {match[1]} > : {match[2]}')

        return matches

    @classmethod
    @lru_cache(maxsize=512)
    def search_pubchem(cls, token: str, min_length: int = 4):
        """
        Search PubChem for substances matching the provided token and retrieve synonyms.

        Parameters
        ----------
        token : str
            The token to search for in PubChem. It could be a drug name or any related term.
        min_length : int
            The minimal length of a string. Used to skip ambiguous tokens.

        Returns
        -------
        list[str]
            A list of messages reporting found matches or synonyms related to the token.
            If the token matches a drug name in the internal drugbank, a message is returned.
            If no matches are found, a message is returned containing all collected synonyms
            related to the token from PubChem.
        """
        def frac_numeric(string: str):
            return float(np.sum([char.isnumeric() for char in string])) / len(string)

        if not len(token) >= min_length:
            return tuple()

        matches = []
        all_synonyms = []
        try:
            substances = pubchempy.get_substances(token, namespace='name')

        except (TimeoutError, HTTPError, ConnectionResetError) as err:
            cls.log_error('Connection error', 'search_pubchem', str(err))
            substances = []

        except Exception as err:
            cls.log_error('Unexpected error', 'search_pubchem', str(err))
            substances = []

        for substance in substances:  # iterate over all retrieved entries and collect all synonyms
            pattern = re.compile(r'(?=.*[a-zA-Z])\A[\da-zA-Z\s\-,.]+\Z')  # skip fully chemical names
            synonyms = [syn.strip().lower() for syn in substance.synonyms
                        if (frac_numeric(syn) <= 0.7 and re.search(pattern, syn) is not None)]  # new in 1.0.0
            if synonyms:
                all_synonyms.extend(synonyms)

        all_synonyms = sorted(set(all_synonyms))

        for synonym in all_synonyms:  # check if one of the synonyms matches drug name
            if synonym in cls.drugnames:
                matches.append(f'The < {token} > matches < {synonym} >')

        if not matches:  # if no drug is matched, retain information about all synonyms
            if all_synonyms:
                matches.append(f'The < {token} > has the following synonym/s: < {all_synonyms} >')

        return tuple(matches)

    @classmethod
    @lru_cache(maxsize=512)
    def search_rxreasoner(cls, token: str, min_length: int = 4):
        """
        Search through the RxReasoner website using the REST API for drug information.

        Parameters
        ----------
        token : str
            The token to search for in RxReasoner.
        min_length : int
            The minimal length of a string. Used to skip ambiguous tokens.

        Returns
        -------
        str or None
            A message indicating whether matches were found for the token in RxReasoner,
            along with a list of matched drugs. If no matches are found, returns None.
        """

        if not len(token) >= min_length:
            return None

        token_ = token.lower().strip().replace(' ', '%20')
        url = f'https://www.rxreasoner.com/drugs/{token_}'

        try:
            response = urlopen(url, data=None, timeout=8)

            drugs = []
            html_content = response.read().decode('utf-8')
            tree = html.fromstring(html_content)
            api = tree.xpath(".//a[starts-with(@href, '/substances') and .//strong]")

            for active in api:
                drugs.append(active.xpath(".//strong")[0].text_content().strip().lower())

            if drugs:
                return f"The < {token} > matches < {' | '.join(drugs)} >"
            else:
                return None

        except (TimeoutError, HTTPError, ConnectionResetError) as err:
            cls.log_error('Connection error', 'search_rxreasoner', str(err))
            return None

        except Exception as err:
            cls.log_error('Unexpected error', 'search_rxreasoner', str(err))
            return None

    @classmethod
    @lru_cache(maxsize=512)
    def search_ndrugs(cls, token: str, headers: str = 'full', min_length: int = 4):
        """
        Search through the ndrugs.com website for drug information.

        Parameters
        ----------
        token : str
            The token to search for in ndrugs.com. This should be the name of a drug or active substance.
        headers : str or dict, optional
            The HTTP headers to use for the request. Options are 'main', 'alt', 'full', or a custom dictionary of headers.
            Default is 'full'.
        min_length : int
            The minimal length of a string. Used to skip ambiguous tokens.

        Returns
        -------
        str or None
            A message indicating whether matches were found for the token on ndrugs.com,
            along with a list of matched drugs. If no matches are found, returns None.
        """

        if not len(token) >= min_length:
            return None

        token_ = token.lower().strip().replace(' ', '%20')
        url = f'https://www.ndrugs.com/?s={token_}'
        hd = {'user-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0'}
        alt_hd = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        full_hd = {'authority': 'www.google.com',
                   'accept': 'text/html,application/xhtml+xml,application/xml,q=0.9,image/avif,image/webp',
                   'accept_language': 'en-US,en;q=0.9',
                   'cache-control': 'max-age=0',
                   'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}

        try:
            if headers == 'main':
                response = urlopen(Request(url, headers=hd), data=None, timeout=8)
            elif headers == 'alt':
                response = urlopen(Request(url, headers=alt_hd), data=None, timeout=8)
            elif headers == 'full':
                response = urlopen(Request(url, headers=full_hd), data=None, timeout=8)
            elif isinstance(headers, dict):
                response = urlopen(Request(url, headers=headers), data=None, timeout=8)
            else:
                raise ValueError("Incorrect headers provided")

            drugs = []
            html_content = response.read().decode('utf-8')
            tree = html.fromstring(html_content)
            data = tree.xpath("//p[@class='br' and .//a]")

            for entry in data:
                drugs.append(entry.xpath(".//a")[0].text_content().strip().lower())

            if drugs:
                return f"The < {token} > matches < {' | '.join(drugs)} >"
            else:
                return None

        except (TimeoutError, HTTPError, ConnectionResetError) as err:
            cls.log_error('Connection error', 'search_ndrugs', str(err))
            return None

        except Exception as err:
            cls.log_error('Unexpected error', 'search_ndrugs', str(err))
            return None

    @classmethod
    @lru_cache(maxsize=512)
    def search_pill_in_trip(cls, token: str, headers: str = 'full', min_length: int = 4):
        """
        Search through the pillintrip.com website for drug information.

        Parameters
        ----------
        token : str
            The token to search for in pillintrip.com, typically the name of a medicine.
        headers : str or dict, optional
            The HTTP headers to use for the request. Options are 'main', 'alt', 'full',
            or a custom dictionary of headers. Default is 'full'.
        min_length : int
            The minimal length of a string. Used to skip ambiguous tokens.

        Returns
        -------
        str or None
            A message indicating whether matches were found for the token on pillintrip.com,
            along with a list of matched drugs. If no matches are found, returns None.
        """

        if not len(token) >= min_length:
            return None

        token_ = token.lower().strip().replace(' ', '%20')
        url = f'https://www.pillintrip.com/medicine/{token_}'
        hd = {'user-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0'}
        alt_hd = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        full_hd = {'authority': 'www.google.com',
                   'accept': 'text/html,application/xhtml+xml,application/xml,q=0.9,image/avif,image/webp',
                   'accept_language': 'en-US,en;q=0.9',
                   'cache-control': 'max-age=0',
                   'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}

        try:
            if headers == 'main':
                response = urlopen(Request(url, headers=hd), data=None, timeout=8)
            elif headers == 'alt':
                response = urlopen(Request(url, headers=alt_hd), data=None, timeout=8)
            elif headers == 'full':
                response = urlopen(Request(url, headers=full_hd), data=None, timeout=8)
            elif isinstance(headers, dict):
                response = urlopen(Request(url, headers=headers), data=None, timeout=8)
            else:
                raise ValueError("Incorrect headers provided")

            drugs = []
            html_content = response.read().decode('utf-8')
            tree = html.fromstring(html_content)
            data = tree.xpath("//div[@class='med_info_comp_block' and //a]")[0].xpath(".//a")

            for entry in data:
                drugs.append(entry.text_content().strip().lower())

            if drugs:
                return f"The < {token} > matches < {' | '.join(drugs)} >"
            else:
                return None

        except (TimeoutError, HTTPError) as err:
            cls.log_error('Connection error', 'search_pill_in_trip', str(err))
            return None

        except Exception as err:
            cls.log_error('Unexpected error', 'search_pill_in_trip', str(err))
            return None

    def auto_process(self):
        """
        Go through every entry and check if it's not a drug
        """
        self.clean_tokens()
        self.collect_tokens()

        for name, token in self.tokens:

            key = None

            if token in self.automatic_changes.keys():
                key = token
            elif self.plain_string(token) in self.automatic_changes.keys():
                key = self.plain_string(token)

            if key is not None:
                function, arguments = self.automatic_changes[key]
                self.function_mapping[function](*arguments)
                print(f'\n| Auto-executing {function} with {arguments} |')
                time.sleep(0.05)
                continue

    def process(self, exec_mode: str = 'delayed', auto_mode: bool = True, search_queries: str = 'exact',
                search_pubchem: bool = True, search_rxreasoner: bool = True, search_ndrugs: bool = False,
                search_pillintrip: bool = True, search_pme: bool = True, search_chembl: bool = True):
        """
        Start an interactive session for processing tokens with various search options.

        This method allows users to input tokens and perform various actions based on the specified parameters.
        Possible input options are displayed after typing help.

        Parameters
        ----------
        exec_mode : str, optional
            Option to either execute the changes immediately ('immediate') or stack them in changes
            to be executed after finishing the session ('delayed'). Default is 'delayed'.

        auto_mode : bool, optional
            Option to save executed commands and automatically use them when the same tokens are encountered.
            Default is True.

        search_queries : str | None, optional
            Option to search for the current token in previous queries:
                - None: skip search
                - 'exact': print only if the token exactly matches a previous query
                - 'full': print if the token is found within a previous query.
            Default is 'exact'.

        search_pubchem : bool, optional
            If True, use the PubChem API to search for a synonym of the provided token. Default is True.

        search_rxreasoner : bool, optional
            If True, use the REST API to search for the token on the RxReasoner website. Default is True.

        search_ndrugs : bool, optional
            If True, attempt to retrieve information from ndrugs.com. Default is False due to anti-access protection.

        search_pillintrip : bool, optional
            If True, attempt to retrieve information from pillintrip.com. Default is True.

        search_pme : bool, optional
            If True, check the internal Pharmaceutical Manufacturing Encyclopedia for the entry. Default is True.

        search_chembl : bool, optional
            If True, check the ChEMBL 34 database for the entry. Default is True.
        """

        self.clean_tokens()
        self.collect_tokens()

        self.execution_mode = exec_mode
        self.automatic_mode = auto_mode

        for name, token in self.tokens:
            if (len(self.changes) % 16 == 0) and (self.execution_mode == 'delayed'):
                print(f'\n<<< {len(self.changes)} changes reached. Saving. >>>')

            db_matches, db_status = self.search_db(token)

            if self.automatic_mode:
                key = None
                if token in self.automatic_changes.keys():
                    key = token
                elif self.plain_string(token) in self.automatic_changes.keys():
                    key = self.plain_string(token)

                if key is not None:
                    function, arguments = self.automatic_changes[key]
                    self.function_mapping[function](*arguments)
                    print(f'\n| Auto-executing {function} with {arguments} |')
                    time.sleep(0.05)
                    continue

            print(f'\nCurrent token: < {token} >')
            print(f'Original name: < {name} >\n')

            if db_status == 0:  # match is a drug - no need to print anything more
                print('---< DrugBank >---')
                print(db_matches)

            elif db_status == 1:  # synonym, brand, or mixture found | don't consider other options
                print('---< DrugBank >---')
                for match in db_matches:
                    print(match)

            elif db_status in [2, 3]:  # similar name / nothing was found | check secondary options
                if db_status == 2:
                    print('---< DrugBank >---')
                    for match in db_matches:
                        print(match)

                if search_pme:  # secondary source
                    if output := self.search_pme(token):
                        print('\n---< PME >---')
                        for match in output:
                            print(match)

                if search_chembl:  # secondary source
                    if output := self.search_pme(token):
                        print('\n---< ChEMBL >---')
                        for match in output:
                            print(match)

                if search_pubchem:  # secondary source | use PubChem API
                    if output := self.search_pubchem(token):
                        print('\n---< PubChem >---')
                        for match in output:
                            print(match)

                if search_queries is not None:  # secondary source
                    if output := self.search_queries(token, search_queries):
                        print('\n---< Previous queries >---')
                        for match in output:
                            print(match)

                if search_rxreasoner:  # tertiary source
                    if (output := self.search_rxreasoner(token)) is not None:
                        print('\n---< RxReasoner >---')
                        print(output)

                if search_ndrugs:  # tertiary source
                    if (output := self.search_ndrugs(token, headers='full')) is not None:
                        print('\n---< nDrugs >---')
                        print(output)

                if search_pillintrip:  # tertiary source
                    if (output := self.search_pill_in_trip(token, headers='full')) is not None:
                        print('\n---< PillInTrip >---')
                        print(output)

            for sub_selection in input().strip().split(','):  # given the information, make a choice

                if sub_selection == 'auto_on':
                    self.automatic_mode = True

                elif sub_selection == 'auto_off':
                    self.automatic_mode = False

                elif sub_selection == 'save':
                    joblib.dump(self, f'DridProcessor_v_{self.num_updates}.joblib')

                if sub_selection == 'ss':
                    if (string := input(f'Provide string to substitute: ')) == '':
                        string = token
                    replace_string = input(f'Provide substitution: ')
                    self.substitute_by_string(string, replace_string)

                elif sub_selection == 'sr':
                    if (regex := input(f'Provide regex to substitute: ')) == '':
                        regex = self.plain_string(token)
                    replace_string = input(f'Provide substitution: ')
                    self.substitute_by_regex(regex, replace_string)

                elif sub_selection == 'rs':
                    if (string := input(f'Provide string to remove: ')) == '':
                        string = token
                    self.remove_by_string(string)

                elif sub_selection == 'rrs':
                    if (string := input(f'Provide string to remove: ')) == '':
                        string = token
                    self.remove_remaining_by_string(string)

                elif sub_selection == 'rr':
                    if (regex := input(f'Provide regex to remove: ')) == '':
                        regex = self.plain_string(token)
                    self.remove_by_regex(regex)

                elif sub_selection == 'ms':
                    if (string := input(f'Provide string to move: ')) == '':
                        string = token
                    self.move_by_string(string)

                elif sub_selection == 'mr':
                    if (regex := input(f'Provide regex to move: ')) == '':
                        regex = self.plain_string(token)
                    self.move_by_regex(regex)

                elif sub_selection == 'cs':
                    if (string := input(f'Provide string to capture: ')) == '':
                        string = token
                    self.capture_by_string(string)

                elif sub_selection == 'cr':
                    if (regex := input(f'Provide regex to capture: ')) == '':
                        regex = self.plain_string(token)
                    self.capture_by_regex(regex)

                elif sub_selection == 'es':
                    if (string := input('Provide string to search: ')) == '':
                        string = token
                    extend_string = input('Provide string to extend: ')
                    self.extend_by_string(string, extend_string)

                elif sub_selection == 'er':
                    if (regex := input('Provide regex to search: ')) == '':
                        regex = self.plain_string(token)
                    extend_string = input('Provide string to extend: ')
                    self.extend_by_regex(regex, extend_string)

                elif sub_selection == 'ut':
                    if (key := input(f'Provide name to use as a key: ')) == '':
                        key = name
                    updated_tokens = input(f'Provide new tokens to be used: ')
                    self.update_tokens(key, updated_tokens)

                elif sub_selection == 'pop':
                    if (key := input(f'Provide a string to remove: ')) == '':
                        key = token
                    entry = self.automatic_changes.get(key)
                    self.automatic_changes.pop(key, None)
                    self.changes = [change for change in self.changes if change != entry]

                elif sub_selection == 'clean':
                    self.clean_tokens()

                elif sub_selection == 'cir':
                    if self.chemical_resolver(token):
                        self.move_by_string(token)

                elif sub_selection == 'rev' and exec_mode == 'delayed':
                    _ = self.changes.pop()  # if the method operates in delayed mode, remove last change
                    print(f'Removed the following change: {_}')
                    time.sleep(1)

                elif sub_selection == 'ex_ap':
                    if exec_mode == 'delayed':
                        self.apply_changes(kind='current')
                    return

                elif sub_selection == 'ex':
                    return

                elif sub_selection == 'help':
                    print(f'The following options are available:')
                    print(f'> ss - substitute_by_string')
                    print(f'> sr - substitute_by_regex')
                    print(f'> rs - remove_by_string')
                    print(f'> rrs - remove_remaining_by_string')
                    print(f'> rr - remove_by_regex')
                    print(f'> ms - move_by_string')
                    print(f'> mr - move_by_regex')
                    print(f'> cs - capture_by_string')
                    print(f'> cr - capture_by_regex')
                    print(f'> es - extend_by_string')
                    print(f'> er - extend_by_regex')
                    print(f'> ut - update tokens')
                    print(f'> clean - clean tokens')
                    print(f'> cir - try to resolve current token using Chemical Resolver.')
                    print(f'> rev - if operating in delayed mode, revert last change')
                    print(f'> pop - remove a string from automatic changes')
                    print(f'> ex - end the session')
                    print(f'> ex_ap - end the session and apply stored changes')
                    print(f'> save - save the current version of DridProcessor')
                    print(f'> auto_on - turn the automatic mode on')
                    print(f'> auto_off - turn the automatic mode off')
                    print(f'Multiple options can be selected by separating them by comma')
                else:
                    continue

            display.clear_output(wait=False)


# The functions below were used to prepare the 18 variants of the dataset
"""
Step one
"""


def age_months_2_age_group(value: float):
    """
    Encode age using the FDA age groups in months:
    [0, 144) Children
    [144, 252) Adolescent
    [252, 780) Adult
    [780, 1200) Elderly
    Negative values or values above 100 years are discarded.

    Parameters
    ----------
    value: float
        Age in months

    Returns
    -------
    Union[str, pd.NA]
    """
    if isinstance(value, NAType):
        return pd.NA

    if value < 0:
        return pd.NA
    elif value < 144:
        return 'Children'
    elif value < 252:
        return 'Adolescent'
    elif value < 780:
        return 'Adult'
    elif value < 1200:
        return 'Elderly'
    else:
        return pd.NA


def string_evaluate(entry):
    """
    Use ast package to safely evaluate a string

    Parameters
    ----------
    entry: str

    Returns
    -------
    Any: depends on what's passed

    """
    if isinstance(entry, str):
        entry = ast.literal_eval(entry)
        return entry
    else:
        return pd.NA

def combine_drugs_inds(drug: List[Tuple], indication: List[Tuple]):
    """
    Combine information about drugs with their indications

    Parameters
    ----------
    drug: List[Tuple[str]]
        A tuple with information about sequence number, assigned role code, and drug name
    indication: List[Tuple[str]]
        A tuple with information about sequence number and indication

    Returns
    -------
    List[Tuple[str]]: A list of combined drug-indication tuples
    """

    if isinstance(drug, NAType):
        return pd.NA

    drug_list = [{'Seq': seq, 'Code': code, 'DrugName': name} for seq, code, name in drug]

    if isinstance(indication, NAType):
        for item in drug_list:
            item['IndiName'] = ''
    else:
        indi_map = {seq: name for seq, name in indication}

        for item in drug_list:
            item['IndiName'] = indi_map.get(item['Seq'], '')

    return [tuple(item.values()) for item in drug_list]


def to_date(value: str):
    """
    Convert string-based representation to datetime object

    Parameters
    ----------
    value: str
        String encoding date

    Returns
    -------
    datetime.date
    """

    if isinstance(value, NAType):
        return pd.NA

    split = value.split('-')
    return datetime.date(year=int(split[0]), month=int(split[1]), day=int(split[2]))


def str_to_numeric(value: str, decimals: int = 3):
    """
    Convert a string to numerical representation

    Parameters
    ----------
    value: str
        A string encoding a numerical value
    decimals: int
        Number of decimal digits to keep.

    Returns
    -------
    float
    """

    if isinstance(value, NAType):
        return pd.NA

    return np.round(float(value), decimals)


def split_reactions(value):
    """
    Split amd process a string of reactions.

    Parameters
    ----------
    value: str
        String encoding mulitple reactions

    Returns
    -------
    List[str]
    """

    if isinstance(value, NAType):
        return pd.NA

    return [string.lower().strip() for string in value.split(':')]


def step_1(df: pd.DataFrame):
    """
    In step 1 the following operations are performed:
    - redundant columns removed
    - strings for drugs, indications, dates, weights, and ages are evaluated
    - drugs are combined with their indications (if present)
    - correct datatypes are assigned

    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe exported from MySQL database

    Returns
    -------
    pd.DataFrame
        A processed dataframe
    """


    df = df.reset_index(drop=True).fillna(value=pd.NA)

    df = df.astype({
        'primaryid': 'UInt64',
        'event_dt': 'string',
        'fda_dt': 'string',
        'sex': 'string',
        'age_months': 'string',
        'age_group': 'string',
        'weight_kg': 'string',
        'reactions': 'string',
        'drugs': 'string',
        'indications': 'string'
    })

    drug_list = [string_evaluate(entry) for entry in df.drugs.tolist()]
    indi_list = [string_evaluate(entry) for entry in df.indications.tolist()]

    drug_indi_list = [combine_drugs_inds(drug, indi) for drug, indi in zip(drug_list, indi_list)]
    df['drug_indi'] = drug_indi_list

    del drug_list, indi_list, drug_indi_list
    df = df.drop(columns=['drugs', 'indications'])

    df['event_dt'] = [to_date(entry) for entry in df.event_dt.tolist()]
    df['fda_dt'] = [to_date(entry) for entry in df.fda_dt.tolist()]

    df['reactions'] = df['reactions'].apply(split_reactions)

    df['age_months'] = [str_to_numeric(value, 3) for value in df['age_months'].tolist()]
    df['weight_kg'] = [str_to_numeric(value, 3) for value in df['weight_kg'].tolist()]

    df['age_group'] = df['age_months'].apply(age_months_2_age_group)

    df = df.astype({'primaryid': 'UInt64', 'sex': 'string', 'age_months': 'Float32',
                    'age_group': 'string', 'weight_kg': 'Float32'})

    return df

"""
Step 2
"""

def filter_primary(entry_list: List[Tuple]):
    """
    Iterate over combined drug-indication values and keep only the Primary Subject drugs

    Parameters
    ----------
    entry_list: Tuple[str]
        A List of tuples holding information about drugs (sequence number, role code, drug name, indication).

    Returns
    -------
    primaries: List[Tuple[str]]
        A list with a single drug entry
    """
    if isinstance(entry_list, NAType):
        return pd.NA

    primaries = []

    for entry in entry_list:
        if entry[1] == 'PS':
            primaries.append(entry)

    if not primaries:
        return pd.NA

    return primaries


def filter_primsec(entry_list):
    """
    Iterate over combined drug-indication values and keep Primary and Secondary Suspect drugs

    Parameters
    ----------
    entry_list: Tuple[str]
        A List of tuples holding information about drugs (sequence number, role code, drug name, indication).

    Returns
    -------
    primaries: List[Tuple[str]]
        A list of unique Primary and Secondary Suspect drugs
    """
    if isinstance(entry_list, NAType):
        return pd.NA

    entries = []
    drugnames = []

    for entry in entry_list:
        if entry[1] == 'PS':
            entries.append(entry)
            drugnames.append(entry[2])
        elif entry[1] == 'SS' and entry[2] not in drugnames:
            entries.append(entry)

    if not entries:
        return pd.NA

    return entries


def count_drugs(entry_list: List[Tuple[str]]):
    """
    Count the numbers of Primary Suspect (PS), Secondary Suspect (SS), Concomitant (C), and Interfering (I) drugs

    Parameters
    ----------
    entry_list: List[Tuple[str]]
        A list of tuple with information about sequence number, role code, drug name, and indication

    Returns
    -------

    """
    dd = {'PS': 0, 'SS': 0, 'C': 0, 'I': 0}
    drugnames = []

    if isinstance(entry_list, NAType):
        return pd.Series(dd)

    for entry in entry_list:
        no, code, drug, indi = entry
        if drug not in drugnames and code in dd.keys():
            dd[code] += 1
            drugnames.append(drug)

    return pd.Series(dd)


def count_reactions(reaction_list: List[str]):
    """
    Count the number of reactions in an entry.

    Parameters
    ----------
    reaction_listL List[str]
        A list with reaction names

    Returns
    -------
    int
    """

    if isinstance(reaction_list, NAType):
        return 0

    return len(reaction_list)


def map_llt(entry: List[str], llt_2_pt: Dict[str, str]):
    """
    Map MedDRA Lowest Level Terms (LLT) to corresponding Preferred Terms (PT).

    Parameters
    ----------
    entry: List[str]
        A list with reaction/outcome names
    llt_2_pt: Dict[str, str]

    Returns
    -------
    List[str]: unique Preferred Terms for reactions/outcomes

    """
    if isinstance(entry, list):
        new_entry = [llt_2_pt.get(value, '') for value in entry]
        while '' in new_entry:
            new_entry.pop('')
        return sorted(set(new_entry))

    return pd.NA


def step_2(df: pd.DataFrame, llt_2_pt: Dict[str, str], max_ps: int = None, max_drug: int = None,
           max_reactions: int = None, filter_type: str = 'primary'):
    """
    In step 2 the following operations are performed:
    - mapping of reactions from LLT to PTs
    - counting of drug types
    - filtering of the dataframe based on set max_ps, max_drug, and max_reactions

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 1
    llt_2_pt: Dict[str, str]
        A mapping from MedDRA LLTs to PTs
    max_ps: int
        Maximum number of allowed Primary Suspect entries
    max_drug: int
        Maximum number of allowed drug entries
    max_reactions: int
        Maximum number of allowed reactions.
    filter_type: str
        Type of filter to use. Either 'primary' or 'primsec'

    Returns
    -------

    """
    df['reactions'] = df['reactions'].apply(map_llt, llt_2_pt=llt_2_pt)

    drug_ct = df.drug_indi.apply(lambda entry: count_drugs(entry))
    drug_ct.columns = ['num_ps', 'num_ss', 'num_c', 'num_i']

    df = pd.concat([df, drug_ct], axis=1)

    if max_ps is not None:
        df = df[df.num_ps <= max_ps]

    # we ignore 'Concomitant' and 'Interfering' compounds as they are not considered later on
    df['num_drug'] = df.num_ps + df.num_ss

    if max_drug is not None:
        df = df[df.num_drug <= max_drug]

    df['num_reac'] = df['reactions'].apply(count_reactions)
    if max_reactions is not None:
        df = df[df.num_reac <= max_reactions]

    df = df[(df.num_drug > 0) & (df.num_reac > 0)]

    if filter_type == 'primary':
        df['drugs'] = df['drug_indi'].apply(filter_primary)
    elif filter_type == 'primsec':
        df['drugs'] = df['drug_indi'].apply(filter_primsec)
    else:
        raise ValueError('Allowed options for < filter_type > are: < primary, primsec >')

    df = df.explode('drugs')
    exp = df.drugs.apply(lambda drugs: pd.Series(drugs) if isinstance(drugs, tuple) else pd.Series([pd.NA]*4))
    exp.columns = ['drug_seq', 'role_code', 'drug_name', 'indication']

    df = pd.concat([df, exp], axis=1)
    df = df.drop(columns=['num_ps', 'num_ss', 'num_c', 'num_i', 'num_drug', 'num_reac',
                          'drug_indi', 'drugs', 'drug_seq'])

    df = df.reset_index(drop=True)
    df['indication'] = df['indication'].replace('', pd.NA)
    df = df.astype({'role_code': 'string', 'drug_name': 'string', 'indication': 'string'})

    return df


"""
Step 3
"""


def prepare_weight_mapping(directory: str):
    """
    Calculate quantile-based ranges for expected weights in various scenarios.

    Parameters
    ----------
    directory: str
        Path to a directory with files from Step 2.

    Returns
    -------
    both_known: pd.Series
        A series with values when both sex and age are known
    sex_unknown: pd.Series
        A series with values when age is known, but sex not
    age_estim_sex_known: pd.Series
        A series with values when sex is known and age is estimated from weight
    age_estim_sex_unknown: pd.Series
        A series with values when sex is unknown and age is estimated from weight
    """
    weights = []

    pattern = os.path.join(directory, 'chunk*.joblib')

    for file in sorted(glob.glob(pattern)):
        df = joblib.load(file)
        weights.append(df[['weight_kg', 'sex', 'age_group']])

    df = pd.concat(weights).reset_index(drop=True)
    df = df[~df['weight_kg'].isna()]

    both_known = df.groupby(['age_group', 'sex'])[df.columns].apply(
        lambda group: np.round(np.quantile(group['weight_kg'].to_numpy(), [0.05, 1/3, 2/3, 0.95]), 3))
    sex_unknown = df.groupby(['age_group'])[df.columns].apply(
        lambda group: np.round(np.quantile(group['weight_kg'].to_numpy(), [0.05, 1/3, 2/3, 0.95]), 3))

    age_estim_sex_known = df.groupby(['sex', 'age_group'])[df.columns].apply(
        lambda group: np.round(np.quantile(group['weight_kg'].to_numpy(), [0.1, 1/3, 2/3, 0.9]), 3))
    age_estim_sex_unknown = df.groupby(['age_group'])[df.columns].apply(
        lambda group: np.round(np.quantile(group['weight_kg'].to_numpy(), [0.1, 1/3, 2/3, 0.9]), 3))

    return both_known, sex_unknown, age_estim_sex_known, age_estim_sex_unknown


def map_gender(value: str):
    """
    Map gender to a corresponding numpy array.
    This step here is redundant, but I didn't want to change it to not break the code.

    Parameters
    ----------
    value: str
        Either 'M' or 'F'

    Returns
    -------
    array: numpy.array
        An array encoding gender
    """
    mapping = {'M': np.array([1, 0], dtype=np.uint8),
               'F': np.array([0, 1], dtype=np.uint8)}

    array = mapping.get(value, np.array([0, 0], dtype=np.uint8))
    return array


def map_age_group(value: str):
    """
    Map age to a corresponding numpy array.
    This step here is redundant, but I didn't want to change it to not break the code.

    Parameters
    ----------
    value: str
        Either 'Children', 'Adolescent', 'Adult', or 'Elderly'

    Returns
    -------
    array: numpy.array
        An array encoding age
    """
    mapping = {'Children': np.array([1, 0, 0, 0], dtype=np.uint8),
               'Adolescent': np.array([0, 1, 0, 0], dtype=np.uint8),
               'Adult': np.array([0, 0, 1, 0], dtype=np.uint8),
               'Elderly': np.array([0, 0, 0, 1], dtype=np.uint8),
               }
    array = mapping.get(value, np.array([0, 0, 0, 0], dtype=np.uint8))
    return array


def map_weight(row: pd.Series, wq_both_known: pd.Series, wq_sex_unknown: pd.Series,
               wq_age_estim_sex_known: pd.Series, wq_age_estim_sex_unknown: pd.Series):
    """
    Map weight to a corresponding numpy array.
    This step here is redundant, but I didn't want to change it to not break the code.

    Parameters
    ----------
    row: pd.Series (from df.iterrows())
        A row containing an entry
    wq_both_known: pd.Series
        A series with weight quantiles when both sex and age are known
    wq_sex_unknown: pd.Series
        A series with weight quantiles when age is known, but sex not
    wq_age_estim_sex_known
        A series with weight quantiles when sex is known and age is estimated from weight
    wq_age_estim_sex_unknown
        A series with weight quantiles when sex is known and age is estimated from weight
    """

    def map_to_array(value_, quantiles_):

        if value_ <= quantiles_[0]:  # value belongs to bottom 1% -> probably an outlier
            return 'Unknown'
        elif value_ <= quantiles_[1]:
            return 'Low'
        elif value_ <= quantiles_[2]:
            return 'Average'
        elif value_ <= quantiles_[3]:
            return 'High'
        else:  # value belongs to top 1% -> probably an outlier
            return 'Unknown'

    def estimate_quantiles(value_, age_estim_):
        matches = []
        for _, quantiles_ in age_estim_.items():
            if min(quantiles_) <= value_ <= max(quantiles_):
                matches.append(quantiles_)

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0]

        elif len(matches) > 1:
            return np.mean(matches, axis=0)

        return None

    weight = row['weight_kg']
    sex = row['sex']
    age_group = row['age_group']

    # we have no weight, nothing to be done here
    if not isinstance(weight, (int, float, np.float32)) or np.isnan(weight):
        return 'Unknown'

    # We know both sex and age_group - best scenario! ~ 75% of known weights
    if isinstance(sex, str) and isinstance(age_group, str):
        quantiles = wq_both_known[age_group][sex]
        array = map_to_array(weight, quantiles)
        return array

    # we know only age: also good, since differences between sexes are not so big
    elif isinstance(age_group, str) and not isinstance(sex, str):
        quantiles = wq_sex_unknown[age_group]
        array = map_to_array(weight, quantiles)
        return array

    # we know only sex: not so good
    elif isinstance(sex, str) and not isinstance(age_group, str):
        quantiles = estimate_quantiles(weight, wq_age_estim_sex_known[sex])

        if quantiles is None:
            return 'Unknown'
        else:
            array = map_to_array(weight, quantiles)
            return array

    # we're clueless
    else:
        quantiles = estimate_quantiles(weight, wq_age_estim_sex_unknown)

        if quantiles is None:
            return 'Unknown'
        else:
            array = map_to_array(weight, quantiles)
            return array


def step_3(df: pd.DataFrame, token_map, wq_both_known, wq_sex_unknown, wq_age_estim_sex_known, wq_age_estim_sex_unknown):
    """
    In step 3 the following operations are performed:
        - removal of redundant columns
        - mapping of weights
        - mapping of FAERS drug descriptions to active ingredients
        - removal of rows with missing reactions or actives

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from Step 2
    token_map: pd.DataFrame
        A DataFrame with drug descriptions mapped to active ingredients
    wq_both_known: pd.Series
        A series with weigth quantiles when both sex and age are known
    wq_sex_unknown: pd.Series
        A series with weight quantiles when age is known, but sex not
    wq_age_estim_sex_known
        A series with weight quantiles when sex is known and age is estimated from weight
    wq_age_estim_sex_unknown
        A series with weight quantiles when sex is known and age is estimated from weight

    Returns
    -------
    df: pd.DataFrame
        A processed DataFrame
    """
    df = df.drop(columns=['event_dt', 'fda_dt'])

    df.loc[:, 'weight'] = [map_weight(row, wq_both_known, wq_sex_unknown, wq_age_estim_sex_known, wq_age_estim_sex_unknown)
                                 for idx, row in df.iterrows()]
    df.loc[:, 'sex'] = df['sex'].replace({pd.NA: 'Unknown'})
    df.loc[:, 'age_group'] = df['age_group'].replace({pd.NA: 'Unknown'})
    df.loc[:, 'weight'] = df['weight'].replace({pd.NA: 'Unknown'})

    df = df.drop(columns=['age_months', 'weight_kg'])

    token_map = token_map[['faers_name', 'active']].rename(columns={'faers_name': 'drug_name'})

    df = df.merge(token_map, on='drug_name', how='left')

    df['active'] = df['active'].replace(np.nan, pd.NA).astype({'active': 'string'})
    df = df.dropna(subset=['reactions', 'active'])

    return df


"""
Step 4
"""


def actives_to_smiles(actives: str, smiles_mapping: Dict[str, str]):
    """
    Map active ingredients to corresponding SMILES strings.

    Parameters
    ----------
    actives: str
        ' : ' separated string of active ingredients
    smiles_mapping: Dict[str, str]
        A mapping from active ingredients to SMILES strings

    Returns
    -------
    mapped_entries: List[str]
    """

    if pd.isna(actives) or not isinstance(actives, str):
        return pd.NA,

    mapped_entries = [smiles_mapping.get(entry, 'Missing') for entry in actives.split(' : ')]

    if set(mapped_entries) != {'Missing'}:
        return mapped_entries
    return pd.NA


def check_complete(entries: List[str]):
    """
    Check if a given entry doesn't contain missing SMILES.

    Parameters
    ----------
    entries: List[str]
        A list with SMILES strings

    Returns
    -------
    int
    """

    if not isinstance(entries, list):
        return 0

    if 'Missing' not in entries:
        return 1

    return 0


def step_4(df: pd.DataFrame, smiles_mapping, smiles_col: str = 'SMILES'):
    """
    In step 4 the following operations are performed:
    - mapping of active ingredients to SMILES strings
    - check for completeness of mapping

    Parameters
    ----------
    df
    smiles_mapping
    smiles_col
    """
    smi_map = {row['Name']: row[smiles_col] for idx, row in smiles_mapping.iterrows()}

    df['SMILES'] = [actives_to_smiles(entry, smi_map) for entry in df.active.tolist()]
    df['complete'] = [check_complete(entry) for entry in df.SMILES.tolist()]

    # remove entries where at least one of the components was missing from smi_map
    df = df[df.complete == 1]
    df = df.dropna(subset='SMILES').reset_index(drop=True)
    df = df.drop(columns=['drug_name', 'indication', 'complete'])
    df = df.rename(columns={'age_group': 'Age', 'sex': 'Sex', 'weight': 'Weight'})

    return df


"""
Step 5
Prepare the files for Disproportionality Analysis
"""


def array_to_value(array: np.ndarray, value_dict: dict):
    """
    Get the non-zero index from array and return corresponding class

    Parameters
    ----------
    array: np.ndarray
        A numpy array
    value_dict: Dict[int, str]
        Mapping from index to demographic factor

    Returns
    -------
    str
    """

    if np.sum(array) == 0:
        return 'Unknown'
    return value_dict.get(list(array).index(1), 'Unknown')


def step_5(df: pd.DataFrame, smi_2_idx: Dict[str, int], pt_2_idx: Dict[str, int]):
    """
    In step 5 the following operations are performed:
    - Encoding of reactions and SMILES strings as numerical

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 4
    smi_2_idx: Dict[str, int]
        A dictionary mapping SMILES to integers
    pt_2_idx: Dict[str, int]
        A dictionary mapping reactions to integers

    Returns
    -------
    df: pd.DataFrame
    """

    df = df.drop(columns=['role_code', 'active', 'primaryid'])

    df.loc[:, 'reac_enc'] = df['reactions'].apply(lambda entry: np.array([pt_2_idx.get(item, np.nan) for item in entry], dtype=np.uint16))
    df.loc[:, 'smi_enc'] = df['SMILES'].apply(lambda entry: np.array([smi_2_idx.get(item, np.nan) for item in entry], dtype=np.uint16))

    df = df.drop(columns=['reactions', 'SMILES'])

    df = df.astype({'Sex': 'string', 'Age': 'string', 'Weight': 'string'})

    df = df.reset_index(drop=True)

    return df
