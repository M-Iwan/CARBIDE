from typing import Union, List
import numpy as np
import time
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import pubchempy as pcp
import requests


def cactus_resolver(name: str, resolver: str = 'smiles', sleep_time: int = 6, max_retries: int = 3, verbose: int = 0):
    """
    Resolve chemical identity of a single molecule using NCI/NIH CIR API

    Parameters
    ---------------
    name : str
        Name of a molecule
    resolver : str
        Type of output. available are: smiles, stdinchi, stdinchikey, cas, iupac_name
    sleep_time : int
        Time to wait between attempting a retry
    max_retries : int
        Maximum number of attempts to make
    verbose : int
        Option to control verbosity level. Set to 0 for no logging.
    """

    name = name.lower().strip().replace(' ', '%20')
    url = f'https://cactus.nci.nih.gov/chemical/structure/{name.lower()}/{resolver.lower()}'

    try:
        out = urlopen(url, data=None, timeout=300).read().decode()
        return out
    except HTTPError as e:
        if verbose > 0:
            print(f"Failed to resolve [{name.replace('%20', ' ')}] due to {e}")
        return np.nan
    except URLError or ConnectionResetError as e:
        if verbose > 0:
            print(f"Connection lost while resolving [{name.replace('%20', ' ')}] "
                  f"due to {e}. Retrying in {sleep_time} seconds.")
        retries = 1
        while retries <= max_retries:
            try:
                time.sleep(sleep_time)
                sleep_time += 2
                out = urlopen(url, data=None).read().decode()
                if verbose > 0:
                    print(f"[{name.replace('%20', ' ')}] solved successfully")
                return out
            except HTTPError as e:
                if verbose > 0:
                    print(f"Failed to resolve [{name.replace('%20', ' ')}] "
                          f"after {retries} attempts due to {e}")
                return np.nan
            except (URLError, ConnectionResetError) as e:
                if verbose > 0:
                    print(f"Connection lost while resolving [{name.replace('%20', ' ')}] "
                          f"due to {e}. Retrying in {sleep_time} seconds.")
            retries += 1
        if verbose > 0:
            print(f"Failed to resolve [{name.replace('%20', ' ')}] after {max_retries} attempts")
        return np.nan


def pubchem_resolver(name: str, resolver: Union [str, list], sleep_time: int = 6, max_retries: int = 3):
    """
    Resolve chemical identity of a single molecule using PubChem API

    Parameters
    ---------------
    name   : str
             name of a molecule
    resolver : Union[str | list]
             type of output. available are: smiles, stdinchi, stdinchikey, cas, iupac_name
    sleep_time : int
                 time to wait between attempting a retry
    max_retries : int
                  maximum number of attempts to make
    """

    name = name.lower().strip().replace(' ', '%20')

    try:
        out = pcp.get_properties(resolver, name, 'name')
        if out:
            return out[0]
        else:
            return {}
    except HTTPError as e:
        print(f"Failed to resolve [{name.replace('%20', ' ')}] due to {e}")
        return {}
    except URLError or ConnectionResetError as e:
        print(
            f"Connection lost while resolving [{name.replace('%20', ' ')}] due to {e}. "
            f"Retrying in {sleep_time} seconds.")
        retries = 1
        while retries <= max_retries:
            try:
                time.sleep(sleep_time)
                sleep_time += 2
                out = pcp.get_properties(resolver, name, 'name')
                if out:
                    print(f"[{name.replace('%20', ' ')}] solved successfully")
                    return out
                else:
                    return {}
            except HTTPError as e:
                print(f"Failed to resolve [{name.replace('%20', ' ')}] after {retries} attempts due to {e}")
                return {}
            except (URLError, ConnectionResetError) as e:
                print(f"Connection lost while resolving [{name.replace('%20', ' ')}] due to {e}. "
                      f"Retrying in {sleep_time} seconds.")
                pass
            retries += 1
        print(f"Failed to resolve [{name.replace('%20', ' ')}] after {max_retries} attempts")
        return {}


def proxy_cactus_resolver(names: Union[str, List[str]], proxies: dict, resolver: str = 'smiles',
                          timeout: int = 6, verbose: int = 0) -> dict:
    """
    Resolve chemical identity of a molecule using NCI/NIH CIR API.

    Uses the Chemical Identifier Resolver (CIR) web service to convert various
    chemical identifiers to the specified format.

    Parameters
    ----------
    names: str
        Name of the molecule to resolve (or a list of names).
    proxies: dict
        Dictionary containing proxy configuration for HTTP/HTTPS requests
    resolver: str, optional
        Output format requested. Available options: 'smiles', 'stdinchi',
        'stdinchikey', 'cas', 'iupac_name'. Default is 'smiles'.
    timeout: int, optional
        Request timeout in seconds. Default is 6.
    verbose: int, optional
        Verbosity level. Set to 0 for silent operation. Default is 0.

    Returns
    -------
    str or None
        The resolved chemical identifier in requested format if successful,
        None if resolution failed.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: Technically any
    """

    resolver = resolver.lower().strip()

    if not isinstance(names, (str, list)):  # check type
        raise TypeError('Argument < names > must be a string or a list of strings')
    if isinstance(names, list) and not all(isinstance(name, str) for name in names):
        raise TypeError('When < names > is a list, all elements must be strings')

    input_names = [names] if isinstance(names, str) else names

    outputs = {}
    session = requests.Session()

    for orig_name in input_names:
        name = orig_name.lower().strip().replace(' ', '%20')
        url = f'https://cactus.nci.nih.gov/chemical/structure/{name}/{resolver}'
        try:
            response = session.get(url, timeout=timeout, proxies=proxies)
            if (code := response.status_code) != 200:
                if verbose > 0:
                    print(f'Failed with status code {code} for {orig_name}')
                outputs[orig_name] = None
                continue
            outputs[orig_name] = response.text.strip()

        except requests.exceptions.RequestException as e:
            if verbose > 0:
                print(f'Request exception: {e}')
            outputs[orig_name] = None
    return outputs


def proxy_pubchem_resolver(idxs: Union[str, List[str]], proxies: dict,
                           idx_type: str, timeout: int = 6, verbose: int = 0):
    """
    Retrieve all available properties for a compound from PubChem using its name or CID.
    Passing multiple values is supported.

    Parameters
    ----------
    idxs: Union[str, List[str]]
        Name(s) of the compound or PubChem CID(s)
    proxies : dict
        Dictionary containing proxy configuration for HTTP/HTTPS requests
    idx_type: str
        'cid' or 'name'
    timeout : int, optional
        Request timeout in seconds. Default is 6.
    verbose : int, optional
        Verbosity level. Set to 0 for silent operation. Default is 0.

    Returns
    -------
    dict
        Dictionary containing all available properties

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: Any  # technically
    """

    if not isinstance(idxs, (str, list)):  # check type
        raise TypeError('Argument < names > must be a string or a list of strings')
    if isinstance(idxs, list) and not all(isinstance(idx, str) for idx in idxs):
        raise TypeError('When < names > is a list, all elements must be strings')

    property_list = [
        "MolecularFormula", "CanonicalSMILES",
        "IsomericSMILES", "InChI",
        "InChIKey", "IUPACName",
    ]
    property_string = ",".join(property_list)

    input_ids = [idxs] if isinstance(idxs, str) else idxs

    outputs = {}
    session = requests.Session()

    for orig_id in input_ids:
        idx = orig_id.lower().strip().replace(' ', '%20')
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{idx_type}/{idx}/property/{property_string}/JSON'
        try:
            response = session.get(url, timeout=timeout, proxies=proxies)
            if (code := response.status_code) != 200:
                if verbose > 0:
                    print(f'Failed with status code {code} for {orig_id}')
                outputs[orig_id] = None
                continue
            data = response.json()
            try:
                properties = data['PropertyTable']['Properties'][0]
                outputs[orig_id] = properties

            except (KeyError, IndexError) as e:
                if verbose > 0:
                    print(f"Error extracting properties: {e}")
                outputs[orig_id] = None

        except requests.exceptions.RequestException as e:
            if verbose > 0:
                print(f'Request exception: {e}')
            outputs[orig_id] = None

    return outputs