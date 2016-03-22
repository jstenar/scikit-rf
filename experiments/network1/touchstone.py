import itertools
import re
from itertools import chain
from collections import namedtuple

import numpy as np
import xarray as xr

import network
from network import build_freq_matrix, build_freq_array

k = 1.3806488e-23



class FileFormatError(Exception):
    pass


class ParseError(Exception):
    pass

Token = namedtuple("Token", 'tag lineno rad')



reg_touch = re.compile(r"\s*#\s*[kmgtp]?hz\s+[szygh]\s+" +
                       r"([a-z][a-z])(\s+r\s+[0-9]+)?", re.I)



def unit2mult(x):
    """Creates a unit multiplier from a string *x* containing the
    unit. Assumes a single character means base unit. Otherwise
    first character is used to extract the multiplier:

    ======== =======
     prefix   value
    ======== =======
        a     1e-18
        f     1e-15
        p     1e-12
        n     1e-9
        u     1e-6
        m     1e-3
        k     1e3
        M     1e6
        G     1e9
        T     1e12
        P     1e15
        E     1e18
    ======== =======

        >>> print("%.0e"%unit2mult("nm"))
        1e-09
        >>> print("%.0e"%unit2mult("THz"))
        1e+12
        >>> unit2mult("PHz")==1e15
        True
        >>> unit2mult("EHz")==1e18
        True
"""
    mult = dict(m=1e-3, u=1e-6, n=1e-9, p=1e-12, f=1e-15, a=1e-18,
                k=1e3, M=1e6, G=1e9, T=1e12, P=1e15, E=1e18)
    if len(x) == 1:
        return 1
    try:
        return mult[x[0]]
    except:
        return 1



def is_touchstone(filename, rad):
    return reg_touch.match(rad)



def tokenize(stream):
    for idx, rad in enumerate(stream):
        lineno = idx + 1
        rad = rad.strip()
        if not rad:  # skip empty lines
            continue
        elif rad.startswith("!"):  # Comment line with information
            yield Token("Comments", lineno, rad[1:].strip())
        elif rad.startswith("#"):
            if is_touchstone(None, rad):
                yield Token("Info", lineno, rad[1:].strip())
            else:
                msg = "# format is invalid on line: %s" % lineno
                raise TouchstoneError(msg)
        else:
            yield Token("Data", lineno, rad.split("!", 1)[0].strip())

def read_touchstone(filename):
    with open(filename) as fil:
        comments = []
        info = ""
        data = []
        noisedata = []
        freq = -1
        do_sparameters = True
        for tok in tokenize(fil):
            if tok.tag == "Comments":
                comments.append(tok.rad)
            elif tok.tag == "Info":
                info = tok.rad
            if tok.tag == "Data":
                rowdata = [float(elem) for elem in tok.rad.strip().split()]
                if do_sparameters and (rowdata[0] <= freq):
                    do_sparameters = False
                    
                if do_sparameters:
                    data.append(rowdata)
                    freq = rowdata[0]
                else:
                    noisedata.append(rowdata)
    M = np.array(data)
    freq = M[:, 0]
    matrix = M[:, 1:]
    Nfreq, Nmatrix = matrix.shape
    N = int(np.sqrt(Nmatrix))

    heading = info.strip().split()
    multiplier = unit2mult(heading[0])
    porttype = heading[1]
    format = heading[2].lower()
    if len(heading) > 3:
        Z0 = float(heading[4])
    else:
        Z0 = 1.0


    if format == "ri":
        A = np.fromstring(matrix.tobytes(), dtype=np.complex128)
    elif format == "ma":
        A = matrix[:, ::2] * np.exp(1j * matrix[:, 1::2] / 180 *np.pi)
    elif format == "db":
        A = 10**(matrix[::2] / 20) * np.exp(1j * matrix[1::2] / 180 *np.pi)
    else:
        raise Exception("Unknown format %r, should be one of RI, MA, DB" % format)

    B = A.reshape((Nfreq, N, N)).transpose(0, 2, 1)    
    DX = network.Network(network.build_freq_matrix(B, freq * multiplier), portimpedance=Z0, networkkind=porttype)
    if noisedata:
        ndata = np.array(noisedata)
        freqn = ndata[:, 0] * multiplier
        
        Fmin = 10**(ndata[:, 1] / 10)
        if format == "ri":
            Gopt = ndata[:, 2] * ndata[:, 3]*1j
        elif format == "ma":
            Gopt = ndata[:, 2] * np.exp(1j * ndata[:, 3] / 180 *np.pi)
        elif format == "db":
            B = 10**(ndata[::2] / 20) * np.exp(1j * ndata[1::2] / 180 *np.pi)
        else:
            raise Exception("Unknown format %r, should be one of RI, MA, DB" % format)
        Rn = ndata[:, 4] * Z0
        DA = DX.convert("ABCD")
        DA.dataset["Fmin"] = build_freq_array(Fmin, freqn)
        DA.dataset["Gopt"] = build_freq_array(Gopt, freqn)
        DA.dataset["Yopt"] = build_freq_array(1 / 50. * (1 - Gopt) / (1 + Gopt), freqn)
        DA.dataset["Rn"] = build_freq_array(Rn, freqn)
        CA = DA.dataset.matrix * 0
        CA[..., 0, 0] = DA.dataset.Rn
        CA[..., 1, 0] = 0.5 * (DA.dataset.Fmin - 1) - DA.dataset.Rn * DA.dataset.Yopt
        CA[..., 0, 1] = CA[..., 1, 0].conj()
        CA[..., 1, 1] = DA.dataset.Rn * abs(DA.dataset.Yopt)**2
        DA.dataset["Cmatrix"] = 4 * k * 290 * CA
    else:
        DA = DX
    DA.comments = comments
    return DA.convert(porttype), DA


