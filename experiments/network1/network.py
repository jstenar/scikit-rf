import numpy as np
import xarray as xr
from numpy.random import randint, normal

k = 1.3806488e-23

#How do we indicate dimension for matrix-part

def dB(x):
    return 20 * np.log10(abs(x))

def build_freq_array(data, freq):
    f = xr.Coordinate("Freq", freq, attrs=dict(unit="Hz"))
    return xr.DataArray(data, coords=(f,))
	

def build_freq_matrix(data, freq):
    _, n, m = data.shape
    f = xr.Coordinate("Freq", freq, attrs=dict(unit="Hz"))
    i = xr.Coordinate("i", np.arange(1, n + 1))
    j = xr.Coordinate("j", np.arange(1, m + 1))
    return xr.DataArray(data, coords=(f, i, j))

def build_matrix(data):
    n, m = data.shape
    i = xr.Coordinate("i", np.arange(1, n + 1))
    j = xr.Coordinate("j", np.arange(1, m + 1))
    return xr.DataArray(data, coords=(i, j))

def build_row(data):
    n,  = data.shape
    i = xr.Coordinate("i", np.arange(1, n + 1))
    return xr.DataArray(data, coords=(i,))

def build_column(data):
    n,  = data.shape
    j = xr.Coordinate("j", np.arange(1, n + 1))
    return xr.DataArray(data, coords=(j,))

network_kinds = {"S": dict(unit="1", temperature=290, Z0=50),
                 "G": dict(unit="?", temperature=290, Z0=50),
                 }


def eye(N, dtype=np.float):
    a = np.eye(N)
    i = xr.Coordinate("i", np.arange(1, N + 1))
    j = xr.Coordinate("j", np.arange(1, N + 1))
    return xr.DataArray(a, coords=(i, j))


def add_kind(name, unit, Z0):
    out = dict(name=name, unit=unit, Z0=Z0)
    network_kinds[name] = out
    def wrapper(func):
        out["iv_func"] = func
        network_kinds[name] = out
        return func
    return wrapper




@add_kind("Z", unit="Ohm", Z0=50)
def iv_relation_z(N, Z0):
    Pzz = eye(2 * N)
    return Pzz

@add_kind("Y", unit="S", Z0=50)
def iv_relation_y(N, Z0):
    Pzy = build_matrix(np.zeros((2 * N, 2 * N), dtype=np.float64))
    P = eye(N)
    
    Pzy[:N, N:] = P
    Pzy[N:, :N] = P
    Pzy[:N, :N] = 0
    Pzy[N:, N:] = 0
    return Pzy

@add_kind("G", unit="", Z0=50)
def iv_relation_g(N, Z0):
    if N != 2:
        raise ValueError("G parameters are only available for 2-ports")
    Pzg = build_matrix(np.array([[0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 0, 0, 1]]))
    return Pzg


@add_kind("H", unit="", Z0=50)
def iv_relation_h(N, Z0):
    if N != 2:
        raise ValueError("H parameters are only available for 2-ports")
    Pzh = build_matrix(np.array([[1, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0]]))
    return Pzh

@add_kind("ABCD", unit="", Z0=50)
def iv_relation_h(N, Z0):
    if N != 2:
        raise ValueError("ABCD parameters are only available for 2-ports")
    Pza = build_matrix(np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, -1]]))
    return Pza


@add_kind("S", unit="", Z0=50)
def iv_relation_s(N, Z0):
    i = eye(N) / np.sqrt(Z0) / 2
    v = eye(N) * np.sqrt(Z0) / 2
    P = eye(2*N)
    P[:N, :N] = v
    P[:N, N:] = -i
    P[N:, :N] = v
    P[N:, N:] = i
    return P


@add_kind("S", unit="", Z0=50)
def iv_relation_s(N, Z0):
    x = eye(N) * np.sqrt(Z0)
    y = eye(N) / np.sqrt(Z0)
    Pzs = eye(2 * N)
    Pzs[:N, :N] = x
    Pzs[:N, N:] = x
    Pzs[N:, :N] = -x
    Pzs[N:, N:] = x
    return Pzs


@add_kind("T", unit="", Z0=50)
def iv_relation_t(N, Z0):
    if N != 2:
        raise ValueError("T parameters are only available for 2-ports")
    #Currently assuming Z0 same on all ports
    Pout = iv_relation_s(N, Z0)
    Pzs = iv_relation_s(N, Z0)
    Pst = build_matrix(np.array([[1, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0]]))
    P = Pzs.data @ Pst.data
    Pout[...] = P
    return Pout


class Network:

    def getdoc(self):
        return """Available multiport formats: %r""" % tuple(network_kinds.keys())

    
    def __init__(self, Smatrix, Cmatrix=None, portimpedance=None,
                       networkkind="S", temperature=None, **kw):
        if networkkind not in network_kinds:
           raise ValueError("Unknown network kind: %r" % networkkind)
        if temperature is None:
            _temperature = network_kinds[networkkind].get("temperature", 290),
        else:
            _temperature = temperature
        _unit = network_kinds[networkkind].get("unit", ""),
        attrs = dict(networkkind=networkkind, temperature=_temperature, unit=_unit, )
        datadict = dict(matrix=Smatrix, portimpedance=50)
        if Cmatrix is not None:
            datadict["Cmatrix"] = Cmatrix
        self.dataset = xr.Dataset(datadict, attrs=attrs)
        self.comments = []


    @property
    def kind(self):
        return self.dataset.networkkind


    def __matmul__(self, other):
        A, B = xr.align((self.convert("T").dataset.matrix,
                         other.convert("T").dataset.matrix))
        m = np.array(A.dataset.matrix) @ np.array(B.dataset.matrix)
        result = A.copy()
        result.dataset["matrix"][...] = m
        result.dataset.attrs["networkkind"] = "T"
        return result.convert(self.dataset.networkkind)

    def copy(self):
        if "Cmatrix" in self.dataset:
            N = Network(self.dataset.matrix.copy(), self.dataset["Cmatrix"].copy())
        else:
            N = Network(self.dataset.matrix.copy())
        N.dataset.attrs = self.dataset.attrs
        return N

    def conversion_matrices(self, kind):
        M = np.array(self.dataset.matrix)
        Pz_other = network_kinds[kind]["iv_func"](M.shape[-1], self.dataset.portimpedance)
        Pz_self = network_kinds[self.dataset.networkkind]["iv_func"](M.shape[-1], self.dataset.portimpedance)
        Pother_self = np.linalg.inv(Pz_other) @ np.array(Pz_self)
        M = np.array(self.dataset.matrix)
        Pz_other = network_kinds[kind]["iv_func"](M.shape[-1], self.dataset.portimpedance)
        Pz_self = network_kinds[self.dataset.networkkind]["iv_func"](M.shape[-1], self.dataset.portimpedance)
        Pself_other = np.linalg.inv(Pz_self) @ np.array(Pz_other)
        N = M.shape[-1]
        tau1 = Pself_other[..., :N, :N]
        sigma1 = Pself_other[..., :N, N:]
        tau2 = Pself_other[..., N:, :N]
        sigma2 = Pself_other[..., N:, N:]
        A = np.linalg.inv(tau1 - M @ tau2)
        B = (-sigma1 +  M @ sigma2)
        return Pother_self, A, B
    
    def convert(self, kind):
        if self.dataset.networkkind == kind:
            return self
        Pother_self, A, B = self.conversion_matrices(kind)
        newM = A @ B
        out = self.copy()
        out.dataset.attrs["networkkind"] = kind
        out.dataset["matrix"][...] = newM

        if "Cmatrix" in self.dataset:
            Cm = self.dataset.Cmatrix
            newCm = Cm.copy()
            T = Cm.copy()
            *head, i, j = tuple(range(Cm.ndim))
            order = tuple(head + [j, i])
            T[...] = A
            Th = T.conj()
            th = np.array(Th.data).transpose(*order)
            Th[...] = th
            newCm[...] = T.data @ Cm.data @ Th.data
            out.dataset["Cmatrix"] = newCm
        return out

    def noise_parameters(self):
        A = self.convert("ABCD")
        CA = A.dataset.Cmatrix / (2 * k * 290)
        CA11 = CA[..., 0, 0]
        CA12 = CA[..., 0, 1]
        CA22 = CA[..., 1, 1]
        Rn = CA11.real
        Yopt = (np.sqrt(CA22 / CA11 - (CA12.imag / CA11) ** 2) +
                    1j * (CA12.imag / CA11))
        Fmin = 1 + (CA12 + CA11 * Yopt.conj()).real 
        Gopt = (1 / 50. - Yopt) / (1 / 50. + Yopt)
        return xr.Dataset(dict(Rn=Rn, Yopt=Yopt, Fmin=Fmin, Gopt=Gopt))
        


        
m = np.zeros((5,2,2), dtype=np.complex128) + 1
m[..., 1, 0] = 0.1
m[..., 0, 1] = 0.1
    
c = xr.Dataset({"x": build_freq_matrix(m[:4], np.linspace(0, 3e9, 4))})

Sa = build_freq_matrix(m[:4], np.linspace(0, 3e9, 4))
Sb = build_freq_matrix(m, np.linspace(0, 4e9, 5))

sa = Network(Sa)
sb = Network(Sb)

ta = Network(Sa)
ta.dataset.attrs["networkkind"] = "T"


shape = (5, 2, 2)
M = build_freq_matrix(normal(size=shape) + normal(size=shape)*1j, np.linspace(0, 3e9, 5))

for kind1 in network_kinds:
    A = Network(M, networkkind=kind1)
    for kind2 in network_kinds:
        
        B = A.convert(kind2).convert(kind1)
        if not np.allclose(A.dataset.matrix, B.dataset.matrix):
            raise Exception("Conversion failed: %s->%s->%s" % (kind1, kind2, kind1))
else:
    print("Conversion tests successful")
            
