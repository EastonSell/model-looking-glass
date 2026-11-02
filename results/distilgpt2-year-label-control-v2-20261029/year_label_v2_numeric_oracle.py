"""Independent exact-rational checker for a proposed v2 delivery certificate.

Inputs are fixed raw binary32 site vectors and binary64 planned displacements.
This checks the three numerical inequalities only, not source, model, landing,
untouched coordinates, execution or success of any study.
"""
import math
import struct
from fractions import Fraction


def f32(raw):
    if type(raw) is not bytes or len(raw)%4:raise ValueError('complete f32 bytes required')
    result=[]
    for (bits,) in struct.iter_unpack('<I',raw):
        sign=-1 if bits>>31 else 1;e=(bits>>23)&255;mantissa=bits&0x7fffff
        if e==255:raise ValueError('nonfinite binary32')
        mantissa=mantissa if e==0 else (1<<23)|mantissa
        exponent=-149 if e==0 else e-150
        result.append(sign*(Fraction(mantissa<<exponent,1) if exponent>=0 else Fraction(mantissa,1<<(-exponent))))
    return result


def certificate(q,before,after,displacement):
    if type(q) is not float or not math.isfinite(q) or q<=2**-20:
        raise ValueError('finite positive binary64 dose above minimum required')
    b,z=f32(before),f32(after)
    if len(b)!=len(z) or not b or len(b)!=len(displacement):raise ValueError('vector width')
    if any(type(x) is not float or not math.isfinite(x) for x in displacement):raise ValueError('binary64 displacement')
    q=Fraction.from_float(q);d=[Fraction.from_float(x) for x in displacement]
    D=sum((x*x for x in d),Fraction());R=sum(((x-y)**2 for x,y in zip(z,b)),Fraction())
    E=sum(((x-y-v)**2 for x,y,v in zip(z,b,d)),Fraction())
    lo_d=(q*Fraction(10**12-1,10**12))**2;hi_d=(q*Fraction(10**12+1,10**12))**2
    lo_r=(q*Fraction(4999,5000))**2;hi_r=(q*Fraction(5001,5000))**2;hi_e=(q/5000)**2
    return {'D2':D,'R2':R,'E2':E,'D_ok':lo_d<=D<=hi_d,'R_ok':lo_r<=R<=hi_r,'E_ok':E<=hi_e,
            'pass':lo_d<=D<=hi_d and lo_r<=R<=hi_r and E<=hi_e}
