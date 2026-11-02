"""Independent saved-f32 locality KL/TV; no model import or execution."""
import math
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
CUT = Decimal(-1200)
def kl_tv(before, after):
    if len(before)!=len(after) or not before or any(type(x) is not float or not math.isfinite(x) for x in (*before,*after)):
        raise ValueError('finite matched rows required')
    if before==after:
        return {'kl_decimal':'0','tv_decimal':'0','precision':0,'omitted_p':0,'omitted_q':0}
    pa,pb=max(before),max(after)
    spread=min(1200.,max(pa-min(before),pb-min(after)))
    precision=max(220,math.ceil(spread/math.log(10.))+150)
    with localcontext() as ctx:
        ctx.prec=precision;ctx.rounding=ROUND_HALF_EVEN
        ma,mb=Decimal.from_float(pa),Decimal.from_float(pb)
        da=[Decimal.from_float(x)-ma for x in before]
        db=[Decimal.from_float(x)-mb for x in after]
        cache={}
        def exp_tail(v):
            if v<CUT:return Decimal(0)
            if v not in cache:cache[v]=v.exp()
            return cache[v]
        wa=[exp_tail(v) for v in da]
        wb=[exp_tail(v) for v in db]
        za=sum(wa,Decimal(0));zb=sum(wb,Decimal(0))
        la,lb=za.ln(),zb.ln()
        tv=sum((abs(a/za-b/zb) for a,b in zip(wa,wb)),Decimal(0))/2
        kl=sum(((a/za)*((x-la)-(y-lb)) for a,x,y in zip(wa,da,db) if a),Decimal(0))
        return {'kl_decimal':str(kl),'tv_decimal':str(tv),'precision':precision,
                'omitted_p':sum(x<CUT for x in da),'omitted_q':sum(x<CUT for x in db),
                'scope':'omitted weights below exp(-1200); finite precision is a cross-check'}