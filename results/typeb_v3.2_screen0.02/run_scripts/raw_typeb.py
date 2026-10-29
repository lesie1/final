"""TypeB raw descriptor adapter. No labels or fitted parameters.

Retain the 65 existing raw CE descriptors, with explicit missing bitmap/chip
semantics, plus four counts of events whose bitmap was actually decoded.
Unknown chip IDs are never counted as a physical chip. Missing bitmaps are
not a known zero-error pattern. Counts describe observed events, not a claim
of exhaustive raw coverage. Raw-file absence is handled by the cache builder.
"""
import numpy as np
import raw_events as base

VERSION='typeb_raw_v1_observability_20261029'
RAW_COLUMNS=base.RAW_COLUMNS
FEATURES=base.FEATURES+[f'raw_{w}_bit_observed_events' for w in base.WINDOWS]+['raw_life_bit_observed_events']
events=base.events

def features_at(ev,anchors):
    anchors=np.asarray(anchors,np.int64)
    old=base.features_at(ev,anchors)
    out=np.c_[old,np.full((len(anchors),4),np.nan,np.float32)]
    if ev is None or not len(ev):return out.astype(np.float32)
    t=ev.time.to_numpy(np.int64);hi=np.searchsorted(t,anchors,side='right')
    idx={n:i for i,n in enumerate(FEATURES)}
    bits=np.isfinite(ev.err_bits.to_numpy(float))
    chip=(ev.deviceID.to_numpy()>=0)&(ev.RankId.to_numpy()>=0)
    ct=t[chip]; cc=base._codes(ev.RankId.to_numpy()[chip],ev.deviceID.to_numpy()[chip]) if chip.any() else np.array([],np.int64)
    def count(valid,lo):
        cs=np.r_[0,np.cumsum(valid,dtype=np.int64)];return cs[hi]-cs[lo]
    for w,width in base.WINDOWS.items():
        lo=np.searchsorted(t,anchors-width,side='right');n=count(bits,lo)
        out[:,idx[f'raw_{w}_bit_observed_events']]=n
        for key in ('adjacent_dq_events','multi_dq_events','multi_beat_events','risky_events'):
            out[:,idx[f'raw_{w}_{key}']]=np.where(n>0,out[:,idx[f'raw_{w}_{key}']],np.nan)
        a=np.searchsorted(ct,anchors-width,side='right');b=np.searchsorted(ct,anchors,side='right')
        out[:,idx[f'raw_{w}_distinct_chips']]=np.where(b>a,base.window_distinct(cc,a,b),np.nan)
    n=count(bits,np.zeros(len(anchors),np.int64))
    out[:,idx['raw_life_bit_observed_events']]=np.where(hi>0,n,np.nan)
    for key in ('risky_events','adjacent_dq_events'):
        out[:,idx['raw_life_'+key]]=np.where(n>0,out[:,idx['raw_life_'+key]],np.nan)
    a=np.zeros(len(anchors),np.int64);b=np.searchsorted(ct,anchors,side='right')
    out[:,idx['raw_life_distinct_chips']]=np.where(b>0,base.window_distinct(cc,a,b),np.nan)
    # Do not count the missing-bitmap placeholder as a new error pattern.
    bt=t[bits];bc=ev.pattern.to_numpy()[bits]
    first=base._first_mark(bc);cs=np.r_[0,np.cumsum(first)]
    lo=np.searchsorted(bt,anchors-7*base.DAY,side='right');bh=np.searchsorted(bt,anchors,side='right')
    out[:,idx['raw_life_new_patterns_7d']]=np.where(bh>lo,cs[bh]-cs[lo],np.nan)
    # File existence in the future must not reveal anything before its first event.
    out[hi==0,:]=np.nan
    return out.astype(np.float32)

def mapping_features():
    items=base.mapping_features()
    for item in items:
        item['description']+='; TypeB adapter excludes missing bitmaps/chips from identities; no decoded bits gives unknown pattern counts, not healthy zero'
    return items+[dict(name=n,column=n,kind='numeric_unknown',unit='events',
        description='Number of raw events with an observed decoded error bitmap, over '+n.split('_')[1]+'. Zero means no observed bitmap; not zero CE errors.') for n in FEATURES[len(base.FEATURES):]]
