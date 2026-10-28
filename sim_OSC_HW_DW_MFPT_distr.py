#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import curve_fit
import time
from numba import jit, float64, int32, int64
from numba import njit
from scipy import interpolate
timestr = time.strftime("%y%m%d")
import os
import shutil
from pathlib import Path
#import multiprocessing as mp
#from functools import partial
import sys
import argparse

# In[2]:
import numpy as np
from numba import njit

@njit
def is_transition(x1, x2, barrier):
    return (x1 <= barrier and x2 > barrier) or (x1 > barrier and x2 <= barrier)

@njit
def _add(t, sums, hist, log_edges):
    p = 1.0
    for k in range(len(sums)):          # sums[0] = count, sums[k] = sum t^k
        sums[k] += p
        p *= t
    b = np.searchsorted(log_edges, np.log(t), side='right') - 1
    hist[min(max(b, 0), len(hist) - 1)] += 1   # overflow clipped into last bin

@njit
def _accumulate(x, start, end, pend, npend, sf, active,
                sums_a, hist_a, sums_f, hist_f, log_edges):
    for i in range(len(x) - 1):
        tr_end = is_transition(x[i], x[i + 1], end)
        tr_start = is_transition(x[i], x[i + 1], start)
        # all-FPT: every pending start is paired with the next end crossing
        if tr_end and npend > 0:
            for j in range(npend):
                _add(i - pend[j], sums_a, hist_a, log_edges)
            npend = 0
        if tr_start:
            pend[npend] = i
            npend += 1
        # first-FPT: first start, then first end after it
        if not active:
            if tr_start:
                sf = i
                active = True
        elif tr_end:
            _add(i - sf, sums_f, hist_f, log_edges)
            active = False
    return npend, sf, active


class FPTAccumulator:
    def __init__(self, edges_start, edges_end, kmax=4, nbins=80, tmax_samples=1e10):
        self.pairs = [(s, e) for s in edges_start for e in edges_end]
        P = len(self.pairs)
        self.log_edges = np.linspace(0.0, np.log(tmax_samples), nbins + 1)
        self.sums_a = np.zeros((P, kmax + 1)); self.hist_a = np.zeros((P, nbins), np.int64)
        self.sums_f = np.zeros((P, kmax + 1)); self.hist_f = np.zeros((P, nbins), np.int64)
        self.pend = [np.empty(0) for _ in range(P)]
        self.npend = [0] * P
        self.sf = [0.0] * P
        self.active = [False] * P
        self.last = None

    def update(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        x = X if self.last is None else np.concatenate(([self.last], X))
        n = len(x) - 1
        for p, (s, e) in enumerate(self.pairs):
            need = self.npend[p] + n
            if len(self.pend[p]) < need:                      # grow pending buffer
                new = np.empty(max(need, 2 * len(self.pend[p])))
                new[:self.npend[p]] = self.pend[p][:self.npend[p]]
                self.pend[p] = new
            npend, sf, act = _accumulate(
                x, s, e, self.pend[p], self.npend[p], self.sf[p], self.active[p],
                self.sums_a[p], self.hist_a[p], self.sums_f[p], self.hist_f[p],
                self.log_edges)
            self.pend[p][:npend] -= n                         # re-origin to x[-1]
            self.npend[p], self.sf[p], self.active[p] = npend, sf - n, act
        self.last = x[-1]

    #def save(self, fname, dt_sample):
    #    np.savez(fname, pairs=np.array(self.pairs), dt_sample=dt_sample,
    #             log_edges=self.log_edges,
    #             sums_all=self.sums_a, hist_all=self.hist_a,
    #             sums_first=self.sums_f, hist_first=self.hist_f)
    #def save_and_reset(self, fname, dt_sample):
    #    self.save(fname, dt_sample)
    #    for a in (self.sums_a, self.hist_a, self.sums_f, self.hist_f):
    #        a[:] = 0          # in-place; pend/sf/active/last untouched
    def save(self, fname, dt_sample, **meta):
        np.savez(fname, pairs=np.array(self.pairs), dt_sample=dt_sample,
                log_edges=self.log_edges,
                sums_all=self.sums_a, hist_all=self.hist_a,
                sums_first=self.sums_f, hist_first=self.hist_f,
                **meta)

    def save_and_reset(self, fname, dt_sample, **meta):
        self.save(fname, dt_sample, **meta)
        for a in (self.sums_a, self.hist_a, self.sums_f, self.hist_f):
            a[:] = 0    

@njit
def is_transition(x1, x2, barrier):
    """ checks whether two points, x1,x2, cross a certain point: barrier.
     'barrier' should not be cunfused with position of barrier (of potential)  """
    if x1 <= barrier and x2 > barrier:
        return True
    if x1 > barrier and x2 <= barrier:
        return True
    return False

@njit
def get_ffpt_single(traj, start, end):
    """ calculates the sum of all times for First-fp.events and counts their ouccurence, 
    one direction only.
    start & end: position at which passage-time-calculation starts & ends  """
    count = 0
    ffpt_sum = 0.0
    start_found = -1
    for i in range(len(traj) - 1):
        if start_found < 0:
            if is_transition(traj[i], traj[i + 1], start):
                start_found = i
        elif is_transition(traj[i], traj[i + 1], end):
            ffpt_sum += i - start_found
            count += 1
            start_found = -1
    return ffpt_sum, count

@njit
def get_fpt_single(traj, start, end):
    """ calculates the sum of all times for All-fp.events and counts their ouccurence, 
    one direction only.
    start & end: position at which passage-time-calculation starts & ends   """
    count = 0
    fpt_sum = 0.0
    end_found = -1 

    # iterate through from end, add all diffs from end found
    # to every start found, update end when found
    for i in range(1, len(traj)):
        i = len(traj) - i - 1
        if is_transition(traj[i], traj[i + 1], end):
            end_found = i
        if end_found >= 0 and is_transition(traj[i], traj[i + 1], start):
            fpt_sum += end_found - i
            count += 1
    return fpt_sum, count


def get_mfpt(trajs, edges_start, edges_end, dt, return_sum=False):
    """ calculates mAfpt from all positions 
    in edges_start to all positions in edges_end, back and forth. """ 
    
    if isinstance(trajs, np.ndarray) and trajs.ndim > 1:
        pass
    elif not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    fpts_sum = np.zeros((len(edges_start), len(edges_end)))
    counter = np.zeros((len(edges_start), len(edges_end)), dtype=np.int64)

    for traj in trajs:
        if isinstance(traj, str):
            traj = np.load(traj)
        for i, start in enumerate(edges_start):
            for j, end in enumerate(edges_end):
                fpt_sum, count = get_fpt_single(traj, start, end)
                fpts_sum[i, j] += fpt_sum
                counter[i, j] += count
    if return_sum:
        return fpts_sum, counter
    mfpts = np.zeros_like(fpts_sum)
    mfpts[counter != 0] = fpts_sum[counter != 0] * dt / counter[counter != 0]
    return mfpts


def get_mffpt(trajs, edges_start, edges_end, dt, return_sum=False):
    """ calculates mFfpt from all positions 
    in edges_start to all positions in edges_end, back and forth. """ 
    if isinstance(trajs, np.ndarray) and trajs.ndim > 1:
        pass
    elif not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    ffpts_sum = np.zeros((len(edges_start), len(edges_end)))
    counter = np.zeros((len(edges_start), len(edges_end)), dtype=np.int64)

    for traj in trajs:
        if isinstance(traj, str):
            traj = np.load(traj)
        for i, start in enumerate(edges_start):
            for j, end in enumerate(edges_end):
                ffpt_sum, count = get_ffpt_single(traj, start, end)
                ffpts_sum[i, j] += ffpt_sum
                counter[i, j] += count
    if return_sum:
        return ffpts_sum, counter
    mffpts = np.zeros_like(ffpts_sum)
    mffpts[counter != 0] = ffpts_sum[counter != 0] * dt / counter[counter != 0]
    return mffpts


# ### Markovian Embedding with a combination of single-exponential and oscillatory-exponential Memory Kernel
# In[3]:
## Double well potential
@jit(nopython=True)
def integrate_exp_osc_only_DW_init(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0, U0):
    m_o = gamma_o*tau_o/2  ; a_o = m_o *(tau_o**-2 + tau_phi**-2) #;a_e = gamma_e/tau_e
    x = np.zeros(int(1.1*(nsteps/stride)),dtype=np.float64)
    #v = np.zeros(int(nsteps/stride),dtype=np.float64)
    x[0]=x0
    #v[0]=v0
        
    fac_rand_o=math.sqrt(2*gamma_tot*gamma_o/dt)        #dimless osci-rand-force
    #fac_rand_e=math.sqrt(2*gamma_tot*gamma_e/dt)      #dimless exp-rand-force 
    
    m = tau_m*gamma_tot
    fac_pot=4*U0

    xx=x0 # full system
    zz=z0 # couples the osc. part
    vv=v0 # full system
    ww=w0 # couples the osc. part
    #yy=y0 # couples the exp. part
    
    for var in range(1,int(1.1*nsteps)):
        xi = random.gauss(0.0,1.0) 
        fr_o = fac_rand_o*xi

       # xi2 = random.gauss(0.0,1.0) 
       # fr_e = fac_rand_e*xi2

        kx1=dt*vv*tau_D
        kz1=dt*ww*tau_D        # couples the osc. part
        #kv1=dt* (tau_D*a_o*(zz-xx) + tau_D*a_e*(yy-xx)- gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kv1=dt* (tau_D*a_o*(zz-xx) - gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kw1=dt*(-tau_D*gamma_o*(ww) + tau_D*a_o*(xx-zz) + gamma_tot * fr_o) /m_o # couples the osc. part 
        #ky1=dt*(-tau_D*a_e*(yy-xx)/ gamma_e +  fr_e)  # couples the exp. part

        x1=xx+kx1/2
        v1=vv+kv1/2
        z1=zz+kz1/2
        w1=ww+kw1/2
        #y1=yy+ky1/2
        
        kx2=dt*v1*tau_D
        kz2=dt*w1*tau_D     
        #kv2=dt*(tau_D*a_o*(z1-x1) + tau_D*a_e*(y1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kv2=dt*(tau_D*a_o*(z1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kw2=dt*(-tau_D*gamma_o*(w1) + tau_D*a_o*(x1-z1) + gamma_tot * fr_o) /m_o
        #ky2=dt*(-tau_D*a_e*(y1-x1) / gamma_e +  fr_e) 
            
        x2=xx+kx2/2
        v2=vv+kv2/2
        z2=zz+kz2/2
        w2=ww+kw2/2
        #y2=yy+ky2/2

        kx3=dt*v2*tau_D
        kz3=dt*w2*tau_D        
        #kv3=dt*(tau_D*a_o*(z2-x2) + tau_D*a_e*(y2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kv3=dt*(tau_D*a_o*(z2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kw3=dt*(-tau_D*gamma_o*(w2) + tau_D*a_o*(x2-z2) + gamma_tot * fr_o) /m_o
        #ky3=dt*(-tau_D*a_e*(y2-x2)/ gamma_e +  fr_e) 

        x3=xx+kx3
        v3=vv+kv3
        z3=zz+kz3
        w3=ww+kw3
        #y3=yy+ky3

        kx4=dt*v3*tau_D
        kz4=dt*w3*tau_D       
        #kv4=dt*(tau_D*a_o*(z3-x3) + tau_D*a_e*(y3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kv4=dt*(tau_D*a_o*(z3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kw4=dt*(-tau_D*gamma_o*(w3) + tau_D*a_o*(x3-z3) + gamma_tot * fr_o) /m_o
        #ky4=dt*(-tau_D*a_e*(y3-x3)/ gamma_e +  fr_e) 
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        zz+= (kz1+(2*kz2)+(2*kz3)+kz4)/6
        ww+= (kw1+(2*kw2)+(2*kw3)+kw4)/6
        #yy+= (ky1+(2*ky2)+(2*ky3)+ky4)/6
        
        if var%stride==0:
            x[int(var/stride)]=xx
            #v[int(var/stride)]=vv
    return x[int(0.1*nsteps/stride):] ,xx,vv,zz,ww

@jit(nopython=True)
def integrate_exp_osc_only_DW_follow(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0,U0):
    m_o = gamma_o*tau_o/2  ; a_o = m_o *(tau_o**-2 + tau_phi**-2) #;a_e = gamma_e/tau_e
    x = np.zeros(int((nsteps/stride)),dtype=np.float64)
    #v = np.zeros(int(nsteps/stride),dtype=np.float64)
    x[0]=x0
    #v[0]=v0
        
    fac_rand_o=math.sqrt(2*gamma_tot*gamma_o/dt)        #dimless osci-rand-force
    #fac_rand_e=math.sqrt(2*gamma_tot*gamma_e/dt)      #dimless exp-rand-force
    m = tau_m*gamma_tot
    fac_pot=4*U0

    xx=x0 # full system
    zz=z0 # couples the osc. part
    vv=v0 # full system
    ww=w0 # couples the osc. part
    #yy=y0 # couples the exp. part
    
    for var in range(1,int(nsteps)):
        xi = random.gauss(0.0,1.0) 
        fr_o = fac_rand_o*xi

       # xi2 = random.gauss(0.0,1.0) 
       # fr_e = fac_rand_e*xi2

        kx1=dt*vv*tau_D
        kz1=dt*ww*tau_D        # couples the osc. part
        #kv1=dt* (tau_D*a_o*(zz-xx) + tau_D*a_e*(yy-xx)- gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kv1=dt* (tau_D*a_o*(zz-xx) - gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kw1=dt*(-tau_D*gamma_o*(ww) + tau_D*a_o*(xx-zz) + gamma_tot * fr_o) /m_o # couples the osc. part 
        #ky1=dt*(-tau_D*a_e*(yy-xx)/ gamma_e +  fr_e)  # couples the exp. part

        x1=xx+kx1/2
        v1=vv+kv1/2
        z1=zz+kz1/2
        w1=ww+kw1/2
        #y1=yy+ky1/2
        
        kx2=dt*v1*tau_D
        kz2=dt*w1*tau_D     
        #kv2=dt*(tau_D*a_o*(z1-x1) + tau_D*a_e*(y1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kv2=dt*(tau_D*a_o*(z1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kw2=dt*(-tau_D*gamma_o*(w1) + tau_D*a_o*(x1-z1) + gamma_tot * fr_o) /m_o
        #ky2=dt*(-tau_D*a_e*(y1-x1) / gamma_e +  fr_e) 
            
        x2=xx+kx2/2
        v2=vv+kv2/2
        z2=zz+kz2/2
        w2=ww+kw2/2
        #y2=yy+ky2/2

        kx3=dt*v2*tau_D
        kz3=dt*w2*tau_D        
        #kv3=dt*(tau_D*a_o*(z2-x2) + tau_D*a_e*(y2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kv3=dt*(tau_D*a_o*(z2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kw3=dt*(-tau_D*gamma_o*(w2) + tau_D*a_o*(x2-z2) + gamma_tot * fr_o) /m_o
        #ky3=dt*(-tau_D*a_e*(y2-x2)/ gamma_e +  fr_e) 

        x3=xx+kx3
        v3=vv+kv3
        z3=zz+kz3
        w3=ww+kw3
        #y3=yy+ky3

        kx4=dt*v3*tau_D
        kz4=dt*w3*tau_D       
        #kv4=dt*(tau_D*a_o*(z3-x3) + tau_D*a_e*(y3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kv4=dt*(tau_D*a_o*(z3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kw4=dt*(-tau_D*gamma_o*(w3) + tau_D*a_o*(x3-z3) + gamma_tot * fr_o) /m_o
        #ky4=dt*(-tau_D*a_e*(y3-x3)/ gamma_e +  fr_e) 
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        zz+= (kz1+(2*kz2)+(2*kz3)+kz4)/6
        ww+= (kw1+(2*kw2)+(2*kw3)+kw4)/6
        #yy+= (ky1+(2*ky2)+(2*ky3)+ky4)/6
        
        if var%stride==0:
            x[int(var/stride)]=xx
            #v[int(var/stride)]=vv

    #print("xx,vv,zz,ww out:",xx,vv,zz,ww)
    
    return x ,xx,vv,zz,ww

@jit(nopython=True)
def integrate_exp_osc_only(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0, y0,U0):
    m_o = gamma_o*tau_o/2  ; a_o = m_o *(tau_o**-2 + tau_phi**-2) #;a_e = gamma_e/tau_e
    x = np.zeros(int(1.1*(nsteps/stride)),dtype=np.float64)
    #v = np.zeros(int(nsteps/stride),dtype=np.float64)
    x[0]=x0
    #v[0]=v0
        
    fac_rand_o=math.sqrt(2*gamma_tot*gamma_o/dt)        #dimless osci-rand-force
    #fac_rand_e=math.sqrt(2*gamma_tot*gamma_e/dt)      #dimless exp-rand-force
    
    m = tau_m*gamma_tot
    fac_pot=4*U0

    xx=x0 # full system
    zz=z0 # couples the osc. part
    vv=v0 # full system
    ww=w0 # couples the osc. part
    #yy=y0 # couples the exp. part
    
    for var in range(1,int(1.1*nsteps)):
        xi = random.gauss(0.0,1.0) 
        fr_o = fac_rand_o*xi

       # xi2 = random.gauss(0.0,1.0) 
       # fr_e = fac_rand_e*xi2

        kx1=dt*vv*tau_D
        kz1=dt*ww*tau_D        # couples the osc. part
        #kv1=dt* (tau_D*a_o*(zz-xx) + tau_D*a_e*(yy-xx)- gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kv1=dt* (tau_D*a_o*(zz-xx) - gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kw1=dt*(-tau_D*gamma_o*(ww) + tau_D*a_o*(xx-zz) + gamma_tot * fr_o) /m_o # couples the osc. part 
        #ky1=dt*(-tau_D*a_e*(yy-xx)/ gamma_e +  fr_e)  # couples the exp. part

        x1=xx+kx1/2
        v1=vv+kv1/2
        z1=zz+kz1/2
        w1=ww+kw1/2
        #y1=yy+ky1/2
        
        kx2=dt*v1*tau_D
        kz2=dt*w1*tau_D     
        #kv2=dt*(tau_D*a_o*(z1-x1) + tau_D*a_e*(y1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kv2=dt*(tau_D*a_o*(z1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kw2=dt*(-tau_D*gamma_o*(w1) + tau_D*a_o*(x1-z1) + gamma_tot * fr_o) /m_o
        #ky2=dt*(-tau_D*a_e*(y1-x1) / gamma_e +  fr_e) 
            
        x2=xx+kx2/2
        v2=vv+kv2/2
        z2=zz+kz2/2
        w2=ww+kw2/2
        #y2=yy+ky2/2

        kx3=dt*v2*tau_D
        kz3=dt*w2*tau_D        
        #kv3=dt*(tau_D*a_o*(z2-x2) + tau_D*a_e*(y2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kv3=dt*(tau_D*a_o*(z2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kw3=dt*(-tau_D*gamma_o*(w2) + tau_D*a_o*(x2-z2) + gamma_tot * fr_o) /m_o
        #ky3=dt*(-tau_D*a_e*(y2-x2)/ gamma_e +  fr_e) 

        x3=xx+kx3
        v3=vv+kv3
        z3=zz+kz3
        w3=ww+kw3
        #y3=yy+ky3

        kx4=dt*v3*tau_D
        kz4=dt*w3*tau_D       
        #kv4=dt*(tau_D*a_o*(z3-x3) + tau_D*a_e*(y3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kv4=dt*(tau_D*a_o*(z3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kw4=dt*(-tau_D*gamma_o*(w3) + tau_D*a_o*(x3-z3) + gamma_tot * fr_o) /m_o
        #ky4=dt*(-tau_D*a_e*(y3-x3)/ gamma_e +  fr_e) 
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        zz+= (kz1+(2*kz2)+(2*kz3)+kz4)/6
        ww+= (kw1+(2*kw2)+(2*kw3)+kw4)/6
        #yy+= (ky1+(2*ky2)+(2*ky3)+ky4)/6
        
        if var%stride==0:
            x[int(var/stride)]=xx
            #v[int(var/stride)]=vv
    return x[int(0.1*nsteps/stride):] #,v #,yy,ww,y2y


# In[4]:
## Harmonic well potential
@jit(nopython=True)
def integrate_exp_osc_only_HW_init(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0, U0):
    m_o = gamma_o*tau_o/2  ; a_o = m_o *(tau_o**-2 + tau_phi**-2) #;a_e = gamma_e/tau_e
    x = np.zeros(int(1.1*(nsteps/stride)),dtype=np.float64)
    #v = np.zeros(int(nsteps/stride),dtype=np.float64)
    x[0]=x0
    #v[0]=v0
        
    fac_rand_o=math.sqrt(2*gamma_tot*gamma_o/dt)        #dimless osci-rand-force
    #fac_rand_e=math.sqrt(2*gamma_tot*gamma_e/dt)      #dimless exp-rand-force
    
    m = tau_m*gamma_tot
    k=8*U0
    fac_pot=  k #4*U0

    xx=x0 # full system
    zz=z0 # couples the osc. part
    vv=v0 # full system
    ww=w0 # couples the osc. part
    #yy=y0 # couples the exp. part
    
    for var in range(1,int(1.1*nsteps)):
        xi = random.gauss(0.0,1.0) 
        fr_o = fac_rand_o*xi

       # xi2 = random.gauss(0.0,1.0) 
       # fr_e = fac_rand_e*xi2

        kx1=dt*vv*tau_D
        kz1=dt*ww*tau_D        # couples the osc. part
        #kv1=dt* (tau_D*a_o*(zz-xx) + tau_D*a_e*(yy-xx)- gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kv1=dt* (tau_D*a_o*(zz-xx) - gamma_tot*fac_pot*xx )/m # full system
        kw1=dt*(-tau_D*gamma_o*(ww) + tau_D*a_o*(xx-zz) + gamma_tot * fr_o) /m_o # couples the osc. part 
        #ky1=dt*(-tau_D*a_e*(yy-xx)/ gamma_e +  fr_e)  # couples the exp. part

        x1=xx+kx1/2
        v1=vv+kv1/2
        z1=zz+kz1/2
        w1=ww+kw1/2
        #y1=yy+ky1/2
        
        kx2=dt*v1*tau_D
        kz2=dt*w1*tau_D     
        #kv2=dt*(tau_D*a_o*(z1-x1) + tau_D*a_e*(y1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kv2=dt*(tau_D*a_o*(z1-x1)- gamma_tot*fac_pot*x1 )/m
        kw2=dt*(-tau_D*gamma_o*(w1) + tau_D*a_o*(x1-z1) + gamma_tot * fr_o) /m_o
        #ky2=dt*(-tau_D*a_e*(y1-x1) / gamma_e +  fr_e) 
            
        x2=xx+kx2/2
        v2=vv+kv2/2
        z2=zz+kz2/2
        w2=ww+kw2/2
        #y2=yy+ky2/2

        kx3=dt*v2*tau_D
        kz3=dt*w2*tau_D        
        #kv3=dt*(tau_D*a_o*(z2-x2) + tau_D*a_e*(y2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kv3=dt*(tau_D*a_o*(z2-x2) - gamma_tot*fac_pot*x2 )/m
        kw3=dt*(-tau_D*gamma_o*(w2) + tau_D*a_o*(x2-z2) + gamma_tot * fr_o) /m_o
        #ky3=dt*(-tau_D*a_e*(y2-x2)/ gamma_e +  fr_e) 

        x3=xx+kx3
        v3=vv+kv3
        z3=zz+kz3
        w3=ww+kw3
        #y3=yy+ky3

        kx4=dt*v3*tau_D
        kz4=dt*w3*tau_D       
        #kv4=dt*(tau_D*a_o*(z3-x3) + tau_D*a_e*(y3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kv4=dt*(tau_D*a_o*(z3-x3) - gamma_tot*fac_pot*x3 )/m
        kw4=dt*(-tau_D*gamma_o*(w3) + tau_D*a_o*(x3-z3) + gamma_tot * fr_o) /m_o
        #ky4=dt*(-tau_D*a_e*(y3-x3)/ gamma_e +  fr_e) 
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        zz+= (kz1+(2*kz2)+(2*kz3)+kz4)/6
        ww+= (kw1+(2*kw2)+(2*kw3)+kw4)/6
        #yy+= (ky1+(2*ky2)+(2*ky3)+ky4)/6
        
        if var%stride==0:
            x[int(var/stride)]=xx
            #v[int(var/stride)]=vv
    return x[int(0.1*nsteps/stride):] ,xx,vv,zz,ww


@jit(nopython=True)
def integrate_exp_osc_only_HW_follow(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0,U0):
    m_o = gamma_o*tau_o/2  ; a_o = m_o *(tau_o**-2 + tau_phi**-2) #;a_e = gamma_e/tau_e
    x = np.zeros(int((nsteps/stride)),dtype=np.float64)
    #v = np.zeros(int(nsteps/stride),dtype=np.float64)
    x[0]=x0
    #v[0]=v0
        
    fac_rand_o=math.sqrt(2*gamma_tot*gamma_o/dt)        #dimless osci-rand-force
    #fac_rand_e=math.sqrt(2*gamma_tot*gamma_e/dt)      #dimless exp-rand-force
    
    m = tau_m*gamma_tot
    k=8*U0
    fac_pot=  k #4*U0
    
    xx=x0 # full system
    zz=z0 # couples the osc. part
    vv=v0 # full system
    ww=w0 # couples the osc. part
    #yy=y0 # couples the exp. part
    
    for var in range(1,int(nsteps)):
        xi = random.gauss(0.0,1.0) 
        fr_o = fac_rand_o*xi

       # xi2 = random.gauss(0.0,1.0) 
       # fr_e = fac_rand_e*xi2

        kx1=dt*vv*tau_D
        kz1=dt*ww*tau_D        # couples the osc. part
        #kv1=dt* (tau_D*a_o*(zz-xx) + tau_D*a_e*(yy-xx)- gamma_tot*fac_pot*(xx**3-xx))/m # full system
        kv1=dt* (tau_D*a_o*(zz-xx) - gamma_tot*fac_pot*xx )/m # full system
        kw1=dt*(-tau_D*gamma_o*(ww) + tau_D*a_o*(xx-zz) + gamma_tot * fr_o) /m_o # couples the osc. part 
        #ky1=dt*(-tau_D*a_e*(yy-xx)/ gamma_e +  fr_e)  # couples the exp. part

        x1=xx+kx1/2
        v1=vv+kv1/2
        z1=zz+kz1/2
        w1=ww+kw1/2
        #y1=yy+ky1/2
        
        kx2=dt*v1*tau_D
        kz2=dt*w1*tau_D     
        #kv2=dt*(tau_D*a_o*(z1-x1) + tau_D*a_e*(y1-x1)- gamma_tot*fac_pot*(x1**3-x1))/m
        kv2=dt*(tau_D*a_o*(z1-x1)- gamma_tot*fac_pot*x1 )/m
        kw2=dt*(-tau_D*gamma_o*(w1) + tau_D*a_o*(x1-z1) + gamma_tot * fr_o) /m_o
        #ky2=dt*(-tau_D*a_e*(y1-x1) / gamma_e +  fr_e) 
            
        x2=xx+kx2/2
        v2=vv+kv2/2
        z2=zz+kz2/2
        w2=ww+kw2/2
        #y2=yy+ky2/2

        kx3=dt*v2*tau_D
        kz3=dt*w2*tau_D        
        #kv3=dt*(tau_D*a_o*(z2-x2) + tau_D*a_e*(y2-x2) - gamma_tot*fac_pot*(x2**3-x2))/m
        kv3=dt*(tau_D*a_o*(z2-x2) - gamma_tot*fac_pot*x2 )/m
        kw3=dt*(-tau_D*gamma_o*(w2) + tau_D*a_o*(x2-z2) + gamma_tot * fr_o) /m_o
        #ky3=dt*(-tau_D*a_e*(y2-x2)/ gamma_e +  fr_e) 

        x3=xx+kx3
        v3=vv+kv3
        z3=zz+kz3
        w3=ww+kw3
        #y3=yy+ky3

        kx4=dt*v3*tau_D
        kz4=dt*w3*tau_D       
        #kv4=dt*(tau_D*a_o*(z3-x3) + tau_D*a_e*(y3-x3) - gamma_tot*fac_pot*(x3**3-x3))/m
        kv4=dt*(tau_D*a_o*(z3-x3) - gamma_tot*fac_pot*x3 )/m
        kw4=dt*(-tau_D*gamma_o*(w3) + tau_D*a_o*(x3-z3) + gamma_tot * fr_o) /m_o
        #ky4=dt*(-tau_D*a_e*(y3-x3)/ gamma_e +  fr_e) 
            
        xx=xx + (kx1+2*kx2+2*kx3+kx4)/6
        vv=vv +(kv1+2*kv2+2*kv3+kv4)/6
        zz+= (kz1+(2*kz2)+(2*kz3)+kz4)/6
        ww+= (kw1+(2*kw2)+(2*kw3)+kw4)/6
        #yy+= (ky1+(2*ky2)+(2*ky3)+ky4)/6
        
        if var%stride==0:
            x[int(var/stride)]=xx
            #v[int(var/stride)]=vv

    #print("xx,vv,zz,ww out:",xx,vv,zz,ww)
    
    return x ,xx,vv,zz,ww

# In[5]:



def savedata(tau_m, tau_phi, tau_o, fpts_tot, coun_tot, ffptsum,fcounts,hist_tot, j,U0):

    base_fpt = f"fptsum U0={U0:.0f} j_trajpatch={j:.0f}, tau_m={tau_m:.5f}, tau_phi={tau_phi:.3f}, tau_o={tau_o:.3f}.txt"
    base_ffpt = f"ffptsum U0={U0:.0f} j_trajpatch={j:.0f}, tau_m={tau_m:.5f}, tau_phi={tau_phi:.3f}, tau_o={tau_o:.3f}.txt"
    base_cnt = f"counts U0={U0:.0f} j_trajpatch={j:.0f}, tau_m={tau_m:.5f}, tau_phi={tau_phi:.3f}, tau_o={tau_o:.3f}.txt"
    base_fcnt = f"fcounts U0={U0:.0f} j_trajpatch={j:.0f}, tau_m={tau_m:.5f}, tau_phi={tau_phi:.3f}, tau_o={tau_o:.3f}.txt"
    base_hist = f"hist U0={U0:.0f} j_trajpatch={j:.0f}, tau_m={tau_m:.5f}, tau_phi={tau_phi:.3f}, tau_o={tau_o:.3f}.txt"

    np.savetxt(pathFPTS / base_fpt,fpts_tot,header=f"N_fac={N_fac}, dt_fac={dt_fac}, stride={stride}")
    np.savetxt(pathFPTS / base_ffpt,ffptsum,header=f"N_fac={N_fac}, dt_fac={dt_fac}, stride={stride}")

    np.savetxt(pathFPTS / base_cnt, coun_tot)
    np.savetxt(pathFPTS / base_fcnt, fcounts)

    np.savetxt(pathPots / base_hist, hist_tot )

# In[6]:


np.seterr(divide = 'ignore') 
def data_with_paral_i(param_tuple):
    tau_m, tau_o, tau_phi, U0 = param_tuple
    tau_m=0.01
    tau_D=1;  U0 = 3 ; L =1;  gamma_tot = tau_D/L**2 ; gamma_e = 0 ; gamma_o = 1-gamma_e 
    nsteps=int(max(tau_m, tau_D, tau_phi, tau_o)*N_fac); step = str(nsteps)[0]+'e'+str(len(str(nsteps))-1)
    dt = min(tau_m, tau_D, tau_phi, tau_o)*dt_fac

    acc = FPTAccumulator([1, -1], [-1, 1])

    #inits
    m_o = gamma_o * tau_o / 2.0
    # 1. Calculate correct coupling constant k_y
    k_y = m_o * (1.0 / tau_o**2 + 1.0 / tau_phi**2)
    # 2. Calculate correct standard deviations
    sigma_z = np.sqrt(1.0 / k_y)
    sigma_w = np.sqrt(1.0 / m_o)

    # 3. Initialize
    x0 = 1.0 
    v0 = random.gauss(0.0, math.sqrt(1.0 / tau_m))
    z0 = random.gauss(x0, sigma_z)  # Centered at x0, correct variance
    w0 = random.gauss(0.0, sigma_w) # Non-zero initial velocity


    #init:
    j=0;
    X,x0,v0,z0,w0 = integrate_exp_osc_only_DW_init(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0,U0)
    np.seterr(divide = 'ignore') 
    hist_unnorm,edges= np.histogram(X, bins=100,range=(-1.6,1.6) , density=False)
    fptsum,counts = get_mfpt(X, [1,-1],[-1,1] , stride*dt, return_sum=True)
    ffptsum,fcounts = get_mffpt(X, [1,-1],[-1,1] , stride*dt, return_sum=True)
    savedata(tau_m,tau_phi,tau_o,fptsum,counts,ffptsum,fcounts,hist_unnorm,j,U0)

    acc.update(X)
    acc.save_and_reset(pathFPTS / f'U0_{U0:.1f}_fpt_0.npz', stride*dt,
        tau_m=tau_m, tau_o=tau_o, tau_phi=tau_phi)

    np.savetxt(pathPots / (timestr + 'edges.txt'), edges)
        
    #patching traj
    for j in range(1,patch_traj):
        X,x0,v0,z0,w0 = integrate_exp_osc_only_DW_follow(nsteps, dt, stride, tau_D, tau_m, tau_phi,tau_o,gamma_tot, gamma_o, x0, z0, v0,w0,U0)  
        np.seterr(divide = 'ignore') 
        hist_unnorm,edges=np.histogram(X, bins=100,range=(-1.6,1.6) , density=False)
        fptsum,counts = get_mfpt(X, [1,-1],[-1,1] , stride*dt, return_sum=True)
        ffptsum,fcounts = get_mffpt(X, [1,-1],[-1,1] , stride*dt, return_sum=True)
        savedata(tau_m,tau_phi,tau_o,fptsum,counts,ffptsum,fcounts,hist_unnorm,j,U0)
        
        acc.update(X)
        acc.save_and_reset(pathFPTS / f'U0_{U0:.1f}_fpt_{j}.npz', stride*dt,
            tau_m=tau_m, tau_o=tau_o, tau_phi=tau_phi)         
    return


if __name__ == "__main__":
    # Define the paths for copying files
    timestr = time.strftime("%y%m%d") 
    pathPar = Path('DATAZ/'+timestr+'OSC_distr_repeat_fix/')
    pathFPTS = pathPar / 'fpts/'
    pathPots = pathPar / 'pots/'
    shutil.copy(sys.argv[0], pathPar)
    N_fac = int(1e10) ; dt_fac = 0.01 ; stride = 5; patch_traj = 4

    # Argument parser setups
    parser = argparse.ArgumentParser(description="Run GLE (Single Osc) MFPT Computation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--params', type=str, required=True, help="Parameter string for the job")
    args = parser.parse_args()

    # Split the parameter string into a tuple of four numbers
    param_str = args.params
    param_tuple = tuple(map(float, param_str.split(',')))

    # Call the function with the parsed parameters
    data_with_paral_i(param_tuple)

