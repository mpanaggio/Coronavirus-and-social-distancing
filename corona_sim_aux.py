import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from ipywidgets import IntSlider,FloatSlider,interact
def time_step(state,params,dt):
    state=state.copy()
    S=state[1]    
    I=state[2]
    R=state[3]
    D=state[4]
    N=S+I+R
    c=params['c']
    tau=params['tau']
    r=params['r']
    delta=params['delta']
    # infections
    infections=np.random.binomial(n=S,p=np.min([dt*c*tau*I/N,1]))
    recoveries=np.random.binomial(n=I,p=np.min([dt*r,1]))
    deaths=np.random.binomial(n=I,p=np.min([dt*delta,1]))
    if I+infections-recoveries-deaths<0:
        recoveries=I+infections-deaths
    #print(infections,recoveries,deaths)
    state[0]+=dt
    state[1]-=infections
    state[2]=state[2]+infections-recoveries-deaths
    state[3]+=recoveries
    state[4]+=deaths
    return state
def plot_state(state,N0,normalized=False,ls='',lb2=''):
    if normalized:
        fac=N0
    else:
        fac=1
    l1,=plt.plot(state[0,:],state[1,:]/fac,'b'+ls,label='S'+lb2)
    l2,=plt.plot(state[0,:],state[2,:]/fac,'r'+ls,label='I'+lb2)
    l3,=plt.plot(state[0,:],state[3,:]/fac,'g'+ls,label='R'+lb2)
    l4,=plt.plot(state[0,:],state[4,:]/fac,'k'+ls,label='D'+lb2)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.xlabel('time (days)')
    plt.ylabel('fraction of population')
    plt.tight_layout()
    return l1,l2,l3,l4

def print_state(state,N0):
    print("Total infected: {:.0f}, percentage of population: {:.3f}%".format(N0-state[1,-1],(N0-state[1,-1])/N0*100))
    print("Total deaths: {:.0f}, percentage of population: {:.3f}%".format(state[4,-1],(state[4,-1])/N0*100))


def simulate_epidemic(init_state,tmax,params):
    state=init_state.reshape(-1,1).copy()
    new_state=0*state
    while state[0,-1]<tmax:
        new_state=time_step(state[:,-1],params,params['dt'])
        state=np.concatenate([state,new_state.reshape(-1,1)],axis=1)
    return state

def simulate_intervention(init_state,params):
    state=simulate_epidemic(init_state,params['intervention time'],params)
    params_after_intervention=params.copy()
    params_after_intervention['intial state']=state[:,-1]
    params_after_intervention['c']=(1-params['contact reduction factor'])*params['c']
    state_wo=np.concatenate([state.copy(),
                             simulate_epidemic(state[:,-1],
                                               params['t_max'],
                                               params)],
                            axis=1)
    state_w=np.concatenate([state.copy(),
                            simulate_epidemic(state[:,-1],
                                              params['t_max'],
                                              params_after_intervention)],
                           axis=1)
    return state_wo,state_w

def make_interactive_plot(N0=300e6,
                  I0=100,
                  contact_rate=20,
                  transmission_probability=0.01,
                  mortality_rate=0.03,
                  infection_duration=10,
                  intervention_time=100,
                  contact_rate_reduction_factor=0.4,
                  dt=1,
                  t_max=500):
    fig, ax = plt.subplots(figsize=(8,3))
    init_state=np.array([0,N0-I0,I0,0,0])
    params={}
    params['N0']=N0
    params['dt']=1
    params['t_max']=500
    params['c']=contact_rate
    params['tau']=transmission_probability
    params['r']=(1-mortality_rate)/infection_duration
    params['delta']=mortality_rate/infection_duration
    params['intervention time']=intervention_time
    params['contact reduction factor']=contact_rate_reduction_factor
    state_wo,state_w=simulate_intervention(init_state,params)
    l1,l2,l3,l4=plot_state(state_wo,N0,normalized=True,ls='',lb2=' without SD')
    l5,l6,l7,l8=plot_state(state_w,N0,normalized=True,ls='--',lb2=' with SD')
    l9=plt.axvline(params['intervention time'],c='gray')
    def update(contact_rate=FloatSlider(description='c', step=0.1, min=0,max=100.0,value=contact_rate),
           transmission_probability=FloatSlider(description='p', 
                           step=0.001, min=0,max=0.1, value=transmission_probability),
           infection_duration=FloatSlider(description='d', 
                         step=0.1, min=0,max=100, value=infection_duration),
           mortality_rate=FloatSlider(description='m', 
                         step=0.001, min=0,max=0.1, value=mortality_rate),
           intervention_time=FloatSlider(description='t', 
                                         step=0.1, min=0,max=500, value=intervention_time),
           contact_rate_reduction_factor=FloatSlider(description='k', 
                                                     step=0.001, min=0,max=1, value=contact_rate_reduction_factor)
           ):
    
        params['c']=contact_rate
        params['tau']=transmission_probability
        params['r']=(1-mortality_rate)/infection_duration
        params['delta']=mortality_rate/infection_duration
        params['intervention time']=intervention_time
        params['contact reduction factor']=contact_rate_reduction_factor
        state_wo,state_w=simulate_intervention(init_state,params)
        l1.set_xdata(state_wo[0,:])
        l1.set_ydata(state_wo[1,:]/N0)
        l2.set_xdata(state_wo[0,:])
        l2.set_ydata(state_wo[2,:]/N0)
        l3.set_xdata(state_wo[0,:])
        l3.set_ydata(state_wo[3,:]/N0)
        l4.set_xdata(state_wo[0,:])
        l4.set_ydata(state_wo[4,:]/N0)

        l5.set_xdata(state_w[0,:])
        l5.set_ydata(state_w[1,:]/N0)
        l6.set_xdata(state_w[0,:])
        l6.set_ydata(state_w[2,:]/N0)
        l7.set_xdata(state_w[0,:])
        l7.set_ydata(state_w[3,:]/N0)
        l8.set_xdata(state_w[0,:])
        l8.set_ydata(state_w[4,:]/N0)
        l9.set_xdata(params['intervention time'])
        print("R0: {:.3f}".format(params['c']*params['tau']/(params['r']+params['delta']), "(values greater than 1 lead to an outbreak)"))
        print("Without social distancing:")
        print_state(state_wo,params['N0'])
        print("With social distancing:")
        print_state(state_w,params['N0'])

        fig.canvas.draw_idle()
    return (l1,l2,l3,l4,l5,l6,l7,l8,l9),params,update