import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from ipywidgets import IntSlider,FloatSlider,interact,Checkbox,GridBox,Layout,interactive_output,FloatText
def time_step(state,params,dt):
    state=state.copy()
    S=state[1]    
    I=state[2]
    R=state[3]
    D=state[4]
    Q=state[5]
    V=state[6]
    cI=state[7]
    N=S+I+R+Q+V
    c=params['c']
    p_t=params['p_t']
    r=params['r']
    delta=params['delta']
    p_q=params['p_q']
    c_q=params['c_q']
    u=params['u']
    # infections
    infections=np.random.binomial(n=S,p=np.min([dt*c*p_t*I/N,1]))
    recoveries=np.random.binomial(n=I,p=np.min([dt*r,1]))
    deaths=np.random.binomial(n=I,p=np.min([dt*delta,1]))
    if I+infections-recoveries-deaths<0:
        recoveries=I+infections-deaths
    recoveries2=np.random.binomial(n=Q,p=np.min([dt*r,1]))
    deaths2=np.random.binomial(n=Q,p=np.min([dt*delta,1]))
    if Q-recoveries2-deaths2<0:
        recoveries2=Q-deaths2
    
    #print(infections,recoveries,deaths)
    state[0]+=dt
    state[1]-=infections
    state[2]=state[2]+infections-recoveries-deaths
    state[3]+=recoveries+recoveries2
    state[4]+=deaths+deaths2
    # quarantines
    quarantines=np.random.binomial(n=int(np.min([dt*c_q,state[2]])),p=p_q)
    state[5]+=quarantines-recoveries2-deaths2
    state[2]-=quarantines
    
    # vaccinations
    vaccinations=np.min([dt*u,state[1]])
    #vaccinations=np.random.binomial(n=np.min([dt*u,state[1]+state[2]]),p=state[1]/(state[1]+state[2]))
    state[6]+=vaccinations
    state[1]-=vaccinations
    state[7]+=infections
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
    l5,=plt.plot(state[0,:],state[5,:]/fac,'m'+ls,label='Q'+lb2)
    l6,=plt.plot(state[0,:],state[6,:]/fac,'c'+ls,label='V'+lb2)
    l7,=plt.plot(state[0,:],state[7,:]/fac,'y'+ls,label='cumulative I'+lb2)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.xlabel('time (days)')
    if normalized:
        plt.ylabel('fraction of population')
    else:
        plt.ylabel('# of people')
    plt.tight_layout()
    return l1,l2,l3,l4,l5,l6,l7

def print_state(state,N0):
    print("Total infected: {:.0f}, percentage of population: {:.3f}%".format(state[7,-1],(state[7,-1])/N0*100))
    print("Total deaths: {:.0f}, percentage of population: {:.3f}%".format(state[4,-1],(state[4,-1])/N0*100))


def simulate_epidemic(init_state,tmax,params):
    state=init_state.reshape(-1,1).copy()
    new_state=0*state
    while state[0,-1]<tmax:
        new_state=time_step(state[:,-1],params,params['dt'])
        state=np.concatenate([state,new_state.reshape(-1,1)],axis=1)
    return state

def simulate_intervention(init_state,params,seed=1):
    np.random.seed(int(seed))
    params_after_intervention=params.copy()
    params['p_q']=0
    params['u']=0
    state=simulate_epidemic(init_state,params['intervention time'],params)
    params_after_intervention['intial state']=state[:,-1]
    params_after_intervention['c']=(1-params['contact reduction factor'])*params['c']
    params_after_intervention['r']=params['r']*(1+params['recovery increase factor'])
    np.random.seed(int(seed))
    
    state_wo=np.concatenate([state.copy(),
                             simulate_epidemic(state[:,-1],
                                               params['t_max'],
                                               params)],
                            axis=1)
    np.random.seed(int(seed))
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
                  recovery_rate_increase_factor=0.5,
                  detection_probability=0.5,
                  max_quarantine=20,
                  vaccination_rate=20,
                  dt=1,
                  t_max=500,normalized=False):
    fig, ax = plt.subplots(figsize=(8,4))
    init_state=np.array([0,N0-I0,I0,0,0,0,0,I0])
    params={}
    params['N0']=N0
    params['dt']=1
    params['t_max']=500
    params['c']=contact_rate
    params['p_t']=transmission_probability
    params['r']=(1-mortality_rate)/infection_duration
    params['delta']=mortality_rate/infection_duration
    params['p_q']=detection_probability
    params['c_q']=max_quarantine
    params['u']=vaccination_rate
    params['intervention time']=intervention_time
    params['contact reduction factor']=contact_rate_reduction_factor
    params['recovery increase factor']=contact_rate_reduction_factor
    params['t_max']=t_max
    state_wo,state_w=simulate_intervention(init_state,params,seed=1)
    l1,l2,l3,l4,l5,l6,l7=plot_state(state_wo,N0,normalized=normalized,ls='',lb2=' without SD')
    l8,l9,l10,l11,l12,l13,l14=plot_state(state_w,N0,normalized=normalized,ls='--',lb2=' with SD')
    l15=plt.axvline(params['intervention time'],c='gray')
    style = {'description_width': 'initial'}
    xlim=FloatSlider(description='t range', step=0.1, min=0,max=t_max,value=t_max,style=style)
    ylim=FloatSlider(description='y range', step=0.1, min=0,max=N0,value=N0,style=style)
    seed=FloatSlider(description='random seed', step=1, min=0,max=100,value=N0,style=style)
    contact_rate=FloatSlider(description='contact rate', step=0.1, min=0,max=100.0,value=contact_rate,style=style)
    transmission_probability=FloatSlider(description='transmission prob',step=0.001, min=0,max=0.1, value=transmission_probability,style=style)
    infection_duration=FloatSlider(description='infection duration', step=0.1, min=0,max=100, value=infection_duration,style=style)
    mortality_rate=FloatSlider(description='mortality rate',step=0.001, min=0,max=0.1, value=mortality_rate,style=style)
    detection_probability=FloatSlider(description='detection prob', step=0.01, min=0,max=1, value=detection_probability,style=style)
    max_quarantine=FloatSlider(description='max quarantine', step=1, min=0,max=100, value=max_quarantine,style=style)
    vaccination_rate=FloatSlider(description='vaccination rate', step=1, min=0,max=t_max, value=vaccination_rate,style=style)
#     intervention_time=FloatSlider(description='t', step=0.1, min=0,max=500, value=intervention_time)
    intervention_time=FloatText(value=intervention_time, description='intervention time',style=style,layout=Layout(width='50%', height='30px'))
    contact_rate_reduction_factor=FloatSlider(description='contact reduction', step=0.001, min=0,max=1, value=contact_rate_reduction_factor,style=style)
    recovery_rate_increase_factor=FloatSlider(description='recovery increase', step=0.001, min=0,max=1, value=recovery_rate_increase_factor,style=style)
    boxS = Checkbox(False, description='Show S?')
    boxI = Checkbox(True, description='Show I?')
    boxR = Checkbox(False, description='Show R?')
    boxD = Checkbox(True, description='Show D?')
    boxQ = Checkbox(True, description='Show Q?')
    boxV = Checkbox(True, description='Show V?')
    boxcI = Checkbox(True, description='Show cumulative I?')
    print_time=FloatText(value=t_max, description='time to print',layout=Layout(width='50%', height='30px'))
    ui=GridBox(children=[xlim,ylim,
                      seed,
                      contact_rate,transmission_probability,
                      infection_duration,mortality_rate,
                      intervention_time,
                      detection_probability,max_quarantine,
                      vaccination_rate,
                      contact_rate_reduction_factor,recovery_rate_increase_factor,
                      boxS,boxI,
                      boxR,boxD,
                      boxQ,boxV,
                      boxcI,print_time
                     ],
        layout=Layout(
            width='100%',
            grid_template_rows='auto auto auto',
            grid_template_columns='30% 30% 30%')
       )
    def update(xlim=xlim,
               ylim=ylim,
               seed=seed,
               contact_rate=contact_rate,
               transmission_probability=transmission_probability,
               infection_duration=infection_duration,
               mortality_rate=mortality_rate,
               detection_probability=detection_probability,
               max_quarantine=max_quarantine,
               vaccination_rate=vaccination_rate,
               intervention_time=intervention_time,
               contact_rate_reduction_factor=contact_rate_reduction_factor,
               recovery_rate_increase_factor=recovery_rate_increase_factor,
               boxS = boxS,
               boxI = boxI,
               boxR = boxR,
               boxD = boxD,
               boxQ = boxQ,
               boxV = boxV,
               boxcI = boxcI,
               print_time=print_time
           ):
    
        params['c']=contact_rate
        params['p_t']=transmission_probability
        params['r']=(1-mortality_rate)/infection_duration
        params['delta']=mortality_rate/infection_duration
        params['p_q']=detection_probability
        params['c_q']=max_quarantine
        params['u']=vaccination_rate
        params['intervention time']=intervention_time
        params['contact reduction factor']=contact_rate_reduction_factor
        params['recovery increase factor']=recovery_rate_increase_factor
        state_wo,state_w=simulate_intervention(init_state,params,seed=seed)
        if normalized:
            fac=N0
        else:
            fac=1
        l1.set_xdata(state_wo[0,:])
        l1.set_ydata(state_wo[1,:]/fac)            
        l2.set_xdata(state_wo[0,:])
        l2.set_ydata(state_wo[2,:]/fac)
        l3.set_xdata(state_wo[0,:])
        l3.set_ydata(state_wo[3,:]/fac)
        l4.set_xdata(state_wo[0,:])
        l4.set_ydata(state_wo[4,:]/fac)
        l5.set_xdata(state_wo[0,:])
        l5.set_ydata(state_wo[5,:]/fac)
        l6.set_xdata(state_wo[0,:])
        l6.set_ydata(state_wo[6,:]/fac)
        l7.set_xdata(state_wo[0,:])
        l7.set_ydata(state_wo[7,:]/fac)

        l8.set_xdata(state_w[0,:])
        l8.set_ydata(state_w[1,:]/fac)
        l9.set_xdata(state_w[0,:])
        l9.set_ydata(state_w[2,:]/fac)
        l10.set_xdata(state_w[0,:])
        l10.set_ydata(state_w[3,:]/fac)
        l11.set_xdata(state_w[0,:])
        l11.set_ydata(state_w[4,:]/fac)
        l12.set_xdata(state_w[0,:])
        l12.set_ydata(state_w[5,:]/fac)
        l13.set_xdata(state_w[0,:])
        l13.set_ydata(state_w[6,:]/fac)
        l14.set_xdata(state_w[0,:])
        l14.set_ydata(state_w[7,:]/fac)
        l15.set_xdata(params['intervention time'])
        
        if boxS:
            l1.set_visible(True)
            l8.set_visible(True)
        else:
            l1.set_visible(False)
            l8.set_visible(False)
            
        if boxI:
            l2.set_visible(True)
            l9.set_visible(True)
        else:
            l2.set_visible(False)
            l9.set_visible(False)
            
        if boxR:
            l3.set_visible(True)
            l10.set_visible(True)
        else:
            l3.set_visible(False)
            l10.set_visible(False)
            
        if boxD:
            l4.set_visible(True)
            l11.set_visible(True)
        else:
            l4.set_visible(False)
            l11.set_visible(False)
            
        if boxQ:
            l5.set_visible(True)
            l12.set_visible(True)
        else:
            l5.set_visible(False)
            l12.set_visible(False)
        if boxV:
            l6.set_visible(True)
            l13.set_visible(True)
        else:
            l6.set_visible(False)
            l13.set_visible(False)
        if boxcI:
            l7.set_visible(True)
            l14.set_visible(True)
        else:
            l7.set_visible(False)
            l14.set_visible(False)
        plt.xlim(0,xlim)
        plt.ylim(0,ylim)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1) 
        print("State at time t={}".format(print_time))
        i=np.argwhere(state_wo[0,:]>=print_time)[0]
        print_state_wo=state_wo[1:,i]
        print_state_w=state_w[1:,i]
        labels=['S','I','R','D','Q','V','cumulative I']
        for k in range(len(labels)):
            print('{} (with intervention): {} {} (without intervention): {}'.format(labels[k],str(print_state_wo[k][0]).ljust(10),labels[k],str(print_state_w[k][0]).ljust(10)))
        print("")
        print("R0: {:.3f}".format(params['c']*params['p_t']/(params['r']+params['delta']), "(values greater than 1 lead to an outbreak)"))
        print("")
        print("Final Epidemic results:")
        print("Without social distancing:")
        print_state(state_wo,params['N0'])
        print("With social distancing:")
        print_state(state_w,params['N0'])

        fig.canvas.draw_idle()
    
    output=interactive_output(update,  {'xlim': xlim,
                                        'ylim': ylim,
                                        'seed':seed,
                                        'contact_rate':contact_rate,
                                        'transmission_probability':transmission_probability,
                                        'infection_duration':infection_duration,
                                        'mortality_rate':mortality_rate,
                                        'detection_probability':detection_probability,
                                        'max_quarantine':max_quarantine,
                                        'vaccination_rate':vaccination_rate,
                                        'intervention_time':intervention_time,
                                        'contact_rate_reduction_factor':contact_rate_reduction_factor,
                                        'recovery_rate_increase_factor':recovery_rate_increase_factor,
                                        'boxS': boxS,
                                        'boxI': boxI,
                                        'boxR': boxR,
                                        'boxD': boxD,
                                        'boxQ': boxQ,
                                        'boxV': boxV,
                                        'boxcI': boxcI,'print_time':print_time}
                                 )  
    return (l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15),params,output,ui