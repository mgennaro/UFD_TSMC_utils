import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def drawCMD(pbs,cat,starnum,savepath):

    #General plot settings

    star = cat.iloc[starnum]

    fig = plt.figure(figsize=(7.5,5)) 
    ax = fig.add_subplot(111)
    ax.set_xlabel(pbs[0].name+'-'+pbs[1].name)
    ax.set_ylabel(pbs[1].name)
    ax.hexbin(cat[pbs[0].name]-cat[pbs[1].name],cat[pbs[1].name],cmap='Blues',bins='log',gridsize=200)
    ax.scatter(star[pbs[0].name]-star[pbs[1].name],star[pbs[1].name],marker="*",s=85,c='yellow')
    ax.set_ylim(ax.get_ylim()[::-1])
    print("saving"+savepath+'CMD_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf')
    fig.savefig(savepath+'CMD_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf',bbox_inches='tight')
    plt.close(fig)
    

def drawmagpairs(pbs,cat,starnum,magsin,magsou,logpst,loglik,logpri,cmpval,savepath):

    star = cat.iloc[starnum]   
    err = 5*np.amax(np.asarray([star['err_'+pb.name] for pb in pbs ]))
    smg = np.asarray([star[pb.name] for pb in pbs ])

    f =  plt.figure(figsize=(15,7))

    ax =f.add_subplot(111)
    ax1=f.add_subplot(241)
    ax2=f.add_subplot(242)
    ax3=f.add_subplot(243)
    ax4=f.add_subplot(244)

    ax5=f.add_subplot(245)
    ax6=f.add_subplot(246)
    ax7=f.add_subplot(247)
    ax8=f.add_subplot(248)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    
    ax1.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=logpst,cmap='RdYlBu_r',s=20,lw=0)
    ax1.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax1.set_ylabel('Input',fontsize=15)
    ax1.set_ylim(smg[1]+err,smg[1]-err)
    ax1.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax1.locator_params(tight=True, nbins=5)
    ax1.set_title('Log posterior',fontsize=15)
        
    ax2.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=logpri,cmap='RdYlBu_r',s=20,lw=0)
    ax2.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax2.set_ylim(smg[1]+err,smg[1]-err)
    ax2.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax2.locator_params(tight=True, nbins=5)
    ax2.set_title('Log prior',fontsize=15)
   
    ax3.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=loglik,cmap='RdYlBu_r',s=20,lw=0)
    ax3.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax3.set_ylim(smg[1]+err,smg[1]-err)
    ax3.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax3.locator_params(tight=True, nbins=5)
    ax3.set_title('Log likelihood',fontsize=15)
    
    ax4.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=cmpval,cmap='RdYlBu_r',s=20,lw=0)
    ax4.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax4.set_ylim(smg[1]+err,smg[1]-err)
    ax4.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax4.locator_params(tight=True, nbins=5)
    ax4.set_title('Completeness',fontsize=15)
    
    ax5.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=logpst,cmap='RdYlBu_r',s=20,lw=0)
    ax5.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax5.set_ylabel('Output',fontsize=15)
    ax5.set_ylim(smg[1]+err,smg[1]-err)
    ax5.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax5.locator_params(tight=False, nbins=5)
   
    ax6.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=logpri,cmap='RdYlBu_r',s=20,lw=0)
    ax6.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax6.set_ylim(smg[1]+err,smg[1]-err)
    ax6.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax6.locator_params(tight=True, nbins=5)
    
    ax7.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=loglik,cmap='RdYlBu_r',s=20,lw=0)
    ax7.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax7.set_ylim(smg[1]+err,smg[1]-err)
    ax7.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax7.locator_params(tight=True, nbins=5)
    
    ax8.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=cmpval,cmap='RdYlBu_r',s=20,lw=0)
    ax8.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax8.set_ylim(smg[1]+err,smg[1]-err)
    ax8.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax8.locator_params(tight=True, nbins=5)

    plt.subplots_adjust(wspace=0.025)
    plt.subplots_adjust(hspace=0.05)
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in f.axes], visible=False)

    plt.setp(ax1.get_yticklabels(),visible=True)
    plt.setp(ax5.get_yticklabels(),visible=True)
    plt.setp(ax5.get_xticklabels(),visible=True)
    plt.setp(ax6.get_xticklabels(),visible=True)
    plt.setp(ax7.get_xticklabels(),visible=True)
    plt.setp(ax8.get_xticklabels(),visible=True)


    ax.xaxis.set_label_coords(0.5, -0.075)
    ax.yaxis.set_label_coords(-0.075, 0.5)
    ax.set_xlabel(pbs[0].name+'-'+pbs[1].name,fontsize=17)
    ax.set_ylabel(pbs[1].name,fontsize=17)
    print("saving"+savepath+'Mdraws_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf')
    plt.savefig(savepath+'Mdraws_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf',bbox_inches='tight')
    plt.close(f)




