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
    meancol = np.mean(cat[pbs[0].name]-cat[pbs[1].name])
    stdvcol = np.std(cat[pbs[0].name]-cat[pbs[1].name])
    ax.set_xlim(meancol-5*stdvcol,meancol+6*stdvcol)

    print("saving"+savepath+'CMD_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf')
    fig.savefig(savepath+'CMD_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf',bbox_inches='tight')
    plt.close(fig)
    

def drawmagpairs(pbs,cat,starnum,magsin,magsou,logpst,loglik,logpri,cmpval,savepath):

    star = cat.iloc[starnum]   
    err = 5*(np.amax(np.asarray([star['err_'+pb.name] for pb in pbs ]))+0.02)
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
    #ax1.set_ylim(smg[1]+err,smg[1]-err)
    #ax1.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax1.locator_params(tight=True, nbins=5)
    ax1.set_title('Log posterior',fontsize=15)
        
    ax2.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=logpri,cmap='RdYlBu_r',s=20,lw=0)
    ax2.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    #ax2.set_ylim(smg[1]+err,smg[1]-err)
    #ax2.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax2.locator_params(tight=True, nbins=5)
    ax2.set_title('Log prior',fontsize=15)
   
    ax3.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=loglik,cmap='RdYlBu_r',s=20,lw=0)
    ax3.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    #ax3.set_ylim(smg[1]+err,smg[1]-err)
    #ax3.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax3.locator_params(tight=True, nbins=5)
    ax3.set_title('Log likelihood',fontsize=15)
    
    ax4.scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=cmpval,cmap='RdYlBu_r',s=20,lw=0)
    ax4.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    #ax4.set_ylim(smg[1]+err,smg[1]-err)
    #ax4.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax4.locator_params(tight=True, nbins=5)
    ax4.set_title('Completeness',fontsize=15)
    
    ax5.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=logpst,cmap='RdYlBu_r',s=20,lw=0)
    ax5.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    ax5.set_ylabel('Output',fontsize=15)
    #ax5.set_ylim(smg[1]+err,smg[1]-err)
    #ax5.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax5.locator_params(tight=False, nbins=5)
   
    ax6.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=logpri,cmap='RdYlBu_r',s=20,lw=0)
    ax6.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    #ax6.set_ylim(smg[1]+err,smg[1]-err)
    #ax6.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax6.locator_params(tight=True, nbins=5)
    
    ax7.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=loglik,cmap='RdYlBu_r',s=20,lw=0)
    ax7.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    #ax7.set_ylim(smg[1]+err,smg[1]-err)
    #ax7.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax7.locator_params(tight=True, nbins=5)
    
    ax8.scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=cmpval,cmap='RdYlBu_r',s=20,lw=0)
    ax8.scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
    #ax8.set_ylim(smg[1]+err,smg[1]-err)
    #ax8.set_xlim(smg[0]-smg[1]-err,smg[0]-smg[1]+err)
    ax8.locator_params(tight=True, nbins=5)


    inlx0,inlx1 = ax1.get_xlim()
    oulx0,oulx1 = ax5.get_xlim()
    newlimx0 = np.amin([inlx0,inlx1,oulx0,oulx1])
    newlimx1 = np.amax([inlx0,inlx1,oulx0,oulx1])

    inly0,inly1 = ax1.get_ylim()
    ouly0,ouly1 = ax5.get_ylim()
    newlimy0 = np.amin([inly0,inly1,ouly0,ouly1])
    newlimy1 = np.amax([inly0,inly1,ouly0,ouly1])

    for a in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]:
        a.set_xlim(newlimx0,newlimx1)
        a.set_ylim(newlimy1,newlimy0)

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

    return [[newlimx0,newlimx1],[newlimy0,newlimy1]]

def drawmagpairs2(pbs,cat,starnum,magsin,magsou,df,cols,savepath,axlim):
    
    star = cat.iloc[starnum]   
    err = 5*(np.amax(np.asarray([star['err_'+pb.name] for pb in pbs ]))+0.02)
    smg = np.asarray([star[pb.name] for pb in pbs ])
    ncols = len(cols)

    f =  plt.figure(figsize=(1.+3.5*ncols,7))

    ax =f.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    axU = []
    axD = []

    for j,col in enumerate(cols):   
        axU.append(f.add_subplot(2,ncols,j+1)) 
        axD.append(f.add_subplot(2,ncols,j+1+ncols)) 
        axU[j].scatter(magsin[:,0]-magsin[:,1],magsin[:,1],c=df[cols[j]],cmap='RdYlBu_r',s=20,lw=0)
        axD[j].scatter(magsou[:,0]-magsou[:,1],magsou[:,1],c=df[cols[j]],cmap='RdYlBu_r',s=20,lw=0)
        axU[j].scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
        axD[j].scatter(smg[0]-smg[1],smg[1],marker='*',c='yellow',s=85)
        axU[j].locator_params(tight=True, nbins=5)
        axD[j].locator_params(tight=True, nbins=5)
        axU[j].set_title(cols[j],fontsize=15)

    axU[0].set_ylabel('Input',fontsize=15)
    axD[0].set_ylabel('Output',fontsize=15)

    #  inlx0,inlx1 = axU[0].get_xlim()
    #oulx0,oulx1 = axD[0].get_xlim()
    #newlimx0 = np.amin([inlx0,inlx1,oulx0,oulx1])
    #newlimx1 = np.amax([inlx0,inlx1,oulx0,oulx1])
    #inly0,inly1 = axU[0].get_ylim()
    #ouly0,ouly1 = axD[0].get_ylim()
    #newlimy0 = np.amin([inly0,inly1,ouly0,ouly1])
    #newlimy1 = np.amax([inly0,inly1,ouly0,ouly1])

    newlimx0 = axlim[0][0]
    newlimx1 = axlim[0][1]
    newlimy0 = axlim[1][0]
    newlimy1 = axlim[1][1]

    for a in axU:
        a.set_xlim(newlimx0,newlimx1)
        a.set_ylim(newlimy1,newlimy0)

    for a in axD:
        a.set_xlim(newlimx0,newlimx1)
        a.set_ylim(newlimy1,newlimy0)
  
    plt.subplots_adjust(wspace=0.025)
    plt.subplots_adjust(hspace=0.05)
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in f.axes], visible=False)

    plt.setp(axU[0].get_yticklabels(),visible=True)
    plt.setp(axD[0].get_yticklabels(),visible=True)
    for a in axD:
      plt.setp(a.get_xticklabels(),visible=True)

    ax.xaxis.set_label_coords(0.5, -0.075)
    ax.yaxis.set_label_coords(-0.075, 0.5)
    ax.set_xlabel(pbs[0].name+'-'+pbs[1].name,fontsize=17)
    ax.set_ylabel(pbs[1].name,fontsize=17)
    print("saving"+savepath+'Mdraws2_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf')
    plt.savefig(savepath+'Mdraws2_'+pbs[0].name+'_'+pbs[1].name+'star_'+("%.0f" % starnum)+'.pdf',bbox_inches='tight')
    plt.close(f)


