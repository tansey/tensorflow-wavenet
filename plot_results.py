import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

sizes = [256, 512, 1024, 2048]
softmax = [ 2.5780, 3.2630, 3.9440, 4.7100 ]
unsmoothed = [ 2.4750, 3.1990, 4.0300, 4.6710 ]
smoothed = [ 2.4820, 3.2970, 3.8730, 4.6240]

with sns.axes_style('white'):
    plt.rc('font', weight='bold')
    plt.rc('grid', lw=2)
    plt.rc('lines', lw=3)
    
    plt.figure(1)
    plt.plot(sizes, softmax, c='gray', lw=4, label='Multinomial')
    plt.plot(sizes, unsmoothed, c='orange', lw=4, label='Unsmoothed DP')
    plt.plot(sizes, softmax, c='blue', lw=4, label='SDP')
    plt.ylabel('TV Error', weight='bold', fontsize=24)
    plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig('plots/mnist-scores-{0}.pdf'.format(dist.replace('_','-')), bbox_inches='tight')
    plt.clf()
    plt.close()
