import numpy as np
import scipy.special as scip
import scipy.optimize as scio
import csv


def _T_tyxf7(dfconstr, nux, n, t, tw, K, p, N):
    newnux = np.zeros(3)
    if dfconstr == "no":
        dfoldg = nux.copy()

        #scipy digamma is slow? https://gist.github.com/timvieira/656d9c74ac5f82f596921aa20ecb6cc8
        for i in range(0, K):
            constn = 1 + (1/n[i]) * np.sum(t[:, i] * (np.log(tw[:, i]) - tw[:, i])) + scip.digamma((dfoldg[i] + p)/2) - np.log((dfoldg[i] + p)/2)
            temp = scip.digamma((dfoldg[i] + p)/2)
            print(f'digamma 7 value:{temp}')
            print(constn)
            f = lambda v : np.log(v/2) - scip.digamma(v/2) + constn
            #Verify this outputs the same as R: may need to set rtol to 0
            newnux[i] = scio.brentq(f, 0.0001, 1000, xtol=0.00001)

            if newnux[i] > 200:
                newnux[i] = 200.

            if newnux[i] < 2:
                newnux[i] = 2.

    else:
        dfoldg = nux[0]
        constn = 1 + (1/N) * np.sum(t *(np.log(tw) - tw)) + scip.digamma( (dfoldg + p) / 2) - np.log( (dfoldg + p) / 2)

        print(constn)
        f = lambda v : np.log(v/2) - scip.digamma(v/2) + constn
            #Verify this outputs the same as R: may need to set rtol to 0
        dfsamenewg = scio.brentq(f, 0.0001, 1000, xtol=0.01)

        if dfsamenewg > 200:
            dfsamenewg = 200.
        
        if dfsamenewg < 2:
            dfsamenewg = 2.

        newnux = np.repeat(dfsamenewg, K)

    return newnux

def _T_tyxf8(dfconstr, nux, n, t, tw, K, p, N):
    newnux = np.zeros(3)

    if(dfconstr == "no"):
        dfoldg = nux.copy()
        
        for i in range(0, K):
            constn = 1 + (1 / n[i]) * np.sum(t[:, i] * (np.log(tw[:, i]) - tw[:, i])) + scip.digamma((dfoldg[i] + p)/2) - np.log( (dfoldg[i] + p)/2)
            temp = scip.digamma((dfoldg[i] + p)/2)
            print('dfoldg')
            print(dfoldg)
            #print((dfoldg[i]+p)/2)
            #print(f'digamma 8 value:{temp}')
            constn = -constn
            print(constn)
            newnux[i] = (-np.exp(constn) + 2 * (np.exp(constn)) * (np.exp(scip.digamma(dfoldg[i] / 2)) - ( (dfoldg[i]/2) - (1/2)))) / (1 - np.exp(constn))

            if newnux[i] > 200:
                newnux[i] = 200.

            if newnux[i] < 2:
                newnux[i] = 2.

    else:
        dfoldg = nux[0]
        constn = 1 + (1 / N) * np.sum(t * (np.log(tw) - tw)) + scip.digamma((dfoldg + p)/2) - np.log( (dfoldg + p)/2)
        constn = -constn

        print(constn)
        dfsamenewg = (-np.exp(constn) + 2 * (np.exp(constn)) * (np.exp(scip.digamma(dfoldg / 2)) - ( (dfoldg/2) - (1/2)))) / (1 - np.exp(constn))

        if dfsamenewg > 200:
            dfsamenewg = 200.

        if dfsamenewg < 2:
            dfsamenewg = 2.

        newnux = np.repeat(dfsamenewg, K)

    return newnux

data = []
mats = {'t': [], 'tw': []}
with open('res5.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

#print(data)
#print(data[1][0:3])
for i in data[1:]:
    mats['t'].append(i[0:3])
    mats['tw'].append(i[3:6])

mats['L'] = np.array(data[1][-1]).astype(float)

mats['t'] = np.array(mats['t']).astype(float)
mats['tw'] = np.array(mats['tw']).astype(float)

par = {'K': 3, 'nux':np.array([201.,201.,201.])}


test1_7 = _T_tyxf7('no', par['nux'], np.sum(mats['t'], axis=0), mats['t'], mats['tw'], 3, 21., 21.)
test2_7 = _T_tyxf7('yes', par['nux'], np.sum(mats['t'], axis=0), mats['t'], mats['tw'], 3, 21., 21.)

test1_8 = _T_tyxf8('no', par['nux'], np.sum(mats['t'], axis=0), mats['t'], mats['tw'], 3, 21., 21.)
test2_8 = _T_tyxf8('yes', par['nux'], np.sum(mats['t'], axis=0), mats['t'], mats['tw'], 3, 21., 21.)


print(f'tyfx7,dfconstr=no:{test1_7}; tyfx7,dfconstr=yes:{test2_7};tyfx8,dfconstr=no:{test1_8};tyfx8,dfconstr=yes:{test2_8}')