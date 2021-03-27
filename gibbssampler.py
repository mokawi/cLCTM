import numba as nb
from numba.typed import List
from numba import njit, jit
import numpy as np
from math import log

import logging
logging.basicConfig(filename="numbags.log", level=logging.DEBUG)

@njit
def calc_mu_sigma(
        noise,
        n_c_c,
        sigma_prior,
        mu_prior,
        sum_mu_c_c
    ):

    var_inverse = noise + n_c_c + 1/sigma_prior
    sigma = noise + 1/var_inverse

    c1 = n_c_c + noise/sigma_prior
    c2 = 1+n_c_c * (sigma_prior/noise)
    mu = sum_mu_c_c/c1 + mu_prior/c2

    return mu, sigma


@nb.jit(nopython=True)
def multinomial(weights):
    """Adapted from https://blog.bruce-hill.com/a-faster-weighted-random-choice
    by Bruce Hill, original "alias" method by Walker (1974)
    """
    N = len(weights)
    avg = np.sum(weights)/N
    aliases = List([(1., 0.)]*N)
    
    smalls = List()
    bigs = List()
    for i, w in enumerate(weights):
        if w < avg:
            smalls.append((i, w/avg))
        else:
            bigs.append((i, w/avg))
    #print(smalls, bigs)

    ibg = 0
    ism = 0
    big = bigs[0]
    small = smalls[0]
    
    while True:
        aliases[small[0]] = (small[1], big[0])
        big = (big[0], big[1] - (1-small[1]))
        if big[1] < 1:
            small = big
            ibg += 1
            if ibg >= len(bigs): break
            big = bigs[ibg]
        else:
            ism += 1
            if ism >= len(smalls): break
            small = smalls[ism]

    r = np.random.random()*N
    #print(r)
    i = int(r)
    odds, alias = aliases[i]
    #print((i,r-i),odds, alias)
    return int(alias) if (r-i) > odds else int(i)

#@njit("""
#(u4[::1],u4[::1],u4[::1],
#f8[:,:],
#u4[::1], u4[::1],
#u4[:,:], u4[:,:],
#f8[:,:], f8[::1], f8[::1],
#f8[::1], f8, f8,
#f8[::1], f8,
#u4[::1], u4[::1], u4, b1, u4)
#""")
@jit(nopython=True)
def gibbslctm(
        doc_ids,
        topics,
        concepts,
        wordvectors,
        n_z, n_c,
        n_dz, n_zc,
        sum_mu_c, #mu_c_dot_mu_c,
        mu_c, sigma_c,
        mu_prior, sigma_prior, noise,
        alpha_vec, beta,
        token_neighbors,
        consec_sampled_num, max_consec=100,
        faster=True,
        n_iter=1
    ):
   
    n_dims = wordvectors.shape[1]

    for iteration in range(n_iter):
        print("# Iter", iteration)
        num_z_changed = 0
        num_c_changed = 0
        num_omit = 0

        n_concepts = len(n_c)

        for w in range(len(doc_ids)):
            d = doc_ids[w]
            z = topics[w]
            c = concepts[w]
            wv = wordvectors[w]

            # Draw new topic
            ################

            n_dz[d, z] -= 1
            n_zc[z, c] -= 1
            n_z[z] -= 1

            # Sample topic

            c1 = (n_zc[:,c] + beta)/(n_z + beta * n_concepts)
            c2 = n_dz[d] + alpha_vec
            p = c1/c2
            p = p/p.sum()

            z_new = multinomial(p)

            if z_new != z:
                num_z_changed +=1
            z = z_new

            # update counts
            topics[w] = z
            n_dz[d, z] += 1
            n_zc[z, c] += 1
            n_z[z] += 1

            if faster:
                if consec_sampled_num[w] > max_consec:
                    num_omit += 1
                    continue

            # Draw new concept
            ##################

            sum_mu_c[c] -= wv
            n_c[c] -= 1
            n_zc[z,c] -= 1
 
            mu_c[c], sigma_c[c] = calc_mu_sigma(
                noise, n_c[c],
                sigma_prior, mu_prior,
                sum_mu_c[c]
            )
            #mu_c_dot_mu_c = np.inner(mu_c[c], mu_c[c])
            mu_c_dot_mu_c = np.sum(mu_c[c]*mu_c[c])

            # Sample new concept
            p = np.zeros(len(token_neighbors[d]))
            for n in token_neighbors[d]:
                t1 = -0.5 + n_dims + np.log(sigma_c[n])
                t2 = -(0.5/sigma_c[n]) * (mu_c_dot_mu_c - 2 /mu_c[n] @ wv)
                p[n] = np.log(n_zc[z, n] + beta) + t1 + t2

            maxp = p.max()    
            #p = p/p.sum() # Cuz those numbers tend to be super high # Removed to align w/ C implementation
            p = np.exp(p - maxp) # Now softmax it. Temperature is highest coefficient in the C implementation
            p = p/p.sum()

            c_new = multinomial(p)

            if c != c_new:
                num_c_changed += 1
                if faster:
                    consec_sampled_num[w] = 0
                else:
                    consec_sampled_num[w] += 1
            c = c_new
            concepts[w] = c

            # Update counts

            sum_mu_c[c] += wv
            n_c[c] += 1
            n_zc[z,c] += 1


            mu_c[c], sigma_c[c] = calc_mu_sigma(
                noise, n_c[c],
                sigma_prior, mu_prior,
                sum_mu_c[c]
            )

#    return (
#        topics, concepts, mu_c, sigma_c,
#        sum_mu_c, n_c, n_z, n_dz, n_zc,
#        consec_sampled_num,
#        n_c_changed, n_z_changed, n_omit
#    )
