import numba
from numba import njit, jit, objmode
import numpy as np
from math import log

@njit
def calc_mu_sigma(
        noise,
        n_c_c,
        sigma_prior,
        mu_prior,
        sum_mu_c_c
    ):

    var_inverse = noise + n_c + 1/sigma_prior
    sigma = noise + 1/var_inverse

    c1 = n_c_c + noise/sigma_prior
    c2 = 1+n_c_c * (sigma_prior/noise)
    mu = sum_mu_c_c/c1 + mu_prior/c2

    return mu, sigma

@jit("f8[:](f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
def softmax(z):
    num = np.exp(z)
    s = num / esum(z)
    return s

@njit
def gibbslctm(
        doc_ids,
        topics,
        concepts,
        wordvectors,
        n_z, n_c,
        n_dz, n_zc,
        sum_mu_c, #mu_c_dot_mu_c,
        mu_c, sigma_c,
        mu_prior, sigma_prior
        alpha_vec, beta,
        token_neighbors,
        consec_sampled_num, max_consec=100,
        faster=True,
        n_iter=1
    ):
   
    n_dims = wordvectors.shape[1]

    for iteration in range(n_iter):
        num_z_changed = 0
        num_c_changed = 0
        num_omit = 0

        n_concepts = len(n_c)

        for w in range(doc_ids):
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

            z_new = np.random.multinomial(1, p=p)
            if z_new != z:
                num_z_changed +=1
            z = z_new

            # update counts
            topics[w] = z
            n_dz[d, z] += 1
            n_zc[z, c] += 1
            n_z[z] += 1

            if faster:
                if consec_sampled_num > max_consec:
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
            mu_c_dot_mu_c = np.inner(mu_c[c], mu_c[c])

            # Sample new concept
            neighbors = token_neighbors[d]
            t1 = -0.5 + n_dims + np.log(sigma_c[neighbors])
            t2 = -(0.5/sigma_c[neighbors]) * (mu_c_dot_mu_c - 2 /mu_c[neighbors] @ wv)
            p = softmax(np.log(n_zc[z, neighbors] + beta) + t1 + t2)

            c_new = np.random.multinomial(1,p).argmax()

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
