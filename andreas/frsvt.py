def frsvt(A, Q_tilda, tau, k, p, q): # q isn't used???
    m, n = A.shape
    l = k + p #>0
    if rank(A) < l:
        is_range_prop = False#
    # Find optimal (for NNM ~> lowest rank) X*
    # X* = SVT(A) ~=Q*SVT(B) to avoid SVD on A (mn)

    # Step 1: estimate orthonormal Q (mn) with k-rank << n
    if not is_range_prop:
        Sigma = np.random.multivariate_normal() #nl
        Y = A@Sigma #(ml)
        Q = sp.linalg.qr(Y, pivoting=True)# interface to lapack dgeqp3, also exists a low-level wrapper
    else:
        Sigma = np.random.multivariate_normal() # np
        Y = A@Sigma #Y(mp)
        ## Initialize Q via range propagation
            ## orthonormal biasis evolve slow
        Q_Y = gram_schmidt(Y) #todo
        Q = concat_by_col(Q_tilda, Y)
    ## power interation
    eta = 2
    while eta > 0: #what is eta?
        Q = sp.linalg.qr(A@A.T@Q)
        eta -= 1
    #Step 2: SVT on B (~kn)
        ## SVD (by eigen decomposition if pos semi-def) + shrinkage on SV
    # C = WP = WVDV.T
    H, C = sp.linalg.qr(A.T@Q)
    # Form a pos semi-def, W(cpl, mn), P(cpl, nn, herm pos semi-def)
    W, P = sp.linalg.polar(C)
    V, D = sp.linalg.eig(P)
    Q_tilda = Q@V
    SS_D = np.sign(D) * np.max(np.abs(D)-tau, 0) #soft shrinkage
    SVT_A = Q_tilda@SS_D@(H@W@V).T
    return SVT_A, Q_tilda

def gram_schmidt?
