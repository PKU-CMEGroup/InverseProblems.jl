import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import sys

from constants import __logZero, __wtSumThresh
import adaptiveIntegrators as ai
import P2quantile as P2

# --------------------------------------------------
# NUTS sequence of checks (precomputed)
# -------------------------------------------------


# sequence of stop-checks during NUTS iteration
checks = 0
k = 0


def subTreePlan(nleaf):
    global checks
    global k
    checks = np.zeros((nleaf - 1, 2), dtype=int)
    k = 0

    def Uturn(a, b):
        global k
        global checks
        checks[k, 0] = a
        checks[k, 1] = b
        k += 1

    def subUturn(a, b):
        if (a != b):
            m = math.floor((a + b) / 2)
            subUturn(a, m)
            subUturn(m + 1, b)
            Uturn(a, b)

    subUturn(1, nleaf)
    return (checks)


# utilities for storing states (avoiding storing the complete orbit)

class stateStore:
    def __init__(self, d, n):
        self.__stateStack = np.zeros((d, n))
        self.__stateStackId = np.zeros(n, dtype=int)
        self.__stateStackUsed = np.full(n, False, dtype=bool)

    def statePush(self, id_, state):
        full = True
        for i in range(len(self.__stateStackId)):
            if (not self.__stateStackUsed[i]):
                self.__stateStack[:, i] = state
                self.__stateStackId[i] = id_
                self.__stateStackUsed[i] = True
                full = False
                break
        if (full):
            sys.exit("stack full")

    def stateRead(self, id_):
        inds = np.where(self.__stateStackId == id_)[0]
        if (len(inds > 0)):
            for i in range(len(inds)):
                if (self.__stateStackUsed[inds[i]]):
                    return (self.__stateStack[:, inds[i]])
        sys.exit("element not found for read in stack")

    def stateDeleteRange(self, from_, to_):
        self.__stateStackUsed[np.logical_and(self.__stateStackId >= from_, self.__stateStackId <= to_)] = False

    def stateReset(self):
        self.__stateStackUsed = np.full(len(self.__stateStackUsed), False, dtype=bool)

    def dump(self):
        print("values")
        print(self.__stateStack)
        print("ids")
        print(self.__stateStackId)
        print("used")
        print(self.__stateStackUsed)


# --------------------------------------
# NUT stop condition
# --------------------------------------

def stopCondition(qm, vm, qp, vp):
    tmp = qp - qm
    return (sum(vp * tmp) < 0.0 or sum(vm * tmp) < 0.0)


# def stopCondition(qm,vm,qp,vp,thresh=-0.8):
#    tmp = qp-qm
#    nn = np.sqrt(sum(tmp*tmp))
#    return(sum(vp*tmp)/(nn*np.sqrt(sum(vp*vp))) <thresh or sum(vm*tmp)/(nn*np.sqrt(sum(vm*vm)))<thresh)


# -----------------------------------------
# WALNUTS sampler
#
# -----------------------------------------

def WALNUTS(lpFun,
            q0,  # intial position configuration
            generated=lambda q: q,  # apply this function to q before storing samples
            integrator=ai.fixedLeapFrog,
            H0=0.2,  # initial big step size / fixed big step size if adaptH=False
            stepSizeRandScale=0.2,  # step size randomization scale
            delta0=0.05,  # initial integrator tolerance
            # multinomial=False, # if not, then biased progressive
            numIter=2000,
            warmupIter=1000,
            M=10,  # number of NUTS iterations
            igrAux=ai.integratorAuxPar(),
            adaptH=True,
            adaptHtarget=0.8,  # desired fraction of steps where crudest step size is accepted
            adaptDelta=True,
            adaptDeltaTarget=0.6,
            adaptDeltaQuantile=0.9,
            recordOrbitStats=False
            ):
    # -----------------------------------
    # setup
    # -----------------------------------

    # big step size
    H = H0

    if (adaptH):
        if (adaptHtarget < 0.0 or adaptHtarget > 1.0): sys.exit("bad adaptHtarget")
        igrConstQ = P2.P2quantile(1.0 - adaptHtarget)

    # integrator tolerance
    delta = delta0
    if (adaptDelta):
        if (adaptDeltaTarget < 0.0): sys.exit("bad adaptDeltaTarget")
        energyErrorInfFacs = np.zeros(warmupIter)

    # make tables for all of the sub-uturn checks that needs to be done during one iteration
    plans = []
    for i in range(0, M):
        plans.append(subTreePlan(2 ** i))

    # allocated memory for samples
    d = q0.size
    g0 = generated(q0)
    dg = g0.size
    samples = np.zeros((dg, numIter + 1))
    samples[:, 0] = g0

    # allocate memomry for intermediate states
    states = stateStore(3 * d, 2 * (M + 1) + 1)

    # memory for quantities stored for all states in orbit
    Hs = np.zeros(2 ** M)
    Ifs = np.zeros(2 ** M, dtype=int)
    Ibs = np.zeros(2 ** M, dtype=int)
    cs = np.zeros(2 ** M, dtype=int)
    lwts = np.zeros(2 ** M)

    # assorted stored states
    qc = q0  # current state

    # diagnositics info
    diagnostics = np.zeros((numIter, 24))

    if (recordOrbitStats):
        orbitMin = np.zeros((dg, numIter))
        orbitMax = np.zeros((dg, numIter))

    # --------------------------------------
    # main MCMC iteration loop
    # --------------------------------------
    for iterN in range(1, numIter + 1):
        if (iterN % 1000 == 0):
            print("iteration # ", iterN, " H = ", H, " delta = ", delta)

        # -------------------------------------
        # per iteration diagnostics info
        # -------------------------------------
        nevalF = 0
        nevalB = 0
        Lold_ = 0
        L_ = 0
        orbitLen_ = 0.0
        orbitLenSam_ = 0.0
        NdoublSampled_ = 0
        NdoublComputed_ = 0
        indexStat_ = 0.0
        indexStatOld_ = 0.0
        timeLenF_ = 0.0
        timeLenB_ = 0.0

        warmup = iterN <= warmupIter

        # --------------------------------------
        # set up before starting integrating
        # --------------------------------------

        # integration directions: 1=backward, 0=forward
        B = np.floor(random.uniform(low=0.0, high=2.0, size=M)).astype(int)

        # how many backward steps could there possibly be
        nleft = sum(B * (2 ** np.arange(0, M)))

        # I0 is the index of the zeroth state in Hs, Ifs, Ibs
        I0 = nleft
        Ifs[I0] = 0  # not used
        Ibs[I0] = 0  # not used
        cs[I0] = 0  # not used
        lwts[I0] = 0.0

        # endpoints of accepted and proposed trajectory
        a = 0
        b = 0
        maxFint = 0
        maxBint = 0

        # full momentum refresh
        v = random.normal(size=d)

        # endpoints of current orbit (p=forward, m=backward)
        qp = qc
        qm = qc
        vp = v
        vm = v

        # current proposal
        qProp = qc
        qPropLast = qc

        # evaluate at current state
        f0, grad0 = lpFun(qc)

        # gradients at either endpoint
        gp = grad0
        gm = grad0

        # Hamiltonian at initial point
        Hs[I0] = -f0 + 0.5 * sum(v ** 2)

        # index selection-related quantities
        multinomialLscale = Hs[I0]
        WoldSum = 1.0

        lwtSumb = 0.0
        lwtSumf = 0.0

        # reject orbit if numerical problems occur
        forcedReject = False

        # stop if both multinomial bias at both ends are zero
        bothEndsPassive = False
        stopCode = 0

        if (recordOrbitStats):
            orbitMin[:, iterN - 1] = generated(qc)
            orbitMax[:, iterN - 1] = generated(qc)

        # ------------------------------------------
        # NUT iteration loop
        # ------------------------------------------
        for i in range(M):
            # integration direction
            xi = (-1) ** B[i]
            # proposed new endpoints
            at = a + xi * (2 ** i)
            bt = b + xi * (2 ** i)

            numStates = max(bt, b) - min(at, a) + 1

            # more bookkeeping
            expandFurther = True
            qPropLast = qProp
            Lold_ = L_
            indexStatOld_ = indexStat_

            if (i == 0):  # single first integration step required

                HLoc = random.uniform(low=H * (1 - stepSizeRandScale),
                                      high=H * (1 + stepSizeRandScale), size=1)[0]
                orbitLen_ += HLoc
                if (xi == 1):  # forward integration
                    intOut = integrator(qp, vp, gp, Hs[I0], HLoc, xi, lpFun, delta, igrAux)
                    qp = intOut.q
                    vp = intOut.v
                    gp = intOut.grad
                    nevalF += intOut.nEvalF
                    nevalB += intOut.nEvalB
                    Hs[I0 + 1] = -intOut.lp + 0.5 * sum(vp * vp)
                    Ifs[I0 + 1] = intOut.If
                    Ibs[I0 + 1] = intOut.Ib
                    cs[I0 + 1] = intOut.c
                    lwts[I0 + 1] = intOut.lwt
                    if (warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                    maxFint = 1
                    timeLenF_ = HLoc
                    if (not np.isfinite(Hs[I0 + 1])):
                        forcedReject = True
                        stopCode = 999
                        break

                    lwtSumf = lwts[I0 + 1]
                    Wnew = np.exp(-Hs[I0 + 1] + multinomialLscale + lwtSumf)

                    # combined categorical and old/new selection
                    # if(random.uniform()<min(1.0,Wnew)):
                    qProp = qp
                    L_ = 1
                    indexStat_ = timeLenF_

                    if (recordOrbitStats):
                        orbitMin[:, iterN - 1] = np.minimum(orbitMin[:, iterN - 1], generated(qp))
                        orbitMax[:, iterN - 1] = np.maximum(orbitMax[:, iterN - 1], generated(qp))

                else:
                    intOut = integrator(qm, vm, gm, Hs[I0], HLoc, xi, lpFun, delta, igrAux)
                    qm = intOut.q
                    vm = intOut.v
                    gm = intOut.grad
                    nevalF += intOut.nEvalF
                    nevalB += intOut.nEvalB
                    Hs[I0 - 1] = -intOut.lp + 0.5 * sum(vm * vm)
                    Ifs[I0 - 1] = intOut.If
                    Ibs[I0 - 1] = intOut.Ib
                    cs[I0 - 1] = intOut.c
                    lwts[I0 - 1] = intOut.lwt
                    if (warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                    maxBint = -1
                    timeLenB_ = HLoc
                    if (not np.isfinite(Hs[I0 - 1])):
                        forcedReject = True
                        stopCode = 999
                        break
                    lwtSumb = lwts[I0 - 1]
                    Wnew = np.exp(-Hs[I0 - 1] + multinomialLscale + lwtSumb)

                    # if(random.uniform()<min(1.0,Wnew)):
                    qProp = qm
                    L_ = -1
                    indexStat_ = -timeLenB_

                    if (recordOrbitStats):
                        orbitMin[:, iterN - 1] = np.minimum(orbitMin[:, iterN - 1], generated(qm))
                        orbitMax[:, iterN - 1] = np.maximum(orbitMax[:, iterN - 1], generated(qm))

                WoldSum = 1.0
                WnewSum = Wnew

                # done building orbit, now check stop condition:
                # if(stopCondition(qm,vm,qp,vp)):
                #    expandFurther = False
                #    NdoublComputed_ = 1
                #    NdoublSampled_ = 1
                #    stopCode = 1
                #    break

            else:  # more than a single integration step, these require sub-u-turn checks
                # work out which sub-u-turn-checks we are doing
                plan = 0
                if (xi == 1):
                    plan = b + plans[i]
                else:
                    plan = a - plans[i]

                WnewSum = 0.0

                for j in range(len(plan)):  # loop over U-turn-checks
                    # print(plan[j,:])
                    if (abs(plan[j, 0] - plan[j, 1]) == 1):  # new integration steps needed
                        HLoc1 = random.uniform(low=H * (1 - stepSizeRandScale),
                                               high=H * (1 + stepSizeRandScale), size=2)

                        if (xi == -1):  # backward integration
                            i1 = plan[j, 0]
                            intOut = integrator(qm, vm, gm, Hs[I0 + i1 + 1], HLoc1[0], xi, lpFun, delta, igrAux)
                            qm = intOut.q
                            vm = intOut.v
                            gm = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i1] = -intOut.lp + 0.5 * sum(vm * vm)
                            Ifs[I0 + i1] = intOut.If
                            Ibs[I0 + i1] = intOut.Ib
                            cs[I0 + i1] = intOut.c
                            lwts[I0 + i1] = intOut.lwt
                            if (warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxBint = i1
                            timeLenB_ += HLoc1[0]
                            if (not np.isfinite(Hs[I0 + i1])):
                                forcedReject = True
                                stopCode = 999
                                break

                            lwtSumb += lwts[I0 + i1]

                            Wnew = np.exp(-Hs[I0 + i1] + multinomialLscale + lwtSumb)
                            WnewSum += Wnew

                            # online categorical sampling
                            if (WnewSum > __wtSumThresh and random.uniform() < Wnew / WnewSum):
                                qProp = qm
                                L_ = i1
                                indexStat_ = -timeLenB_

                            states.statePush(i1, np.concatenate([qm, vm, gm]))
                            orbitLen_ += HLoc1[0]

                            if (recordOrbitStats):
                                orbitMin[:, iterN - 1] = np.minimum(orbitMin[:, iterN - 1], generated(qm))
                                orbitMax[:, iterN - 1] = np.maximum(orbitMax[:, iterN - 1], generated(qm))

                            qtmp = qm
                            vtmp = vm

                            # second integration step
                            i2 = plan[j, 1]
                            intOut = integrator(qm, vm, gm, Hs[I0 + i2 + 1], HLoc1[1], xi, lpFun, delta, igrAux)
                            qm = intOut.q
                            vm = intOut.v
                            gm = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i2] = -intOut.lp + 0.5 * sum(vm * vm)
                            Ifs[I0 + i2] = intOut.If
                            Ibs[I0 + i2] = intOut.Ib
                            cs[I0 + i2] = intOut.c
                            lwts[I0 + i2] = intOut.lwt
                            if (warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxBint = i2
                            timeLenB_ += HLoc1[1]
                            if (not np.isfinite(Hs[I0 + i2])):
                                forcedReject = True
                                break

                            # online categorical sampling
                            Wnew = np.exp(-Hs[I0 + i2] + multinomialLscale + lwtSumb)
                            WnewSum += Wnew
                            if (WnewSum > __wtSumThresh and random.uniform() < Wnew / WnewSum):
                                qProp = qm
                                L_ = i2
                                indexStat_ = -timeLenB_

                            # store state for future u-turn-checking
                            states.statePush(i2, np.concatenate([qm, vm, gm]))
                            orbitLen_ += HLoc1[1]

                            if (recordOrbitStats):
                                orbitMin[:, iterN - 1] = np.minimum(orbitMin[:, iterN - 1], generated(qm))
                                orbitMax[:, iterN - 1] = np.maximum(orbitMax[:, iterN - 1], generated(qm))

                            # uturn check
                            if (stopCondition(qm, vm, qtmp, vtmp)):
                                expandFurther = False
                                break



                        else:  # forward integration
                            i1 = plan[j, 0]
                            intOut = integrator(qp, vp, gp, Hs[I0 + i1 - 1], HLoc1[0], xi, lpFun, delta, igrAux)
                            qp = intOut.q
                            vp = intOut.v
                            gp = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i1] = -intOut.lp + 0.5 * sum(vp * vp)
                            Ifs[I0 + i1] = intOut.If
                            Ibs[I0 + i1] = intOut.Ib
                            cs[I0 + i1] = intOut.c
                            lwts[I0 + i1] = intOut.lwt
                            if (warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxFint = i1
                            timeLenF_ += HLoc1[0]
                            if (not np.isfinite(Hs[I0 + i1])):
                                forcedReject = True
                                stopCode = 999
                                break

                            lwtSumf += lwts[I0 + i1]

                            # online categorical sampling
                            Wnew = np.exp(-Hs[I0 + i1] + multinomialLscale + lwtSumf)
                            WnewSum += Wnew
                            if (WnewSum > __wtSumThresh and random.uniform() < Wnew / WnewSum):
                                qProp = qp
                                L_ = i1
                                indexStat_ = timeLenF_

                            # store state for future u-turn-checking
                            states.statePush(i1, np.concatenate([qp, vp, gp]))
                            orbitLen_ += HLoc1[0]

                            if (recordOrbitStats):
                                orbitMin[:, iterN - 1] = np.minimum(orbitMin[:, iterN - 1], generated(qp))
                                orbitMax[:, iterN - 1] = np.maximum(orbitMax[:, iterN - 1], generated(qp))

                            qtmp = qp
                            vtmp = vp

                            # second integration step
                            i2 = plan[j, 1]
                            intOut = integrator(qp, vp, gp, Hs[I0 + i2 - 1], HLoc1[1], xi, lpFun, delta, igrAux)
                            qp = intOut.q
                            vp = intOut.v
                            gp = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0 + i2] = -intOut.lp + 0.5 * sum(vp * vp)
                            Ifs[I0 + i2] = intOut.If
                            Ibs[I0 + i2] = intOut.Ib
                            cs[I0 + i2] = intOut.c
                            lwts[I0 + i2] = intOut.lwt
                            if (warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxFint = i2
                            timeLenF_ += HLoc1[1]
                            if (not np.isfinite(Hs[I0 + i2])):
                                forcedReject = True
                                break

                            # multinomial/progressive sampling
                            lwtSumf += lwts[I0 + i2]

                            Wnew = np.exp(-Hs[I0 + i2] + multinomialLscale + lwtSumf)
                            WnewSum += Wnew
                            if (WnewSum > __wtSumThresh and random.uniform() < Wnew / WnewSum):
                                qProp = qp
                                L_ = i2
                                indexStat_ = timeLenF_

                            # store state for future u-turn-checking
                            states.statePush(i2, np.concatenate([qp, vp, gp]))
                            orbitLen_ += HLoc1[1]

                            if (recordOrbitStats):
                                orbitMin[:, iterN - 1] = np.minimum(orbitMin[:, iterN - 1], generated(qp))
                                orbitMax[:, iterN - 1] = np.maximum(orbitMax[:, iterN - 1], generated(qp))

                            if (stopCondition(qtmp, vtmp, qp, vp)):
                                expandFurther = False
                                break
                        # done forward integration
                    else:  # no new integration steps needed, only U-turn checks
                        # delete states not needed further

                        im = min(plan[j, :])
                        ip = max(plan[j, :])
                        # print(" not integrating ",im," ",ip)
                        states.stateDeleteRange(im + 1, ip - 1)
                        statep = states.stateRead(ip)
                        statem = states.stateRead(im)

                        if (stopCondition(statem[0:d],
                                          statem[d:(2 * d)],
                                          statep[0:d],
                                          statep[d:(2 * d)])):
                            expandFurther = False
                            break
                # done loop over j (16)

            if (forcedReject):
                print("numerical problems")
                break

            indexStat_ = indexStat_ / (timeLenF_ + timeLenB_)

            if (not expandFurther):
                # proposed subOrbit had a sub-U-turn
                qProp = qPropLast
                L_ = Lold_
                indexStat_ = indexStatOld_
                NdoublSampled_ = i
                NdoublComputed_ = i + 1
                stopCode = 5
                break

            else:

                # the proposed sub-orbit was found to be free of u-turns
                # now check if proposed state should be from old or new sub-orbit
                # note: reject (and not accept) of proposed state from last doubling

                if (not (random.uniform() < WnewSum / WoldSum)):
                    L_ = Lold_
                    indexStat_ = indexStatOld_
                    qProp = qPropLast

                # proposed suborbit free of U-turns
                # final U-turn check
                joinedCrit = stopCondition(qm, vm, qp, vp)
                # stop simulation if multinomial weights at either end are effectivly zero
                bothEndsPassive = lwtSumb < __logZero + 1.0 and lwtSumf < __logZero + 1.0
                if (joinedCrit or bothEndsPassive):
                    if (joinedCrit):
                        stopCode = 4
                    else:
                        stopCode = -4

                    NdoublSampled_ = i + 1
                    NdoublComputed_ = i + 1
                    orbitLenSam_ = orbitLen_
                    break

            # from now on, it is clear that a new doubling will be attempted
            WoldSum += WnewSum

            orbitLenSam_ = orbitLen_
            NdoublSampled_ = i + 1
            NdoublComputed_ = i + 1
            a = min(a, at)
            b = max(b, bt)
            states.stateReset()

        # done NUTS loop

        qc = qProp

        # -------------------------------------
        # store samples and diagnostics info
        # -------------------------------------

        if (maxBint < 0 and maxFint > 0):
            usedSteps = np.r_[maxBint:0, 1:(maxFint + 1)]
        elif (maxBint < 0):
            usedSteps = np.r_[maxBint:0]
        else:
            usedSteps = np.r_[1:(maxFint + 1)]

        enUsedSteps = np.r_[0, usedSteps]
        orbitEnergyError = np.max(Hs[I0 + enUsedSteps]) - np.min(Hs[I0 + enUsedSteps])

        diagnostics[iterN - 1, 0] = L_
        diagnostics[iterN - 1, 1] = NdoublSampled_
        diagnostics[iterN - 1, 2] = orbitLen_
        diagnostics[iterN - 1, 3] = orbitLenSam_
        diagnostics[iterN - 1, 4] = maxFint
        diagnostics[iterN - 1, 5] = maxBint
        diagnostics[iterN - 1, 6] = nevalF
        diagnostics[iterN - 1, 7] = nevalB
        diagnostics[iterN - 1, 8] = np.min(Ifs[I0 + usedSteps])
        diagnostics[iterN - 1, 9] = np.max(Ifs[I0 + usedSteps])
        diagnostics[iterN - 1, 10] = np.min(lwts[I0 + usedSteps])
        diagnostics[iterN - 1, 11] = np.max(lwts[I0 + usedSteps])
        diagnostics[iterN - 1, 12] = 1.0 * bothEndsPassive
        diagnostics[iterN - 1, 13] = 1.0 * (lwtSumb < __logZero + 1.0 or lwtSumf < __logZero + 1.0)
        diagnostics[iterN - 1, 14] = np.mean(Ifs[I0 + usedSteps] != Ibs[I0 + usedSteps])
        diagnostics[iterN - 1, 15] = H
        diagnostics[iterN - 1, 16] = np.mean(Ifs[I0 + usedSteps] == 0)
        diagnostics[iterN - 1, 17] = orbitEnergyError
        diagnostics[iterN - 1, 18] = delta
        diagnostics[iterN - 1, 19] = 1.0 * stopCode
        diagnostics[iterN - 1, 20] = NdoublComputed_
        diagnostics[iterN - 1, 21] = np.min(cs[I0 + usedSteps])
        diagnostics[iterN - 1, 22] = np.max(cs[I0 + usedSteps])
        diagnostics[iterN - 1, 23] = indexStat_

        samples[:, iterN] = generated(qc)

        # -------------------------------------
        # tuning parameter adaptation
        # -------------------------------------

        if (warmup):

            # tuning of local error threshold delta
            if (adaptDelta): energyErrorInfFacs[iterN - 1] = orbitEnergyError / delta
            if (adaptDelta and iterN > 10):
                delta = adaptDeltaTarget / np.quantile(energyErrorInfFacs[0:iterN],
                                                       adaptDeltaQuantile)

            # tuning of big step size H
            if (adaptH and igrConstQ.npush > 10):
                H = ((delta) ** (1.0 / 3.0)) * np.exp(igrConstQ.quantile())

                # done main iteration loop

    if (recordOrbitStats):
        return (samples, diagnostics, orbitMin, orbitMax)
    else:
        return (samples, diagnostics)