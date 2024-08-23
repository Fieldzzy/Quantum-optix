import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

dim = 5  # Dimension of the photon Hilbert space
oc = 1.69E-1
oq = oc
ol = 1.1*oc
theta = np.pi/6
eta = 0.3
eta_s = 1E-5
Omg = 5E-3 * oc
kai = g = G = 1E-3 * oc
t = 0

# Defining basic operators
a = qt.tensor(qt.destroy(dim), qt.qeye(2), qt.qeye(2))
ad = a.dag()
sigmax = qt.tensor(qt.qeye(dim), qt.sigmax(), qt.qeye(2))
sigmaz = qt.tensor(qt.qeye(dim), qt.sigmaz(), qt.qeye(2))
sigmax_s = qt.tensor(qt.qeye(dim), qt.qeye(2), qt.sigmax())
sigmaz_s = qt.tensor(qt.qeye(dim), qt.qeye(2), qt.sigmaz())
sigmap = np.cos(theta) * sigmax + np.sin(theta) * sigmaz


def threth0(os):
    Hfree = oc*ad*a + oq*sigmaz/2 + os*sigmaz_s/2
    HI = 1j*eta*oc*(ad - a) * sigmap + oc*eta_s*(np.complex128(1j)*(ad - a) + 2*eta*sigmap)*sigmax_s
    Hdrive = Omg*(1j*(a - ad)-2*eta*sigmax)
    H = Hfree + HI

    eigens = H.eigenstates()
    eigenvalues = eigens[0]
    eigenstates = eigens[1]

    # Generate collapse operators
    kaip = Sigp_s = Sigp = qt.qzero([dim, 2, 2])
    for j in range(0, len(eigenvalues)):
        for k in range(0, len(eigenvalues)):
            if eigenvalues[k] > eigenvalues[j]:
                kaip = kaip + ((eigenvalues[k] - eigenvalues[j])/oc) * eigenstates[j].dag() * (1j * (a - ad) - 2*eta*sigmax) * eigenstates[k] * \
                       eigenstates[j] * eigenstates[k].dag()
                Sigp = Sigp + 1j * (eigenvalues[k] - eigenvalues[j])/oq *\
                       eigenstates[j].dag() * sigmax * eigenstates[k] * eigenstates[j] * eigenstates[k].dag()
                Sigp_s = Sigp_s + 1j * (eigenvalues[k] - eigenvalues[j])/os *\
                       eigenstates[j].dag() * sigmax_s * eigenstates[k] * eigenstates[j] * eigenstates[k].dag()

    # Generate Lindblad superoperators:
    Lkaip = qt.lindblad_dissipator(kaip)
    LSigp = qt.lindblad_dissipator(Sigp)
    LSigp_s = qt.lindblad_dissipator(Sigp_s)
    Hund = Hfree + HI
    L0 = -1j * (qt.spre(Hund) - qt.spost(Hund)) + kai * Lkaip + g * LSigp + G * LSigp_s
    Lplus = Lmin = -1j * (qt.spre(Hdrive) - qt.spost(Hdrive)) * 0.5

    # Generate Sns and Tns
    # N = 10  # Maximum number of nonzero Sn
    # S = T = list(np.zeros(N+2))
    # S[N] = T[N] = qt.spre(qt.qzero([dim, 2, 2]))
    # temp = range(0, N)
    # for i in reversed(temp):
    #     Temp_SuperS = L0 - 1j * (i+1) * ol * qt.spre(qt.qeye([dim, 2, 2])) + Lmin * S[i+2]
    #     Temp_SuperT = L0 + 1j * (i+1) * ol * qt.spre(qt.qeye([dim, 2, 2])) + Lplus * T[i+2]
    #     S[i+1] = - Temp_SuperS.inv() * Lplus
    #     T[i+1] = - Temp_SuperT.inv() * Lmin
    #
    # # Initialize initial states
    # rho_init = qt.tensor(qt.fock_dm(dim, 0), qt.fock_dm(2, 0), qt.fock_dm(2, 0))
    # Temp = L0 + Lmin * S[1] + Lplus * T[1]
    # rho0 = Temp.inv() * qt.operator_to_vector(rho_init)
    # rho0 = qt.vector_to_operator(rho0)
    # rho0 = rho0 / rho0.tr()
    # rhoss = rho0

    L00 = -1j * (qt.spre(Hund) - qt.spost(Hund))
    Lcs = [kai * Lkaip, g * LSigp, G * LSigp_s]
    rho_ss = qt.steadystate_floquet(L00, Lcs, Hdrive, w_d=ol)

    Scatter = rho_ss * Sigp_s.dag() * Sigp_s
    Scatter = Scatter.tr()
    return [np.real(Scatter), np.abs(Scatter)]

points = 1000
results = list(np.zeros(points))
oss = np.linspace(0.1 * oc, 1.5 * oc, points)
for i in range(0, points):
    os = oss[i]
    results[i] = threth0(os)
    print(i)

results = np.array(results)
plt.semilogy(oss/oc, results[:, 1])
plt.show(block=True)
